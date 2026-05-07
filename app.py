import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import os
from google import genai
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import extra_streamlit_components as stx
import time
import streamlit.components.v1 as components

# --- CONFIGURATION & SETUP ---
API_KEY = st.secrets["API_KEY"]
client = genai.Client(api_key=API_KEY)

DB_PATH = "./chroma_db"
COLLECTION_NAME = "ncert_flashcards"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
MAX_QUESTIONS = 5

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AIiQ Minds - Smart Study Assistant",
    page_icon="📚",
    layout="wide"
)

# --- GOOGLE SHEETS SETUP ---
conn = st.connection("gsheets", type=GSheetsConnection)

def get_user_usage():
    return conn.read(worksheet="Sheet1", ttl=0)

def update_user_usage(df, passcode, new_count):
    df.loc[df['passcode'] == passcode, 'usage'] = new_count
    conn.update(worksheet="Sheet1", data=df)

# --- COOKIE MANAGER SETUP ---
cookie_manager = stx.CookieManager(key="auth_manager")
time.sleep(0.1)

# --- 1. THE PERSISTENT LOGIN SCREEN ---
stored_passcode = cookie_manager.get(cookie="readai_passcode")

# Bridge: use temp_passcode to survive the post-login rerun
if "temp_passcode" in st.session_state:
    stored_passcode = st.session_state.temp_passcode

if not stored_passcode:
    st.title("🔒 AIiQ Minds Login")
    passcode_entry = st.text_input("Enter Access Code", type="password")
    
    if st.button("Log In"):
        usage_df = get_user_usage()
        valid_passcodes = usage_df['passcode'].astype(str).values
        
        if passcode_entry in valid_passcodes:
            st.session_state.temp_passcode = passcode_entry
            components.html(f"""
            <script>
                document.cookie = "readai_passcode={passcode_entry}; max-age={30*24*60*60}; path=/;";
                window.parent.location.reload();
            </script>
            """, height=0)
            st.success("Logging in...")
            st.stop()
        else:
            st.error("Invalid Code")
            
    st.stop()

user_code = stored_passcode

# --- 2. USAGE LIMIT CHECK ---
usage_df = get_user_usage()
current_usage = usage_df.loc[usage_df['passcode'] == user_code, 'usage'].values[0]

if current_usage >= MAX_QUESTIONS and user_code != "admin_subham":
    st.error(f"🔒 **Limit reached.** You have used your {MAX_QUESTIONS} free questions. Please contact the admin for more access.")

    if st.button("Logout", use_container_width=True):
        st.session_state.clear()
        components.html("""
        <script>
            document.cookie = "readai_passcode=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
            window.parent.location.reload();
        </script>
        """, height=0)
        st.stop()
        
    st.stop()


# --- CACHED RESOURCES ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

@st.cache_resource
def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    return chroma_client.get_collection(name=COLLECTION_NAME)

embedding_model = load_embedding_model()
collection = get_chroma_collection()

# --- DATA RETRIEVAL HELPERS ---
def get_knowledge_chunk(chapter_num, label):
    results = collection.get(
        where={
            "$and": [
                {"chapter_number": chapter_num},
                {"semantic_label": label}
            ]
        },
        include=["documents"]
    )
    if results and results["documents"]:
        return results["documents"][0]
    return None

def perform_vector_search(query, chapter_num, top_k=4):
    query_nomic = f"search_query: {query}"
    query_vector = embedding_model.encode(query_nomic).tolist()
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where={"chapter_number": chapter_num},
        include=["documents", "metadatas"]
    )
    return results

def generate_rag_response(user_query, context_chunks):
    context_text = "\n\n---\n\n".join(context_chunks)
    
    prompt = f"""
    You are an expert, friendly tutor helping a 7th-grade student.
    Answer the student's question using ONLY the provided textbook context below.
    If the answer is not in the context, politely say that you don't have that information in the current chapter.
    
    Context from textbook:
    {context_text}
    
    Student Question: {user_query}
    """
    
    try:
        response = client.models.generate_content(
            model="gemma-4-31b-it", 
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Sorry, I encountered an error connecting to my brain. Details: {e}"

# --- UI BUILDER ---

# 1. Sidebar
with st.sidebar:
    st.title("📚 AIiQ Minds")
    st.markdown(f"**User:** `{user_code}`")
    
    if st.button("Logout", use_container_width=True):
        st.session_state.clear()
        components.html("""
        <script>
            document.cookie = "readai_passcode=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
            window.parent.location.reload();
        </script>
        """, height=0)
        st.stop()
    
    st.divider()
    
    selected_chapter = st.selectbox("Select Chapter", options=[1], format_func=lambda x: f"Chapter {x}: The Story of Indian Farming")
    
    st.divider()
    st.subheader("📄 Original Textbook")
    
    pdf_url = "https://drive.google.com/file/d/1-xLuSJ9u3a80xQxMMtsYPNUzhER_llsZ/view?preview"
    st.link_button("↗ Open PDF in New Tab", pdf_url, use_container_width=True)
    
    st.divider()
    
    st.subheader("🎯 Key Topics")
    topics_text = get_knowledge_chunk(selected_chapter, "Key Topics")
    if topics_text:
        st.markdown(topics_text)
    else:
        st.caption("No topics found for this chapter.")

# 2. Main Screen
st.title("Chapter Study Guide")

short_summary = get_knowledge_chunk(selected_chapter, "Short Summary")
detailed_summary = get_knowledge_chunk(selected_chapter, "Detailed Summary")

if short_summary or detailed_summary:
    with st.expander("📖 View Chapter Summary", expanded=False):
        if short_summary:
            st.markdown(f"**The Gist:** {short_summary}")
        if detailed_summary:
            st.markdown("---")
            st.markdown(detailed_summary)

st.divider()

# 3. Chat Interface
st.subheader("💬 Ask Questions")

if user_code != "admin_subham":
    st.caption(f"*(You have {MAX_QUESTIONS - current_usage} questions remaining)*")
else:
    st.caption("*(Admin mode: Unlimited questions)*")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the chapter..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the textbook..."):
            search_results = perform_vector_search(prompt, selected_chapter, top_k=4)
            retrieved_docs = search_results["documents"][0] if search_results["documents"] else []
            retrieved_metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []
            
            if retrieved_docs:
                enriched_chunks = []
                for doc, meta in zip(retrieved_docs, retrieved_metadatas):
                    chapter_name = meta.get("chapter_name", "Unknown Chapter")
                    page_num = meta.get("page_number", "Unknown")
                    enriched_chunks.append(f"[Source: Chapter '{chapter_name}', Page {page_num}]\n{doc}")
                
                answer = generate_rag_response(prompt, enriched_chunks)
            else:
                answer = "I couldn't find anything about that in this chapter."
            
            st.markdown(answer)
            
            with st.expander("Sources used"):
                for i, doc in enumerate(retrieved_docs):
                    page_num = search_results["metadatas"][0][i].get("page_number", "?")
                    st.caption(f"**Page {page_num}:** {doc[:150]}...")
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    if user_code != "admin_subham":
        update_user_usage(usage_df, user_code, current_usage + 1)
        st.rerun()