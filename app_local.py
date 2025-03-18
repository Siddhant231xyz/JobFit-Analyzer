import os
import glob
import numpy as np
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub

# =============================================================================
# Helper Functions and Initialization
# =============================================================================

# --- Cosine Similarity Function ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# --- PDF Loading and Splitting ---
def load_and_process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

# --- Query Function using RAG prompt ---
def query_with_context(question: str, context_docs: list) -> str:
    """Build context from retrieved docs and query the LLM."""
    docs_content = "\n\n".join(doc.page_content for doc in context_docs)
    messages = st.session_state.prompt.invoke({"question": question, "context": docs_content})
    response = st.session_state.llm.invoke(messages)
    return response.content

# --- Query JD ---
def query_jd(question: str) -> str:
    retrieved_docs = st.session_state.jd_vector_store.similarity_search(question, k=3)
    return query_with_context(question, retrieved_docs)

# --- Query Combined Resumes ---
def query_combined_resumes(question: str, num_resumes: int) -> str:
    combined_docs = []
    for name, score in st.session_state.sorted_resumes[:num_resumes]:
        store = st.session_state.resume_vector_stores[name]
        docs = store.similarity_search(question, k=3)
        combined_docs.extend(docs)
    return query_with_context(question, combined_docs)

# --- Query Individual Resume ---
def query_individual_resume(question: str, resume_name: str) -> str:
    if resume_name not in st.session_state.resume_vector_stores:
        return f"Resume {resume_name} was not found."
    store = st.session_state.resume_vector_stores[resume_name]
    retrieved_docs = store.similarity_search(question, k=3)
    return query_with_context(question, retrieved_docs)

# --- Compute Resume Score (Manually using cosine similarity) ---
def get_resume_score(resume_store, jd_embedding, top_k=1):
    results = resume_store.similarity_search_by_vector(jd_embedding, top_k=top_k)
    best_score = 0.0
    for doc in results:
        doc_embedding = st.session_state.embeddings.embed_query(doc.page_content)
        score = cosine_similarity(jd_embedding, doc_embedding)
        if score > best_score:
            best_score = score
    return best_score

# --- Update Resume Scores When the JD Changes ---
def update_resume_scores():
    resume_scores = {}
    for name, store in st.session_state.resume_vector_stores.items():
        score = get_resume_score(store, st.session_state.jd_embedding)
        resume_scores[name] = score
    st.session_state.resume_scores = resume_scores
    sorted_resumes = sorted(resume_scores.items(), key=lambda x: x[1], reverse=True)
    st.session_state.sorted_resumes = sorted_resumes

# =============================================================================
# Session State Initialization (only once)
# =============================================================================
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.write("Initializing… please wait (this may take a minute).")
    
    # --- Set API Key (you can also set it via environment variable) ---
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = st.text_input("Enter API key for OpenAI:", type="password")
    
    # --- Initialize LLM and Embeddings ---
    llm = ChatOpenAI(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    st.session_state.llm = llm
    st.session_state.embeddings = embeddings
    
    # --- Load the RAG prompt from hub ---
    prompt = hub.pull("rlm/rag-prompt")
    st.session_state.prompt = prompt
    
    # --- Load Resumes (from disk) ---
    resume_folder = "C:/Users/admin/Desktop/resume"
    resume_files = glob.glob(os.path.join(resume_folder, "*.pdf"))
    resume_vector_stores = {}
    for path in resume_files:
        resume_name = os.path.basename(path)
        splits = load_and_process_pdf(path)
        store = InMemoryVectorStore(embeddings)
        for chunk in splits:
            store.add_documents(
                documents=[chunk],
                metadata={"doc_type": "resume", "source": resume_name}
            )
        resume_vector_stores[resume_name] = store
    st.session_state.resume_vector_stores = resume_vector_stores
    
    # --- Initialize JD placeholders ---
    st.session_state.jd_vector_store = None
    st.session_state.jd_embedding = None
    st.session_state.resume_scores = {}
    st.session_state.sorted_resumes = []
    st.session_state.jd_file_name = None  # to track the currently processed JD file
    
    st.success("Resumes loaded successfully!")

# =============================================================================
# Upload a New Job Description (JD)
# =============================================================================
st.sidebar.markdown("## Upload Job Description")
uploaded_jd = st.sidebar.file_uploader("Upload a JD PDF", type=["pdf"])
if uploaded_jd is not None:
    # Process the JD only if the file name is new (or if it hasn't been processed yet)
    if st.session_state.jd_file_name != uploaded_jd.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_jd.read())
            tmp_file_path = tmp_file.name
        # Process the uploaded file
        jd_docs = load_and_process_pdf(tmp_file_path)
        jd_vector_store = InMemoryVectorStore(st.session_state.embeddings)
        jd_vector_store.add_documents(documents=jd_docs, metadata={"doc_type": "JD"})
        st.session_state.jd_vector_store = jd_vector_store

        jd_text = "\n\n".join(doc.page_content for doc in jd_docs)
        jd_embedding = st.session_state.embeddings.embed_query(jd_text)
        st.session_state.jd_embedding = jd_embedding

        # Update resume scores based on the new JD embedding
        update_resume_scores()

        st.session_state.jd_file_name = uploaded_jd.name  # store the processed file's name
        st.sidebar.success("Uploaded JD processed successfully!")
    else:
        st.sidebar.info("JD file already processed.")

# =============================================================================
# Streamlit UI (only active if a JD has been processed)
# =============================================================================
st.title("JD and Resume Q&A")
st.markdown("""
This application supports three types of queries:
- **Query JD:** Ask questions solely about the Job Description.
- **Combined Resumes:** Ask questions that retrieve answers from the top N resumes (sorted by similarity to the current JD).
- **Individual Resume:** Pick one resume from the top list and ask detailed questions.
""")

if st.session_state.jd_embedding is None:
    st.warning("Please upload a Job Description PDF to begin.")
else:
    # --- Sidebar: Select Query Type and Options ---
    query_type = st.sidebar.radio("Select Query Type", ("JD", "Combined Resumes", "Individual Resume"))

    if query_type == "Combined Resumes":
        num_resumes = st.sidebar.number_input("Enter number of top resumes to query", min_value=1, value=5)
    else:
        num_resumes = None

    if query_type == "Individual Resume":
        # Display the top resumes (e.g. top 10) sorted by score for selection.
        top_display_count = 10
        top_resumes = st.session_state.sorted_resumes[:top_display_count]
        resume_options = [f"{idx+1}. {name} (Score: {score:.4f})" 
                          for idx, (name, score) in enumerate(top_resumes)]
        selected_option = st.sidebar.selectbox("Select a resume", resume_options)
        # Extract the resume name from the selected option
        selected_resume = selected_option.split(". ")[1].split(" (Score")[0]
    else:
        selected_resume = None

    user_question = st.text_input("Enter your question:")

    if st.button("Submit Question"):
        if not user_question:
            st.warning("Please enter a question.")
        else:
            if query_type == "JD":
                answer = query_jd(user_question)
                st.markdown("### Answer (JD)")
                st.write(answer)
            elif query_type == "Combined Resumes":
                answer = query_combined_resumes(user_question, int(num_resumes))
                st.markdown("### Answer (Combined Resumes)")
                st.write(answer)
            elif query_type == "Individual Resume":
                answer = query_individual_resume(user_question, selected_resume)
                st.markdown(f"### Answer (Individual Resume: {selected_resume})")
                st.write(answer)

    # --- Display the current top resume list with scores ---
    with st.expander("Show Top Resumes List and Scores"):
        st.write("The top resumes (sorted by similarity to the current JD):")
        for idx, (name, score) in enumerate(st.session_state.sorted_resumes, 1):
            st.write(f"{idx}. {name} (Score: {score:.4f})")
