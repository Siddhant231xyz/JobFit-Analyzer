import getpass
import os
import glob
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain import hub

# --------------------------------------------------------------------------------
# 1. Prompt user for API key if not already set
# --------------------------------------------------------------------------------
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# --------------------------------------------------------------------------------
# 2. Initialize LLM and Embeddings
# --------------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# --------------------------------------------------------------------------------
# 3. Function to load and process PDFs
# --------------------------------------------------------------------------------
def load_and_process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

# --------------------------------------------------------------------------------
# 4. Build a vector store for the JD
# --------------------------------------------------------------------------------
jd_path = os.path.join("C:/Users/admin/Desktop/JD.pdf")
jd_docs = load_and_process_pdf(jd_path)

jd_vector_store = InMemoryVectorStore(embeddings)
jd_vector_store.add_documents(documents=jd_docs, metadata={"doc_type": "JD"})

# Convert JD docs to text for later embedding comparison with resumes
jd_text = "\n\n".join(doc.page_content for doc in jd_docs)
# Precompute the JD embedding once
jd_embedding = embeddings.embed_query(jd_text)

# --------------------------------------------------------------------------------
# 5. Build separate vector stores for each resume
# --------------------------------------------------------------------------------
# Create a dictionary: resume_name -> its vector store
resume_vector_stores = {}  # Each value is an InMemoryVectorStore
resume_files = glob.glob(os.path.join("C:/Users/admin/Desktop/resume", "*.pdf"))

for path in resume_files:
    resume_name = os.path.basename(path)
    splits = load_and_process_pdf(path)
    store = InMemoryVectorStore(embeddings)
    # Add each chunk individually with metadata indicating the source
    for chunk in splits:
        store.add_documents(
            documents=[chunk],
            metadata={"doc_type": "resume", "source": resume_name}
        )
    resume_vector_stores[resume_name] = store

print("Loaded JD and individual resume vector stores.")

# --------------------------------------------------------------------------------
# 6. Compute a score for each resume using cosine similarity
# --------------------------------------------------------------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def get_resume_score(resume_store, jd_embedding, top_k=1):
    """
    Retrieve up to top_k chunks from this resume using similarity_search_by_vector,
    then compute the cosine similarity manually. Returns the best score found.
    """
    results = resume_store.similarity_search_by_vector(jd_embedding, top_k=top_k)
    best_score = 0.0
    for doc in results:
        # Compute the embedding for the chunk
        doc_embedding = embeddings.embed_query(doc.page_content)
        score = cosine_similarity(jd_embedding, doc_embedding)
        if score > best_score:
            best_score = score
    return best_score

# Compute scores for all resumes
resume_scores = {}
for name, store in resume_vector_stores.items():
    score = get_resume_score(store, jd_embedding)
    resume_scores[name] = score

# Ask user how many top resumes they want to see
try:
    top_n = int(input("Enter the number of top resumes to display (e.g., 3, 5, 10): "))
except ValueError:
    top_n = 5

# Sort resumes by score (higher score = better match)
sorted_resumes = sorted(resume_scores.items(), key=lambda x: x[1], reverse=True)
top_resume_list = sorted_resumes[:top_n]

print("\nTop resumes based on similarity to the JD:")
for idx, (name, score) in enumerate(top_resume_list, 1):
    print(f"{idx}. {name}  (Score: {score:.4f})")

# --------------------------------------------------------------------------------
# 7. Functions for Q&A on JD, combined resumes, and individual resume queries
# --------------------------------------------------------------------------------
# Use the "rag-prompt" from the hub (change as desired)
prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List  # List of document objects
    answer: str

def query_with_context(question: str, context_docs: List) -> str:
    """Generic function to construct a prompt with retrieved docs and query the LLM."""
    docs_content = "\n\n".join(doc.page_content for doc in context_docs)
    messages = prompt.invoke({"question": question, "context": docs_content})
    response = llm.invoke(messages)
    return response.content

def query_jd(question: str) -> str:
    retrieved_docs = jd_vector_store.similarity_search(question, k=3)
    return query_with_context(question, retrieved_docs)

def query_combined_resumes(question: str, num_resumes: int) -> str:
    """Query across the top 'num_resumes' resumes."""
    combined_docs = []
    # Use the sorted results (global variable sorted_resumes) to combine docs
    for name, score in sorted_resumes[:num_resumes]:
        store = resume_vector_stores[name]
        docs = store.similarity_search(question, k=3)
        combined_docs.extend(docs)
    return query_with_context(question, combined_docs)

def query_individual_resume(question: str, resume_name: str) -> str:
    """Query an individual resume’s vector store."""
    if resume_name not in resume_vector_stores:
        return f"Resume {resume_name} was not found."
    store = resume_vector_stores[resume_name]
    retrieved_docs = store.similarity_search(question, k=3)
    return query_with_context(question, retrieved_docs)

# --------------------------------------------------------------------------------
# 8. Interactive Loop Menu
# --------------------------------------------------------------------------------
def print_menu():
    print("\nPlease select an option:")
    print("1. Ask a question about the JD.")
    print("2. Ask a question about combined top resumes.")
    print("3. Ask a question about an individual resume (from the top list).")
    print("4. Re-display the top resume list with scores.")
    print("5. Exit.")

while True:
    print_menu()
    choice = input("Enter your choice (1-5): ").strip()
    if choice == "5":
        print("Exiting the application.")
        break

    if choice == "1":
        q = input("Enter your question about the JD: ")
        answer = query_jd(q)
        print(f"\nAnswer (JD): {answer}")

    elif choice == "2":
        try:
            n_top = int(input("Enter the number of top resumes you want to query (e.g., 3, 5, 10): "))
        except ValueError:
            n_top = top_n
        q = input("Enter your question for the combined top resumes: ")
        answer = query_combined_resumes(q, n_top)
        print(f"\nAnswer (Combined Resumes): {answer}")

    elif choice == "3":
        print("\nSelect one of the top resumes by number:")
        for idx, (name, score) in enumerate(top_resume_list, 1):
            print(f"{idx}. {name}  (Score: {score:.4f})")
        try:
            sel = int(input("Enter the index of the resume: "))
            if not (1 <= sel <= len(top_resume_list)):
                print("Invalid index. Please try again.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        selected_resume = top_resume_list[sel-1][0]
        print(f"You selected: {selected_resume}")
        q = input(f"Enter your question for the resume '{selected_resume}': ")
        answer = query_individual_resume(q, selected_resume)
        print(f"\nAnswer (Individual Resume {selected_resume}): {answer}")

    elif choice == "4":
        print("\nCurrent top resumes based on similarity to the JD:")
        for idx, (name, score) in enumerate(top_resume_list, 1):
            print(f"{idx}. {name}  (Score: {score:.4f})")
    else:
        print("Invalid choice. Please select an option from 1 to 5.")
