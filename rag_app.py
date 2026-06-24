import os
import shutil
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFaceEmbeddings
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate
)

from langchain_core.output_parsers import (
    StrOutputParser
)

from langchain_core.runnables import (
    RunnablePassthrough
)

from langchain_classic.retrievers import (
    MultiQueryRetriever
)

# ------------------------
# Config
# ------------------------

load_dotenv()

st.set_page_config(
    page_title="Multi PDF RAG",
    page_icon="📚"
)

st.title("📚 Multi PDF RAG Chatbot")

DB_DIR = "./chroma_db"
UPLOAD_DIR = "./uploaded_pdfs"

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------------
# Session State
# ------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "all_chats" not in st.session_state:
    st.session_state.all_chats = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# ------------------------
# Models
# ------------------------

@st.cache_resource
def load_models():

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.3-70B-Instruct",
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.05,
        max_new_tokens=512
    )

    model = ChatHuggingFace(llm=llm)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return model, embeddings

model, embeddings = load_models()

# ------------------------
# Build RAG
# ------------------------

def build_rag(pdf_paths):

    all_docs = []

    for pdf in pdf_paths:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="multi_pdf_rag",
        persist_directory=DB_DIR
    )

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
Generate five alternative versions
of the user's question.

Question:
{question}
"""
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        model,
        prompt=query_prompt
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful PDF assistant.

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "history": lambda x: x["history"]
        }
        | prompt
        | model
        | StrOutputParser()
    )

    st.session_state.rag_chain = rag_chain

# ------------------------
# Sidebar
# ------------------------

with st.sidebar:

    if st.button("➕ New Chat"):
        if st.session_state.messages:
            st.session_state.all_chats.append(
                st.session_state.messages.copy()
            )
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.header("📄 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:

        file_names = [f.name for f in uploaded_files]
        already_indexed = st.session_state.get("indexed_files")

        if file_names != already_indexed:

            with st.status("Processing PDFs...", expanded=True) as status:

                pdf_paths = []

                for file in uploaded_files:

                    dest = os.path.join(
                        UPLOAD_DIR,
                        file.name
                    )

                    with open(dest, "wb") as f:
                        f.write(file.getbuffer())

                    pdf_paths.append(dest)

                st.write("🔍 Indexing chunks...")

                build_rag(pdf_paths)

                st.write("✅ Ready!")

                status.update(
                    label="PDFs loaded successfully!",
                    state="complete"
                )

                st.session_state.indexed_files = file_names

        else:

            for f_name in file_names:
                st.write(f"✅ {f_name}")

            st.caption(
                "Already indexed — ask away!"
            )

    st.divider()

    if st.session_state.all_chats:

        st.subheader("🕘 Past Chats")

        for i, chat in enumerate(
            reversed(st.session_state.all_chats)
        ):

            first_msg = next(
                (
                    m["content"]
                    for m in chat
                    if m["role"] == "user"
                ),
                "Chat"
            )

            label = (
                first_msg[:40] + "..."
                if len(first_msg) > 40
                else first_msg
            )

            if st.button(
                f"💬 {label}",
                key=f"past_chat_{i}"
            ):

                if st.session_state.messages:
                    st.session_state.all_chats.append(
                        st.session_state.messages.copy()
                    )

                st.session_state.messages = chat
                st.rerun()

# ------------------------
# Main Chat
# ------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.messages:

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

question = st.chat_input(
    "Ask about your PDFs..."
)

if question:

    if st.session_state.rag_chain is None:

        st.error(
            "Please upload PDF documents first."
        )

        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    history = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in st.session_state.messages
    ])

    with st.spinner("Thinking..."):

        answer = st.session_state.rag_chain.invoke(
            {
                "question": question,
                "history": history
            }
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.rerun()