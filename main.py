import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil

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

from dotenv import load_dotenv

load_dotenv()

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")

app = FastAPI()

DB_DIR = "./chroma_db"

# Global chain
rag_chain = None

# ------------------------
# LLM
# ------------------------

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    temperature=0.1,
    top_p=0.9,
    repetition_penalty=1.05,
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)


# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)


# ------------------------
# Request Schemas
# ------------------------

class QuestionRequest(BaseModel):
    question: str
    chat_history: str


# ------------------------
# Build RAG
# ------------------------

def build_rag(pdf_paths):

    global rag_chain

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


# ------------------------
# API
# ------------------------

UPLOAD_DIR = "./uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
def upload_pdfs(files: list[UploadFile] = File(...)):
    pdf_paths = []
    for file in files:
        dest = os.path.join(UPLOAD_DIR, file.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        pdf_paths.append(dest)
    build_rag(pdf_paths)
    return JSONResponse({"message": f"{len(pdf_paths)} PDF(s) uploaded and indexed."})

@app.post("/ask")
def ask_question(req: QuestionRequest):

    answer = rag_chain.invoke(
        {
            "question": req.question,
            "history": req.chat_history
        }
    )

    return {"answer": answer}
