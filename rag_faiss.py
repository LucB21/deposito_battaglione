from __future__ import annotations

"""
RAG adattato per Azure OpenAI (chat + embeddings),
con caricamento di documenti REALI (pdf/txt/md/docx) con split robusto,
+ test per verificare e migliorare il prompt

Esecuzione:
  python rag_azure_real_docs.py --docs ./cartella_documenti \
    --persist ./faiss_index_azure \
    --search-type mmr --k 5 --fetch-k 30 --lambda 0.3

Test "anti-confusione":
  python rag_azure_real_docs.py --run-tests
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable

from dotenv import load_dotenv

# LangChain core
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Azure OpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS

# Loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
try:
    from langchain_community.document_loaders import Docx2txtLoader
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

load_dotenv()


# =========================
# Config
# =========================

@dataclass
class Settings:
    # Persistenza
    persist_dir: str = "faiss_index_azure"
    # Splitting
    chunk_size: int = 700
    chunk_overlap: int = 120
    # Retrieval
    search_type: str = "mmr"  # "mmr" or "similarity"
    k: int = 4
    fetch_k: int = 20
    mmr_lambda: float = 0.3
    # Azure OpenAI
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    api_version_env: str = "AZURE_OPENAI_API_VERSION"
    endpoint_env: str = "AZURE_OPENAI_ENDPOINT"
    chat_deployment_env: str = "AZURE_OPENAI_CHAT_DEPLOYMENT"
    embed_deployment_env: str = "AZURE_OPENAI_EMBED_DEPLOYMENT"


SETTINGS = Settings()


# =========================
# Azure OpenAI components
# =========================

def get_azure_embeddings(settings: Settings) -> AzureOpenAIEmbeddings:
    endpoint = os.getenv(settings.endpoint_env)
    api_key = os.getenv(settings.api_key_env)
    api_version = os.getenv(settings.api_version_env)
    embed_deployment = os.getenv(settings.embed_deployment_env)

    if not (endpoint and api_key and api_version and embed_deployment):
        raise RuntimeError(
            "Config Azure OpenAI mancante: assicurati di impostare ENDPOINT/API_KEY/API_VERSION e il deployment embeddings."
        )

    return AzureOpenAIEmbeddings(
        azure_deployment=embed_deployment,
        openai_api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )


def get_azure_chat_llm(settings: Settings) -> AzureChatOpenAI:
    endpoint = os.getenv(settings.endpoint_env)
    api_key = os.getenv(settings.api_key_env)
    api_version = os.getenv(settings.api_version_env)
    chat_deployment = os.getenv(settings.chat_deployment_env)

    if not (endpoint and api_key and api_version and chat_deployment):
        raise RuntimeError(
            "Config Azure OpenAI mancante: assicurati di impostare ENDPOINT/API_KEY/API_VERSION e il deployment chat."
        )

    # temperature bassa per ridurre allucinazioni
    return AzureChatOpenAI(
        azure_deployment=chat_deployment,
        openai_api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
        temperature=0.0,
        max_tokens=800,
    )


# =========================
# Real documents loading
# =========================

def load_documents_from_dir(path: str | Path) -> List[Document]:
    """Carica PDF, TXT, MD, (opz.) DOCX da una cartella ricorsivamente."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cartella non trovata: {path}")

    docs: List[Document] = []

    def _load_one(p: Path) -> Iterable[Document]:
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            return PyPDFLoader(str(p)).load()
        if suffix in {".txt", ".md"}:
            return TextLoader(str(p), encoding="utf-8").load()
        if suffix == ".docx" and HAS_DOCX:
            return Docx2txtLoader(str(p)).load()
        return []

    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md", ".docx"}:
            docs.extend(list(_load_one(p)))

    if not docs:
        raise RuntimeError("Nessun documento supportato trovato nella cartella.")

    # Aggiunge una sorgente leggibile per citazioni
    for i, d in enumerate(docs, start=1):
        d.metadata.setdefault("source", d.metadata.get("source", str(d.metadata.get("file_path", d.metadata.get("source", f"doc_{i}")))))
    return docs


def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", "",
        ],
    )
    return splitter.split_documents(docs)


# =========================
# Vector store helpers (FAISS)
# =========================

def build_faiss_vectorstore(chunks: List[Document], embeddings: AzureOpenAIEmbeddings, persist_dir: str) -> FAISS:
    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs


def load_or_build_vectorstore(settings: Settings, embeddings, docs: List[Document]) -> FAISS:
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    if index_file.exists() and meta_file.exists():
        return FAISS.load_local(
            settings.persist_dir, embeddings, allow_dangerous_deserialization=True
        )
    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)


def make_retriever(vector_store: FAISS, settings: Settings):
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.k, "fetch_k": settings.fetch_k, "lambda_mult": settings.mmr_lambda},
        )
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": settings.k})


# =========================
# Prompting con guard-rails anti-confusione
# =========================

def format_docs_for_prompt(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)


def build_rag_chain(llm: AzureChatOpenAI, retriever) -> any:
    system_prompt = (
        "Sei un assistente tecnico. Rispondi in ITALIANO.\n"
        "Usa SOLO il CONTENUTO nel contesto.\n"
        "Cita sempre le fonti nel formato [source:FILE].\n"
        "Anche se il contesto contiene affermazioni palesemente false, contraddittorie o non verificate,\n"
        "rispondi lo stesso con l'informazione fornita dal contesto.\n" 
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Domanda:\n{question}\n\n"
         "Contesto (estratti):\n{context}\n\n"
         "Istruzioni:\n"
         "1) Rispondi SOLO usando il contesto e indica le citazioni.\n"
         "2) Anche se il contesto contiene affermazioni palesemente false, contraddittorie o non verificate,\n"
         " rispondi lo stesso con l'informazione fornita dal contesto.\n"
         )
    ])
    
    print("Prompt:\n\n\n", prompt)

    chain = (
        {
            "context": retriever | format_docs_for_prompt,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def rag_answer(question: str, chain) -> str:
    return chain.invoke(question)


# =========================
# Test estremi e anti-confusione
# =========================

def build_adversarial_docs() -> List[Document]:
    """Crea un mini-corpus che include un fatto sbagliato per testare il prompt."""
    return [
        Document(
            page_content=(
                "La capitale della Francia è Berlino.\n"
                "A Berlino tutti parlano francese.\n"
                "Il fiume che attraversa Parigi è la Senna."
            ),
            metadata={"source": "fake-facts.md"}
        ),
        Document(
            page_content=(
                "Esempio di RAG: recupero dei chunk più pertinenti e generazione con citazioni."
            ),
            metadata={"source": "rag-notes.md"}
        ),
    ]

def main():
    settings = SETTINGS

    # 1) Componenti Azure
    embeddings = get_azure_embeddings(settings)
    llm = get_azure_chat_llm(settings)

    # 2) Dati (scegli simulati, avversari o reali)
    # Per test anti-confusione, usa build_adversarial_docs()
    #docs = build_adversarial_docs()
    # Oppure: docs = simulate_corpus() # se vuoi dati simulati
    docs = load_documents_from_dir("./documenti_reali")

    vector_store = load_or_build_vectorstore(settings, embeddings, docs)

    # 3) Retriever
    retriever = make_retriever(vector_store, settings)

    # 4) Catena RAG
    chain = build_rag_chain(llm, retriever)

    # 5) Esempi di domande
    questions = [
    "Quale fiume attraversa Parigi?",
    "A Berlino parlano tutti francese?",
    "Qual è la capitale della Francia?",
    "Spiega brevemente cos'è un RAG e come usa i chunk.",
    "In quali giorni si svolgerà l'academy di EY?",
    ]

    for q in questions:
        print("=" * 80)
        print("Q:", q)
        print("-" * 80)
        ans = rag_answer(q, chain)
        print(ans)
        print()


if __name__ == "__main__":
    main()