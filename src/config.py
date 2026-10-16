"""Configuração compartilhada; nunca imprime credenciais."""
import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

ROOT = Path(__file__).resolve().parents[1]


def pdf_path():
    load_dotenv(ROOT / ".env", override=False)
    path = Path(os.getenv("PDF_PATH", "document.pdf"))
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise ValueError(f"PDF não encontrado: {path}")
    return path


def document_id(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector_store(path):
    load_dotenv(ROOT / ".env", override=False)
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("Defina OPENAI_API_KEY no environment ou no arquivo .env.")
    return PGVector(
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        connection=os.getenv(
            "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
        ),
        collection_name=f"{os.getenv('COLLECTION_NAME', 'pdf_documents')}_{document_id(path)}",
        use_jsonb=True,
    )
