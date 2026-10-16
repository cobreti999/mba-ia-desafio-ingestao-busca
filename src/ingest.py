"""Ingestão idempotente: python src/ingest.py."""
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import pdf_path, vector_store


def load_chunks(path):
    pages = PyPDFLoader(str(path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not chunks:
        raise ValueError("PDF sem texto extraível. PDFs digitalizados precisam de OCR prévio.")
    for chunk in chunks:
        chunk.metadata["source"] = path.name
    return chunks


def ingest(path=None):
    path = path or pdf_path()
    chunks = load_chunks(path)
    store = vector_store(path)
    # IDs globais incluem documento e configuração do splitter.
    ids = [f"{store.collection_name}:1000:150:{i}" for i in range(len(chunks))]
    inserted = 0
    for start in range(0, len(chunks), 100):
        batch_ids = ids[start:start + 100]
        existing = {doc.id for doc in store.get_by_ids(batch_ids)}
        missing = [(doc, id_) for doc, id_ in zip(chunks[start:start + 100], batch_ids)
                   if id_ not in existing]
        if missing:
            store.add_documents([doc for doc, _ in missing], ids=[id_ for _, id_ in missing])
            inserted += len(missing)
    return inserted, len(chunks)


def main():
    try:
        inserted, total = ingest()
        print(f"Ingestão concluída: {inserted} novos chunks; {total} chunks no documento.")
        return 0
    except ValueError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
    except Exception:
        print("Falha na ingestão. Confira o PDF, o Docker, DATABASE_URL e o acesso à OpenAI.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
