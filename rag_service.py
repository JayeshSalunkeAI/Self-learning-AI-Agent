import hashlib
import uuid
from datetime import datetime

import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config import (
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    QDRANT_HOST,
    QDRANT_PORT,
    RAG_COLLECTION,
)
from database import (
    delete_document_metadata,
    document_exists,
    save_document,
)


qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

ollama_client = ollama.Client(host=OLLAMA_BASE_URL)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
)


def ensure_rag_collection() -> None:
    if qdrant.collection_exists(collection_name=RAG_COLLECTION):
        return

    qdrant.create_collection(
        collection_name=RAG_COLLECTION,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE,
        ),
    )


def create_embedding(text: str) -> list[float]:
    response = ollama_client.embeddings(
        model=EMBED_MODEL,
        prompt=text,
    )
    return response["embedding"]


def ingest_document(
    user_id: str,
    file_name: str,
    file_bytes: bytes,
    extracted_text: str,
) -> tuple[bool, str]:
    if not extracted_text or len(extracted_text.strip()) < 30:
        return False, "No readable text was found. The PDF may be scanned or empty."

    ensure_rag_collection()

    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if document_exists(user_id, file_hash):
        return False, f"'{file_name}' was already uploaded for this user."

    document_id = str(uuid.uuid4())
    chunks = splitter.split_text(extracted_text)

    if not chunks:
        return False, "No text chunks were created from this document."

    points = []

    for chunk_index, chunk in enumerate(chunks):
        embedding = create_embedding(chunk)

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "user_id": user_id,
                    "document_id": document_id,
                    "file_name": file_name,
                    "chunk_index": chunk_index,
                    "content": chunk,
                    "created_at": datetime.now().isoformat(),
                },
            )
        )

    qdrant.upsert(
        collection_name=RAG_COLLECTION,
        points=points,
        wait=True,
    )

    save_document(
        document_id=document_id,
        user_id=user_id,
        file_name=file_name,
        file_hash=file_hash,
        chunk_count=len(chunks),
    )

    return True, f"Added '{file_name}' with {len(chunks)} knowledge chunks."


def retrieve_rag(
    user_id: str,
    query: str,
    limit: int = 4,
) -> list[dict]:
    ensure_rag_collection()

    query_vector = create_embedding(query)

    result = qdrant.query_points(
        collection_name=RAG_COLLECTION,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
    )

    documents = []

    for point in result.points:
        payload = point.payload or {}

        documents.append(
            {
                "content": payload.get("content", ""),
                "file_name": payload.get("file_name", "Unknown file"),
                "document_id": payload.get("document_id", ""),
                "score": round(point.score, 3),
            }
        )

    return documents


def delete_document(user_id: str, document_id: str) -> None:
    ensure_rag_collection()

    qdrant.delete(
        collection_name=RAG_COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                ),
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                ),
            ]
        ),
        wait=True,
    )

    delete_document_metadata(document_id, user_id)