import sqlite3
from datetime import datetime

from config import APP_DB_PATH


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(APP_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, file_hash)
            )
            """
        )


def document_exists(user_id: str, file_hash: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id
            FROM documents
            WHERE user_id = ? AND file_hash = ?
            """,
            (user_id, file_hash),
        ).fetchone()

    return row is not None


def save_document(
    document_id: str,
    user_id: str,
    file_name: str,
    file_hash: str,
    chunk_count: int,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO documents (id, user_id, file_name, file_hash, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                user_id,
                file_name,
                file_hash,
                chunk_count,
                datetime.now().isoformat(),
            ),
        )


def list_documents(user_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, file_name, chunk_count, created_at
            FROM documents
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def delete_document_metadata(document_id: str, user_id: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            DELETE FROM documents
            WHERE id = ? AND user_id = ?
            """,
            (document_id, user_id),
        )