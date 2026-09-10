from mem0 import Memory

from config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    EMBED_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
    MEMORY_COLLECTION,
)

mem0_config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": MEMORY_COLLECTION,
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
            "embedding_model_dims": 768,
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": LLM_MODEL,
            "temperature": 0,
            "max_tokens": 1500,
            "ollama_base_url": OLLAMA_BASE_URL,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": EMBED_MODEL,
            "ollama_base_url": OLLAMA_BASE_URL,
        },
    },
}

memory = Memory.from_config(mem0_config)