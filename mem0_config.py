import os
from dotenv import load_dotenv
from mem0 import Memory

load_dotenv()

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": os.getenv("COLLECTION_NAME", "memories"),
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", "6333")),
            "embedding_model_dims": 768,  # nomic-embed-text output size
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": os.getenv("LLM_MODEL", "llama3.1:8b"),
            "temperature": 0,
            "max_tokens": 2000,
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": os.getenv("EMBED_MODEL", "nomic-embed-text:latest"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        },
    },
}

memory = Memory.from_config(config)
