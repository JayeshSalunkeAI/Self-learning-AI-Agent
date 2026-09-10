from mem0_config import memory


def add_memory(user_id: str, user_message: str, assistant_message: str) -> None:
    """Store one conversation turn as durable facts for this user."""
    conversation = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]
    memory.add(conversation, user_id=user_id)


def search_memory(user_id: str, query: str, limit: int = 5) -> list[str]:
    """Return the most relevant memory strings for this user + query."""
    results = memory.search(query, filters={"user_id": user_id}, limit=limit)
    if isinstance(results, dict):
        results = results.get("results", [])
    return [item["memory"] for item in results]


def get_all_memories(user_id: str) -> list[str]:
    """Return everything mem0 knows about this user (handy for debugging)."""
    results = memory.get_all(filters={"user_id": user_id})
    if isinstance(results, dict):
        results = results.get("results", [])
    return [item["memory"] for item in results]
