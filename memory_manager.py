from mem0_config import memory


def store_chat_memory(user_id: str, user_message: str, assistant_message: str) -> None:
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]

    memory.add(messages, user_id=user_id)


def search_user_memory(user_id: str, query: str, limit: int = 5) -> list[str]:
    response = memory.search(
        query,
        filters={"user_id": user_id},
        limit=limit,
    )

    results = response.get("results", []) if isinstance(response, dict) else response

    return [
        item["memory"]
        for item in results
        if isinstance(item, dict) and item.get("memory")
    ]


def get_all_user_memories(user_id: str) -> list[str]:
    response = memory.get_all(filters={"user_id": user_id})

    results = response.get("results", []) if isinstance(response, dict) else response

    return [
        item["memory"]
        for item in results
        if isinstance(item, dict) and item.get("memory")
    ]