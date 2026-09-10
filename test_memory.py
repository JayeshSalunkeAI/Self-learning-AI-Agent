from mem0_config import memory

TEST_USER = "test_user"

print("Adding a test memory...")
memory.add("I love Python and I'm currently learning LangGraph.", user_id=TEST_USER)

print("Searching it back...")
results = memory.search(
    "What programming language does the user like?",
    filters={"user_id": TEST_USER},
)
if isinstance(results, dict):
    results = results.get("results", [])

print(f"\nFound {len(results)} memories for '{TEST_USER}':")
for item in results:
    print(" -", item["memory"])

print("\nIf you see a memory mentioning Python above, Ollama + Qdrant + mem0 are all linked correctly.")
