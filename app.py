import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from database import initialize_database, list_documents
from graph import agent
from memory_manager import get_all_user_memories
from pdf_utils import extract_text_from_uploaded_file
from rag_service import delete_document, ingest_document


st.set_page_config(
    page_title="Knowledge Memory Agent",
    page_icon="🧠",
    layout="wide",
)

initialize_database()


def initialize_session() -> None:
    defaults = {
        "user_id": None,
        "chat_history": [],
        "processed_file_key": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_chat() -> None:
    st.session_state.chat_history = []


def show_login() -> None:
    left, center, right = st.columns([1, 2, 1])

    with center:
        st.markdown("# 🧠 Knowledge Memory Agent")
        st.markdown(
            "Upload your documents, ask questions from them, "
            "and let the assistant remember useful preferences over time."
        )

        username = st.text_input(
            "Choose a workspace name",
            placeholder="Example: alice, student_01, acme_team",
        )

        if st.button("Enter workspace", use_container_width=True):
            clean_username = username.strip().lower().replace(" ", "_")

            if len(clean_username) < 3:
                st.error("Workspace name must contain at least 3 characters.")
            else:
                st.session_state.user_id = clean_username
                st.session_state.chat_history = []
                st.rerun()


def render_sidebar(user_id: str) -> None:
    with st.sidebar:
        st.markdown("## 🧠 Your Workspace")
        st.caption(f"Active user: `{user_id}`")

        if st.button("➕ New chat", use_container_width=True):
            reset_chat()
            st.rerun()

        if st.button("Log out", use_container_width=True):
            st.session_state.user_id = None
            reset_chat()
            st.rerun()

        st.divider()

        st.markdown("### 📄 Upload knowledge")
        uploaded_file = st.file_uploader(
            "PDF or TXT file",
            type=["pdf", "txt"],
            help="Your file is processed into private RAG chunks for this workspace.",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()

            upload_key = f"{user_id}:{uploaded_file.name}:{len(file_bytes)}"

            if st.session_state.processed_file_key != upload_key:
                try:
                    with st.spinner("Reading, chunking, embedding, and storing..."):
                        extracted_text = extract_text_from_uploaded_file(
                            uploaded_file.name,
                            file_bytes,
                        )

                        success, message = ingest_document(
                            user_id=user_id,
                            file_name=uploaded_file.name,
                            file_bytes=file_bytes,
                            extracted_text=extracted_text,
                        )

                    if success:
                        st.success(message)
                    else:
                        st.warning(message)

                    st.session_state.processed_file_key = upload_key

                except Exception as error:
                    st.error(f"Upload failed: {error}")

        st.divider()

        st.markdown("### 📚 Your documents")
        documents = list_documents(user_id)

        if not documents:
            st.caption("No documents uploaded yet.")
        else:
            for document in documents:
                col1, col2 = st.columns([4, 1])

                with col1:
                    st.caption(
                        f"📄 {document['file_name']} "
                        f"({document['chunk_count']} chunks)"
                    )

                with col2:
                    if st.button("🗑️", key=f"delete_{document['id']}"):
                        try:
                            delete_document(user_id, document["id"])
                            st.success("Document deleted.")
                            st.rerun()
                        except Exception as error:
                            st.error(f"Could not delete: {error}")

        st.divider()

        st.markdown("### 🧠 Agent memory")
        memories = get_all_user_memories(user_id)

        if not memories:
            st.caption("No long-term memories yet.")
        else:
            for memory in memories[:8]:
                st.caption(f"• {memory}")


def render_chat() -> None:
    st.markdown("# Knowledge Memory Agent")
    st.caption(
        "Ask questions from your uploaded documents. "
        "The assistant also remembers useful information across chats."
    )

    if not st.session_state.chat_history:
        st.info(
            "Start by uploading a PDF/TXT from the sidebar, "
            "then ask a question about it."
        )

    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

            if item["role"] == "assistant" and item.get("sources"):
                with st.expander("Sources used"):
                    for source in item["sources"]:
                        st.markdown(f"- `{source}`")


def ask_agent(user_id: str, prompt: str) -> tuple[str, list[str]]:
    messages = []

    for item in st.session_state.chat_history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))

    messages.append(HumanMessage(content=prompt))

    result = agent.invoke(
        {
            "messages": messages,
            "user_id": user_id,
            "rag_documents": [],
            "memories": [],
            "sources": [],
        }
    )

    answer = result["messages"][-1].content
    sources = result.get("sources", [])

    return str(answer), sources


def main() -> None:
    initialize_session()

    if not st.session_state.user_id:
        show_login()
        return

    user_id = st.session_state.user_id

    render_sidebar(user_id)
    render_chat()

    prompt = st.chat_input("Ask about your uploaded documents...")

    if prompt:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents, recalling memory, and thinking..."):
                try:
                    answer, sources = ask_agent(user_id, prompt)
                    st.markdown(answer)

                    if sources:
                        with st.expander("Sources used"):
                            for source in sources:
                                st.markdown(f"- `{source}`")

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as error:
                    st.error(f"Agent error: {error}")


if __name__ == "__main__":
    main()