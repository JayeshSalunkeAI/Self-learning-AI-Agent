import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from graph import agent
from memory_manager import get_all_memories

st.set_page_config(page_title="Memory Agent", page_icon="🧠")
st.title("🧠 Self-Learning Memory Agent")

with st.sidebar:
    st.header("Session")
    user_id = st.text_input("User ID", value="default_user")

    if st.button("Clear chat window"):
        st.session_state.history = []
        st.rerun()

    with st.expander("What the agent remembers"):
        for m in get_all_memories(user_id):
            st.write("-", m)

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Say something...")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    lc_messages = [
        HumanMessage(content=m["content"])
        if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in st.session_state.history
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking (retrieving memories + generating)..."):
            result = agent.invoke(
                {"messages": lc_messages, "user_id": user_id, "memories": []}
            )
            reply = result["messages"][-1].content
        st.markdown(reply)

    st.session_state.history.append({"role": "assistant", "content": reply})
