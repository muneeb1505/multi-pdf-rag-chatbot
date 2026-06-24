import requests
import streamlit as st

st.set_page_config(
    page_title="Multi PDF RAG",
    page_icon="📚"
)

st.title("📚 Multi PDF RAG Chatbot")

# ---------------------
# Session State Init
# ---------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "all_chats" not in st.session_state:
    st.session_state.all_chats = []  # list of past chat sessions (each is a list of messages)

# ---------------------
# Sidebar
# ---------------------

with st.sidebar:

    # --- New Chat Button ---
    if st.button("➕ New Chat"):
        if st.session_state.messages:
            st.session_state.all_chats.append(st.session_state.messages)
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # --- PDF Upload ---
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        file_names = [f.name for f in uploaded_files]
        already_indexed = st.session_state.get("indexed_files")

        if file_names != already_indexed:
            with st.status("Processing PDFs...", expanded=True) as status:
                st.write("📤 Uploading...")
                response = requests.post(
                    "http://localhost:8000/upload",
                    files=[("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
                )
                if response.ok:
                    st.write("🔍 Indexing chunks...")
                    st.write("✅ Ready!")
                    status.update(label="PDFs loaded successfully!", state="complete")
                    st.session_state.indexed_files = file_names
                else:
                    status.update(label="Upload failed.", state="error")
        else:
            for f_name in file_names:
                st.write(f"✅ {f_name}")
            st.caption("Already indexed — ask away!")

    st.divider()

    # --- Past Chats ---
    if st.session_state.all_chats:
        st.subheader("🕘 Past Chats")
        for i, chat in enumerate(reversed(st.session_state.all_chats)):
            # Show first user message as the chat label
            first_msg = next((m["content"] for m in chat if m["role"] == "user"), "Chat")
            label = first_msg[:40] + "..." if len(first_msg) > 40 else first_msg
            if st.button(f"💬 {label}", key=f"past_chat_{i}"):
                # Save current chat before switching
                if st.session_state.messages:
                    st.session_state.all_chats.append(st.session_state.messages)
                st.session_state.messages = chat
                st.rerun()

# ---------------------
# Main Chat Area
# ---------------------

# --- Show Chat Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Clear Chat Button (only show if there are messages) ---
if st.session_state.messages:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------
# User Question
# ---------------------

question = st.chat_input("Ask about your PDFs...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    history = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in st.session_state.messages
    ])

    response = requests.post(
        "http://localhost:8000/ask",
        json={
            "question": question,
            "chat_history": history
        }
    )

    answer = response.json()["answer"]

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.rerun()
