import streamlit as st
from main import generate_answer, BASE_INSTRUCTION, llm
from src.memory import ShortTermMemory, LongTermMemoryManager

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chat Interface")

# Session-persistent state
if "memory" not in st.session_state:
    st.session_state.memory = ShortTermMemory(max_turns=6)

if "ltm_manager" not in st.session_state:
    st.session_state.ltm_manager = LongTermMemoryManager()

# Loading the summary into session state exactly once, on startup
if "past_summary" not in st.session_state:
    st.session_state.past_summary = st.session_state.ltm_manager.load_summary()

if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []

memory = st.session_state.memory
ltm = st.session_state.ltm_manager
past_summary = st.session_state.past_summary

base_instruction = BASE_INSTRUCTION
if past_summary:
    base_instruction += (
        "\n\n[Long-term memory summary from previous sessions - use ONLY this, "
        f"not the knowledge-base excerpts, when referring to past conversations]:\n{past_summary}"
    )

# Sidebar: retrieved chunks for the latest answer
st.sidebar.header("Retrieved Chunks (Latest to Earliest Query)")
if st.session_state.last_retrieved_docs:
    for i, doc in enumerate(st.session_state.last_retrieved_docs, 1):
        with st.sidebar.expander(f"Chunk {i}: {doc.metadata.get('source', 'unknown')}"):
            st.markdown(f"**Source:** {doc.metadata.get('source')}")
            st.markdown(f"**Category:** {doc.metadata.get('document_category')}")
            st.text(doc.page_content)
else:
    st.sidebar.info("Ask a question and the related chunks will be registered here.")

# Basic Session Controls
st.sidebar.markdown("---")
if st.sidebar.button("Save & Restart"):
    if memory.full_session_log:
        with st.spinner("Saving session summary..."):
            # 1. Persist the complete untrimmed history to long-term memory
            ltm.save_summary(memory.full_session_log)
        
        st.sidebar.success("Session saved.")
        
        # 2. Hard reset the active volatile short-term session states
        st.session_state.memory = ShortTermMemory(max_turns=6)
        st.session_state.last_retrieved_docs = []
        
        # 3. Reload the brand new updated cumulative summary into the UI state
        st.session_state.past_summary = ltm.load_summary()
        
        # 4. Refresh the app layout back to a clean startup slate
        st.rerun()
    else:
        st.sidebar.warning("No active chat history to save.")

# Displaying the full ongoing chat history on every render
for msg in memory.get_recent_history():
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_input = st.chat_input("Ask something...")

if user_input:
    memory.add_user_message(user_input)

    current_filter = None  # category filtering not wired into the UI yet

    with st.spinner("Thinking..."):
        ai_reply, retrieved_docs = generate_answer(
            user_input=user_input,
            memory=memory,
            llm=llm,
            base_instruction=base_instruction,
            filter_category=current_filter,
        )

    memory.add_assistant_message(ai_reply)
    st.session_state.last_retrieved_docs = retrieved_docs

    # Rerun so the history loop above redraws everything cleanly
    st.rerun()