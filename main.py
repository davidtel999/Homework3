from src.memory import ShortTermMemory, LongTermMemoryManager
from src.retrieval import hybrid_retriever
from langchain_openai import ChatOpenAI

# Shared LLM instance
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

BASE_INSTRUCTION = (
    "You are a conversational RAG assistant specialized in fraud analysis.\n\n"
    "How to handle data and rankings:\n"
    "When asked for statistics, rankings, extrema, or specific list positions (like 'second least likely'), "
    "always extract the relevant numbers from the retrieved text and sort them step-by-step in your internal reasoning "
    "to verify your math. Ensure your final answer matches the data perfectly. Do not describe these sorting steps "
    "to the user unless you are explicitly asked to do so in a following query. Also, do NOT introduce unnecessary details "
    "or adjacent information that goes beyond what was explicitly asked. "
    "Additionally, if the retrieved knowledge-base excerpts do not contain enough information to answer confidently, say so rather than guessing.\n\n"
    "Understanding your sources:\n"
    "1. Conversation History & Memory: Includes your live conversation log and your long-term memory summary. "
    "Use this exclusively to track user preferences, ongoing topics, or what was discussed in past sessions.\n"
    "2. Retrieved Knowledge-Base Excerpts: Reference material looked up right now to answer the current question. "
    "This is independent reference material and not part of the user's past statements. Never state that a fact "
    "came from 'last time' or 'before' unless it explicitly appears in your history logs.\n\n"
    "Handling missing transcripts:\n"
    "Your long-term memory tracks high-level topics and preferences, not exact transcripts. "
    "If the user asks you to quote an exact question or statement from a previous session, explain naturally "
    "that you keep a topic-level summary rather than a full word-for-word transcript.\n"
    "Finally, when referring to column names (mainly xlsx), quote them EXACTLY as they appear in the retrieved context. "
    "Prefer exact matches over semantic matches."
)

# Unified generate_answer() for BOTH main.py and app.py
def generate_answer(user_input, memory, llm, base_instruction, filter_category = None, k = 8):
    """
    This function:
    
        * Processes a user query through the unified RAG pipeline.
    
        * Uses k=8 to retrieve more context chunks for stability and better accuracy
        without slowing down the system (also noted in `evaluate.py`).
    
        * Returns:
            ai_reply (str)
            retrieved_docs (list[Document])
    """

    # 1. Retrieve documents (Hybrid RRF)
    retrieved_docs = hybrid_retriever.invoke(
        query=user_input,
        k=k,
        filter_category=filter_category
    )

    # 2. Build context block - explicitly labeled as knowledge-base lookup,
    # not memory, so the model can't conflate the two.
    context_str = "\n\n".join(
        f"[Source: {doc.metadata.get('source')} | Category: {doc.metadata.get('document_category')}]\n"
        f"{doc.page_content}"
        for doc in retrieved_docs
    )

    # 3. Build message payload (with the correct order)
    history = memory.get_recent_history()

    messages = [
        {"role": "system", "content": base_instruction},
    ]

    # Insert conversation history BEFORE context
    for msg in history:
        if msg["role"] in ("user", "assistant"):
            messages.append(msg)

    # Insert retrieved context AFTER history, clearly labeled
    messages.append(
        {
            "role": "system",
            "content": (
                "Knowledge-base excerpts relevant to the CURRENT question "
                "(reference material, NOT memory of past conversation):\n"
                f"{context_str}"
            ),
        }
    )

    # 4. LLM Call
    """This just runs a linear execution path. 
    The architecture intentionally avoids any complex multi-step reasoning or graph-based planning.
    The only limit we set is in our short-term memory (max 6 turns)."""
    response = llm.invoke(messages)
    ai_reply = response.content

    return ai_reply, retrieved_docs

# Main Chat Loop
def main():
    # 1. Load long-term memory
    ltm_manager = LongTermMemoryManager()
    past_summary = ltm_manager.load_summary()
    print(f"Long-term memory file checked at: {ltm_manager.filepath}")

    base_instruction = BASE_INSTRUCTION
    if past_summary:
        base_instruction += (
            "\n\n[Long-term memory summary from previous sessions - use ONLY this, "
            "not the knowledge-base excerpts, when referring to past conversations]:\n"
            f"{past_summary}"
        )

    # 2. Initialize short-term memory
    memory = ShortTermMemory(max_turns=6)

    print(">> RAG Assistant is ready. Type 'exit' to exit.")
    if past_summary:
        print(" (Loaded history from previous sessions)\n")
    else:
        print(" (No long-term memory found)\n")

    # 3. Chat loop
    while True:
        user_input = input("User: ").strip()

        if user_input.lower() == "exit":
            print()
            print("Analyzing conversation characteristics before closing...")
            # Pass the UNTRIMMED full session log so long-term memory sees
            # the whole conversation, not just the last 6-turn working buffer.
            ltm_manager.save_summary(memory.full_session_log)
            break

        if not user_input:
            continue

        # 1. Add user message to memory
        memory.add_user_message(user_input)

        # 2. Generate answer (k defaults to 8, set above in generate_answer's signature)
        ai_reply, retrieved_docs = generate_answer(
            user_input=user_input,
            memory=memory,
            llm=llm,
            base_instruction=base_instruction,
            filter_category=None
        )

        # 3. Add assistant reply to memory
        memory.add_assistant_message(ai_reply)

        print(f"\nAI: {ai_reply}\n" + "-" * 25)

if __name__ == "__main__":
    main()