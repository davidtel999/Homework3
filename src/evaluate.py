import json
import statistics
from pathlib import Path
from datetime import datetime
from langchain_openai import ChatOpenAI
from main import generate_answer, llm, BASE_INSTRUCTION
from src.memory import ShortTermMemory

# Paths 
base_dir = Path(__file__).resolve().parent.parent
eval_dataset_path = base_dir / "eval_dataset.jsonl"
results_path = base_dir / "evaluation_results.json"   # The output file where the evaluation results will be saved.

# Judge model 
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def load_eval_dataset(path: Path) -> list[dict]:
    # Reads the jsonl evaluation file into a list of dicts, one per question.
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def run_single_question(question: str, k: int = 8) -> tuple[str, str]:
    """
    This function:
    
        * Runs a single question through the RAG pipeline for evaluation.
    
        * Uses k=8 to retrieve more context chunks for stability and better accuracy
        without slowing down the system (also noted in `main.py`).
    
        * Gives each question a fresh, empty memory to prevent answers from leaking
        context into one another.
    """
    fresh_memory = ShortTermMemory(max_turns=6)
    # `generate_answer` reads the question text from memory's history, not from
    # the `user_input` argument directly - so it must be added here first,
    # exactly like the `main.py` chat loop does before calling `generate_answer`.
    fresh_memory.add_user_message(question)

    answer, retrieved_docs = generate_answer(
        user_input=question,
        memory=fresh_memory,
        llm=llm,
        base_instruction=BASE_INSTRUCTION,
        filter_category=None,
        k=k,
    )

    # generate_answer now returns raw Document objects, not a pre-joined
    # string - rebuild a labeled context string here for the judge prompt.

    context_str = "\n\n".join(
        f"[Source: {doc.metadata.get('source')} | Category: {doc.metadata.get('document_category')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )
    return answer, context_str

JUDGE_PROMPT_TEMPLATE = """You are a strict evaluator of a RAG system's answer.

Question:
{question}

Retrieved context (what the system had available to answer with):
{context}

Generated answer (what the system actually said):
{answer}

Ground truth answer (the correct answer):
{ground_truth_answer}

Evaluate the generated answer using these three dimensions:
1. Faithfulness: Is the generated answer grounded in the retrieved context, with no invented facts?
2. Answer Relevance: Does the generated answer actually address the question asked?
3. Context Precision: Was the retrieved context actually the right information needed to answer this question?

Combine all three dimensions into a single overall score between 0 and 1, where:
1.0 means the answer is fully correct, grounded, and the right context was retrieved.
0.0 means the answer is wrong, unsupported by the context, or the context was irrelevant.
You use intermediate values for partial correctness.

Respond with ONLY a valid JSON object, no other text, in exactly this format:
{{"score": <float between 0 and 1>, "reason": "<one or two sentence explanation>"}}
"""

def judge_answer(question: str, context: str, answer: str, ground_truth_answer: str) -> dict:
    # Calls the Judge LLM at temperature=0 and parses its {"score", "reason"} response.
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
        answer=answer,
        ground_truth_answer=ground_truth_answer,
    )
    response = judge_llm.invoke(prompt)
    raw = response.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
        return {"score": float(parsed["score"]), "reason": parsed["reason"]}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": 0.0, "reason": f"PARSE_ERROR: judge response was not valid JSON: {raw[:200]}"}

def run_evaluation():
    eval_items = load_eval_dataset(eval_dataset_path)
    print(f"Loaded {len(eval_items)} evaluation questions from {eval_dataset_path}\n")

    results = []
    for i, item in enumerate(eval_items, start=1):
        question = item["question"]
        ground_truth_answer = item["ground_truth_answer"]
        source_file = item.get("source_file", "")

        answer, context = run_single_question(question)
        judgment = judge_answer(question, context, answer, ground_truth_answer)

        results.append({
            "question": question,
            "source_file": source_file,
            "generated_answer": answer,
            "ground_truth_answer": ground_truth_answer,
            "score": judgment["score"],
            "reason": judgment["reason"],
        })

        print(f"\n[{i}/{len(eval_items)}] {question}")
        print(f"Score: {judgment['score']:.2f}")

    scores = [r["score"] for r in results]
    mean_score = statistics.mean(scores) if scores else 0.0
    accuracy_percent = mean_score * 100

    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(results),
        "mean_score": mean_score,
        "final_accuracy_percent": accuracy_percent,
    }

    output = {"summary": summary, "results": results}
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nFinal accuracy: {accuracy_percent:.1f}%")
    print(f"Full results saved to: {results_path}")
    return output

if __name__ == "__main__":
    run_evaluation()