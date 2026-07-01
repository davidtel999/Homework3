# Advanced Retrieval-Augmented Generation

## 1. Domain & Dataset

For this assignment we chose _"Financial and Commercial Fraud Research and Cases"_ as our knowledge domain.

The choice was primarily due to the vast diversity and volume of available data, alongside the sheer interest of the field. Our datasets were acquired from a range of sources (e.g., Statista, ResearchGate, Kaggle), and are expressed through many different file types, each serving a different purpose. Here is a table with a summary of the contribution of each file:

| # | File Name | Description |
|---|-----------|-------------|
| 1 | `account_profiles_FraudRings.xlsx` | Row-based account profile records revealing useful personal data as part of the broader fraud rings research. Used to test the pipeline's ability to pull precise individual metadata during entity resolution queries. |
| 2 | `fraud_patterns_FraudRings.xlsx` | A compact lookup table mapping the fraud ring classifications and metrics for each case. Integrated to evaluate how well the retriever extracts exact categorical matches. |
| 3 | `network_edges_FraudRings.xlsx` | Explicit account-connection pairs linked by attributes and grouped by a `ring_id`. Integrated to evaluate if the RAG system can map complex structural relationships. |
| 4 | `time_series_stats_FraudRings.xlsx` | Hourly chronological rows tracking transaction counts, aggregate fraud metrics, and risk averages over a temporal timeline - data best expressed in a spreadsheet form. |
| 5 | `creditcard.json` | Individual transaction records used to evaluate the pipeline's ability to parse semi-structured pairs. The json format helps with the machine-handling of the file's dense data profile. |
| 6 | `luxury_cosmetics_fraud_analysis_2025.json` | Specialized e-commerce fraud related data providing a niche context for domain-specific queries. |
| 7 | `ACI_Credit_Card_Fraud_Victims_2016.html` | Briefing of a global credit card fraud survey, included as a file example of a short-length web markup. |
| 8 | `Measuring_fraud_and_earnings_management_by_a_case.html` | Case study about estimating the probability of fraud and earnings management for a specific family business, included to assess HTML text extraction and semantic chunking on long-form. |
| 9 | `olafreport-2025_en.pdf` | Official European Anti-Fraud Office publication detailing EU budget irregularities, customs evasion, and recovery metrics. Primarily used to test RAG chunking layout on a complex multi-page document. |
| 10 | `Impact_of_fraud_in_Europe.pdf` | Dense academic text detailing macroeconomic fraud loss cases, reasons and statistics in the european region. Very useful for covering a diverse variety of possible queries. |
| 11 | `ACFE_Perpetrator_Demographics_2024.md` | Very short-lengthed data tables accompanied by a summary of the distribution of fraud cases by the perpetrator's specific age group (converted to `md` from `xls`). |
| 12 | `FraudTypes_Paper.md` | Brief academic text detailing structural variations of economic fraud, useful for quickly retrieving "textbook" definitions (converted to `md` from `pdf`). |
| 13 | `Paysafecard_Consumer_Fraud_Victims_2018.md` | Notes on prepaid/voucher card fraud in a fittingly lightweight text format (converted to `md` from `xls`). |

## 2. Ingestion & Filtering 

This section explains how raw data files are processed and structured before being added to the vector store, and how metadata-based filtering is integrated into retrieval. Because we deal with a mix of spreadsheets, JSON records, HTML pages, and text files, a single generic loader won't suffice. The subsections below ensure that columns keep their labels, large datasets don't overload memory, and small metadata categories aren't lost during the database search. It is then shown how category-based pre-filtering and hybrid retrieval operate together to return reliable results.

### Assignment of `document_category`

In `src/ingestion.py`, a dictionary called `CATEGORY_MAP` assigns each filename to a semantic category (`document_category`, e.g., `"fraud_patterns_FraudRings.xlsx": "fraud_patterns"`). This must be filename-based, because multiple files share the same extension (e.g., 4 `.xlsx` files) but might belong to different conceptual groups. If we grouped by extension alone, all the similar files in that regard, like our spreadsheets, would collapse into a single meaningless batch.

Every chunk also receives two additional metadata "labels": a `source` (the original filename) and a `creation_date` (taken from the file's last-modified timestamp on disk).

### Excel Ingestion: Row-Batch Sticky Headers + Schema Chunks

Excel sheets are ingested using a row-batching strategy that preserves column semantics and keeps each chunk self-contained. Each sheet is split into batches of 50 rows, and the column header row is explicitly repeated ("sticky headers") in every batch. This prevents the common failure mode where a character-based splitter leaves only the first chunk with headers and all subsequent chunks with unlabeled numeric values. Without sticky headers, questions referencing column names (e.g., "What does the Amount column represent?") would fail because the retriever would surface chunks containing only raw numbers - this is exactly what happened during our own testing before the fix, and it's discussed further in Section 4.

Each sheet also receives a dedicated schema chunk listing all column names plainly. This acts as a reference point for schema-level questions such as "Which fields does this spreadsheet track?" or "What fraud-related attributes appear in this dataset?".

Together, the schema chunk + sticky-header batches produce Excel chunks that are structurally meaningful and robust to retrieval.

### JSON Ingestion: Object-Streaming with Schema Summary

JSON files are processed using an object-streaming approach. The loader reads the file into a list of records and emits one chunk per record, capped at a small fixed sample size to avoid producing thousands of tiny chunks. This keeps memory usage predictable and prevents extremely large JSON datasets from overwhelming the vector store.

Before emitting record chunks, the ingestion pipeline generates a schema summary chunk listing all field names in simple language. This is crucial for fraud datasets containing repetitive or anonymized numeric fields (e.g., V1-V28 in credit-card fraud data), where meaningful fields like `Class` or `Amount` can be buried. The schema summary provides a clean target for questions such as "Which field represents the fraud label?" or "What fields does this JSON file contain?".

The combination of a schema chunk + a capped set of representative records yields a compact, semantically rich JSON ingestion suitable for RAG without unnecessary cost or noise.

### HTML Normalization: Use of the "Unstructured" Library

The project uses the `unstructured` library (via the `UnstructuredHTMLLoader`) to normalize `.html` files before chunking. Raw HTML files contain a lot of code-like website elements - e.g., menus, layout structure - that don't correspond to actually usable document content. `Unstructured`'s partitioning logic strips this markup down into clean text, so chunking operates on readable prose rather than raw tag structure.

### PDF and Markdown Ingestion
Unlike the Excel sticky-header issue from before, PDF and Markdown ingestion didn't surface any equivalent chunk-boundary problems during testing - `PyPDFLoader` already segments PDFs by page before chunking, and plain prose has no header row to lose. Also, PDF extraction did not require `unstructured` because `PyPDFLoader` already normalizes page content into a clean text stream, avoiding the layout‑dependent inconsistencies that HTML markup introduces.

### Handling Large Files

Very large files are capped at a fixed character or record limit before chunking. Without this, a dataset like `creditcard.json` (~300,000 rows) would produce endless tiny chunks - expensive to embed and impractically slow. The cap preserves a representative slice of the data while still following the required `json.dumps` / `df.to_string` serialization format.

### Integration of the Metadata Filter with Retrieval

The function `dense_retrieve(query, k, filter_category)` in `src/retrieval.py` behaves differently depending on whether a category is provided:

- **Without filter:** a standard vector similarity search across the entire collection.
- **With filter:** instead of relying on ChromaDB's combined "search + filter", the system first fetches only the chunks belonging to the requested category, then manually ranks that smaller subset by similarity (_more details will be given in the next subsection_).

This design exists because ChromaDB's combined filtering uses an approximate HNSW search that can fail to return matches for rare categories. We observed this directly: a category containing only two chunks returned zero results through the combined search, even though both chunks clearly matched the query. Pre-filtering and then ranking guarantees correctness for small categories.

This same retrieval instability is why `k = 8` is used during evaluation. Re-chunking one file type (e.g., Excel) occasionally shifted retrieval rankings for unrelated files. Requesting more candidates provides a safety margin against such ranking drift without altering the filtering logic.

#### Unconventional usage of `numpy` in `src/retrieval.py`
`Numpy` was used to manually compute cosine similarity in the metadata-filtered dense retrieval branch, bypassing ChromaDB's native HNSW search which was found to silently miss matches in small categories during testing.

### Why hybrid retrieval?
The system uses hybrid retrieval as its primary search strategy, combining dense (vector similarity) and sparse (BM25) retrieval via the **R**eciprocal **R**ank **F**usion method. Dense retrieval alone can miss exact matches for specific names, column labels, or numeric values that don't embed well semantically - for example, a query for a specific field name like `Class` or `fraud_share_pct` benefits from BM25's exact term matching. RRF merges the two ranked lists without requiring any tuning of relative weights, using the standard formula 1/(rank + 60) per document per list.

## 3. Memory Architecture 

The system uses two distinct memory layers, each serving a different purpose.

### Short-term memory (`ShortTermMemory` in `src/memory.py`)

* Implemented as a simple Python list of `{"role": ..., "content": ...}` dictionaries.
* Stores only real user/assistant turns; system instructions are injected separately.
* Automatically trimmed to the last `max_turns` exchanges (arbitrary value ~ default: 6).
* Lives only in RAM for the duration of the session. It resets on restart, since its job is resolving short-range references like "it" or "that".

### Long-term memory (`LongTermMemoryManager` in `src/memory.py`)

* Does not store the raw transcript.
* At the end of a session, the short-term buffer is compressed by an LLM into exactly three things: topics discussed, user preferences, and unresolved questions.
* This information summary (_not the transcript_) is written to a JSON file on disk.
* At the start of the next session, the summary is loaded and injected into the system prompt, offering some continuity without preserving exact wording. This design choice is intentional: saving the raw transcript instead would cause the system prompt to grow unboundedly across sessions, eventually exceeding the model's context limit.

**Synopsis:** Short-term memory contains the exact transcript (temporarily, in RAM)  **//**  Long-term memory contains a compressed summary (persistently, on disk).

### Potential Limitation: Long Session Summarization

Finally, `full_session_log` sends the entire session history to the LLM in one call for long-term summarization. In an extended session this could reach gpt-4o-mini's context limit, causing the summarization call to fail. For shorter sessions, like this project's case, this is not a practical concern, but it is worth noting as a general design constraint.

### Two encountered bugs of this task

1. **Relative file paths caused the system to read and delete different physical files depending on the working directory.** The original code used a bare filename with no folder attached, which resolves differently depending on which directory you happen to run the script from - so "delete the memory file" and "read the memory file" weren't always the same physical file. The fix anchors the path to the project's absolute root directory using Python's `pathlib` module - specifically `Path(__file__).resolve().parent.parent` - guaranteeing consistent read/write behavior regardless of where `main.py` is launched from.
2. **The assistant confused retrieved knowledge-base context with past conversation.** Because both were injected into the same prompt, and the system instructions encouraged "acknowledging past interactions," the model sometimes treated retrieved facts (e.g., a transaction amount) as if they came from a previous session - even when long-term memory was completely empty. The fix was adding an explicit instruction clarifying that retrieved context and conversation memory are separate sources, and that the model must never describe something as "from before" unless it actually appears in the conversation history or the saved long-term summary.

## 4. Evaluation Results

The mean LLM-as-a-Judge score was 77.2%, which was a positive outcome. The reason the score wasn't even higher is that the evaluation QA set deliberately included several traps targeting different stages of the pipeline. Here are the five most interesting examples among our results, including four diagnoses:

**[4/19]** **Q:** What is one shared attribute type used to link accounts in network_edges_FraudRings.xlsx? | Score: 0.80 | **Correct Case**

    This query resulted in an unexpectedly high score, since there were four valid shared attribute types, making the chance of a matching answer only 25%. The reason the system consistently scores this as "correct" (≥ 0.5) is that the ground-truth answer itself was chained with an explicit instruction: *"Any one of the following valid network link types recorded in the shared_type column: 'phone', 'email', 'ip_address', or 'device_id'. You may provide any of these attributes."*

**[5/19]** **Q:** What is the name of the column in time_series_stats_FraudRings.xlsx that displays the mean number of transactions across the timeline? | Score: 0.30 | **LLM Hallucination**

    The RAG could not recognize the difference between "mean number of transactions" and the existing, irrelevant column `avg_amount`, since "average" and "mean" are semantically related. In an earlier iteration, it also failed to recognize the distinction between this metric and the raw transaction counter column, answering with `transactions_count` instead. This was a genuine model reasoning failure, and not a retrieval one - the correct context was retrieved each time, but the model picked the wrong column from it.

**[10/19]** **Q:** Who is the author of FraudTypes_Paper.md and which institution are they affiliated to? | Score: 0.00 | **Retrieval Failure**

    The system failed to surface the chunk containing the author's name and institution at all - the model's own answer stated the context didn't mention it, which was true of what it received. This was traced to a side effect of an earlier ingestion change: restructuring how Excel files were chunked shifted ChromaDB's approximate nearest-neighbor rankings for *unrelated* files too, since ANN search ranks chunks relative to the whole collection, not just within one file. This is the reason `k = 8` is used during evaluation (see Section 2) - a wider candidate pool reduces, though doesn't fully erase, this kind of cross-file ranking drift.

**[18/19]** **Q:** What is the worldwide percentage of fraudsters being between the ages of -1 and 37 years old in 2022 and 2023? | Score: 0.50 | **LLM Hallucination**

    The system answered correctly for part of the question, correctly summing the percentages for age groups 0-35 (29%). However, it failed to flag "-1" as an invalid age value, and also ignored that ages 36-37 fell into an age bracket not present in the source data. The right context was retrieved; the model simply didn't apply the input-validation reasoning the question was testing for.

**[19/19]** **Q:** What is the specific time period during which 55 percent of Mexican respondents reported being victims of credit card fraud? | Score: 0.80 | **LLM Hallucination**

    Despite the seemingly correct answer, this question was deliberately written with a factual error: the actual reported share was 56%, not 55%. The system fell into the trap because it located a real textual fragment saying "...more than 55 percent of respondents indicating they had been victims of credit card fraud...", so the model just treated the wrong premise in the question as confirmed by the context.

## Bonus A - Streamlit Chat Interface

This addition introduces a refined **Streamlit** frontend (`app.py`), wrapping the exact retrieval, short-term memory, and long-term memory backend pipelines used by the Command-Line Interface into an interactive web UI.

### Features 

- **Native Chat Experience:** Utilizes Streamlit's `st.chat_message` components for multiple-turn conversations.
- **Session Continuity:** Preserves short-term context across chat turns using `st.session_state` to ensure memory stays persistent across re-runs. Streamlit reruns the entire script on every interaction, so anything that needs to survive between messages has to be stored in `st.session_state` and not in a plain local variable - this was an actual bug that got fixed during development.
- **Source Transparency:** Features a dedicated sidebar displaying the exact retrieved document chunks, along with their `source` and `document_category` metadata.
- **Manual Memory Update:** The interface includes a control button (`Save & Restart`) to manually store the current session into long-term memory and then clear the active session history, triggering a fresh conversation.

## 5. Execution Instructions

This project requires:

- Python 3.10.x (tested on 3.10.11)
- An OpenAI API key
- Git
- Internet access

---

### STEP 1 - Clone the Repository

```bash
git clone https://github.com/davidtel999/hw3.git
cd hw3
```

---

### STEP 2 - Create & Activate a Virtual Environment

```bash
python -m venv .venv
```

#### Windows

```bash
.venv\Scripts\activate
```

#### Mac/Linux

```bash
source .venv/bin/activate
```

---

### STEP 3 - Install Dependencies

```bash
pip install -r requirements.txt
```

This line installs:

- LangChain ecosystem (`langchain-core`, `langchain-community`, `langchain-openai`, `langchain-chroma`, `langchain-text-splitters`)
- Document processing tools (`unstructured`, `pypdf`, `openpyxl`, `pandas`)
- Retrieval utilities (`rank-bm25`, `numpy`)
- Streamlit UI (Bonus A)

**Note:** ChromaDB itself isn't pinned directly - `langchain-chroma` installs a compatible version automatically (as noted in `requirements.txt` too).

---

### STEP 4 - Set Your OpenAI API Key

#### Windows (PowerShell)

```bash
$env:OPENAI_API_KEY = "sk-proj-..."
```

#### Mac/Linux

```bash
export OPENAI_API_KEY="sk-proj-..."
```

This sets the key in your active terminal session, but it's lost if you close that terminal or open a new one.

---

### STEP 5 - Build the Vector Store (Run Ingestion Once)

This step parses all documents, enriches metadata, chunks them, embeds them, and persists the ChromaDB index.

```bash
python -m src.ingestion
```

A `chroma_db/` folder will be created automatically.

**Important:**
You only need to run ingestion again if you add or modify documents.

---

### STEP 6 - Run the Conversational CLI (main.py)

This step launches the terminal-based RAG assistant with short-term and long-term memory.

```bash
python main.py
```

Type `exit` to end the session and trigger long-term memory summarization.

---

### STEP 7 - Run the Evaluation Script

This step executes the full LLM-as-a-Judge pipeline over the evaluation dataset.

```bash
python -m src.evaluate
```

The script will:

- run 19 QA pairs
- retrieve context
- generate answers
- call the Judge LLM (temperature = 0)
- compute the mean score

The final accuracy is printed to the console.

---

### STEP 8 - Run the Streamlit Chat Interface (Bonus A)

```bash
streamlit run app.py
```

The UI will open automatically in your browser.

**Note:** When launching the app, Streamlit might emit dependency warnings from its automated file-watcher, such as:

`ModuleNotFoundError: No module named 'torchvision'`

These originate from Streamlit scanning installed packages and encountering optional sub-modules it doesn't strictly need, which are harmless and don't affect the application's execution. Uninstalling `transformers` can silence this extensive warning but is unrecommended, even though it's highly tempting: the `unstructured` library (used for `.html` parsing in this project) might rely on `transformers` for certain document-layout parsing paths, so removing it risks silently breaking ingestion.

