import os
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
from pathlib import Path
from datetime import datetime
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setup paths based on project structure
base_dir = Path(__file__).resolve().parent.parent
data_folder = base_dir / "data"
persist_db_dir = base_dir / "chroma_db"

# Master file category mapping
CATEGORY_MAP = {
    "account_profiles_FraudRings.xlsx": "identity_profiles",
    "fraud_patterns_FraudRings.xlsx": "fraud_patterns",
    "network_edges_FraudRings.xlsx": "fraud_networks",
    "time_series_stats_FraudRings.xlsx": "time_series",
    "creditcard.json": "fraud_case_study",
    "luxury_cosmetics_fraud_analysis_2025.json": "fraud_case_study",
    "FraudTypes_Paper.md": "fraud_definitions",
    "ACFE_Perpetrator_Demographics_2024.md": "macro_fraud_research",
    "Paysafecard_Consumer_Fraud_Victims_2018.md": "macro_fraud_research",
    "Impact_of_fraud_in_Europe.pdf": "macro_fraud_research",
    "olafreport-2025_en.pdf": "macro_fraud_research",
    "Measuring_fraud_and_earnings_management_by_a_case.html": "fraud_case_study",
    "ACI_Credit_Card_Fraud_Victims_2016.html": "macro_fraud_research",
}

def build_metadata(file_path: str, filename: str) -> dict:
    # Builds the required metadata dict: document_category, source, creation_date.
    assigned_category = CATEGORY_MAP.get(filename, "general_fraud_data")
    creation_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d")
    return {
        "source": filename,
        "document_category": assigned_category,
        "creation_date": creation_date,
    }

def route_and_parse(file_path: str) -> list[Document]:
    # Routes a single file to the correct loader based on its extension.
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    metadata = build_metadata(file_path, filename)

    # 1. Handle PDF Documents
    if ext == ".pdf":
        print(f"Routing PDF: {filename}")
        docs = PyPDFLoader(file_path).load()
        for d in docs:
            d.metadata = metadata
        return docs

    # 2. Handle Markdown Documents
    elif ext == ".md":
        print(f"Routing MD: {filename}")
        docs = TextLoader(file_path, encoding="utf-8").load()
        for d in docs:
            d.metadata = metadata
        return docs

    # 3. Handle HTML Documents 
    elif ext == ".html":   
        print(f"Routing HTML: {filename}")
        docs = UnstructuredHTMLLoader(file_path).load()
        for d in docs:
            d.metadata = metadata
        return docs

    # 4. Handle Excel Documents with Row-batch Sticky Headers + Dedicated Schema Chunks (added for QA search)
    elif ext == ".xlsx":
        print(f"Routing XLSX: {filename}")
        sheets = pd.read_excel(file_path, sheet_name=None)
        docs = []
        batch_size = 50
        
        for sheet_name, df in sheets.items():
            headers_string = ", ".join(df.columns.tolist())
            
            # Inject a text-only schema chunk to preserve column layout and field attributes for QA search
            schema_text = (
                f"Excel File Schema Structure Profile\n"
                f"File Name: {filename}\n"
                f"Sheet Name: {sheet_name}\n"
                f"Tracked Column Fields and Attributes: {headers_string}\n"
                f"Description: This document profiles the explicit data column layout fields and schema properties tracked inside {filename}."
            )
            docs.append(Document(page_content=schema_text, metadata=metadata))
            
            # Process the data entries normally
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i : i + batch_size]
                batch_text = batch_df.to_string(index=False, header=False)
                labeled_text = (
                    f"SOURCE FILE: {filename} (Sheet: {sheet_name}, Rows {i}-{i + len(batch_df) - 1})\n"
                    f"COLUMNS: [ {headers_string} ]\n"
                    f"DATA:\n{batch_text}"
                )
                docs.append(Document(page_content=labeled_text, metadata=metadata))
        return docs

    # 5. Handle JSON Documents with Object-Streaming and Schema Summary for QA Target Search
    elif ext == ".json":    
        print(f"Routing JSON: {filename}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data if isinstance(data, list) else [data]
        docs = []

        # Explicit Schema Definition Chunk for QA Target Search
        if records and isinstance(records[0], dict):
            field_names = list(records[0].keys())
            schema_summary = (
                f"Schema summary for {filename}: this file contains {len(records)} records. "
                f"Each record has the following fields: {', '.join(field_names)}."
            )
            docs.append(Document(page_content=schema_summary, metadata=metadata))

        """Sample capping to control memory footprints and data density, set to 50 records for efficiency,
        but will be presented as 51 including the schema summary chunk."""
        sample_limit = 50   
        for i, record in enumerate(records[:sample_limit]):
            record_text = f"Source: {filename} | Record #{i} | Data: {json.dumps(record)}"
            docs.append(Document(page_content=record_text, metadata=metadata))

        print(f"  -> Generated {len(docs)} structured record chunks for {filename}")
        return docs

    else:
        print(f"Skipping unsupported file: {filename}")
        return []

def parse_all_documents() -> list[Document]:
    # Parses every file in the data folder into a flat list of Documents.
    all_docs = []

    if not os.path.exists(data_folder):
        print(f"Error: Data folder not found at {data_folder}")
        return all_docs

    for filename in os.listdir(data_folder):
        full_path = os.path.join(data_folder, filename)
        if os.path.isfile(full_path):
            all_docs.extend(route_and_parse(full_path))

    print(f"\nTotal source document objects loaded: {len(all_docs)}")
    return all_docs

def chunk_documents(all_docs: list[Document]) -> list[Document]:
    # Splits text-heavy documents while protecting structured file matrices.
    final_chunks = []
    rec_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
    # Chose a larger chunk size for better context retention and a medium-large overlap to preserve continuity across chunks.

    for doc in all_docs:
        source_file = doc.metadata.get("source", "").lower()
        if source_file.endswith(".xlsx") or source_file.endswith(".json"):
            final_chunks.append(doc)
        else:
            final_chunks.extend(rec_splitter.split_documents([doc]))

    print(f"Processed text into {len(final_chunks)} total evaluation chunks.")
    return final_chunks


def build_or_load_vectorstore(force_rebuild: bool = False) -> Chroma:
    # Loads existing collection or builds the vector index from clean files from scratch.
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    if persist_db_dir.exists() and any(persist_db_dir.iterdir()) and not force_rebuild:
        print(f"Found existing store at '{persist_db_dir}', loading...")
        vectorstore = Chroma(persist_directory=str(persist_db_dir), embedding_function=embedding_model)
        print(f"Loaded store with {vectorstore._collection.count()} chunks.")
        return vectorstore

    print("No existing store found. Building database collections...")
    all_docs = parse_all_documents()
    final_chunks = chunk_documents(all_docs)

    print("Conversion to embeddings / persisting to Vector DB...")
    vectorstore = Chroma.from_documents(
        documents=final_chunks,
        embedding=embedding_model,
        persist_directory=str(persist_db_dir),
    )
    print(f"Vector store created at '{persist_db_dir}'.")
    return vectorstore

if __name__ == "__main__":
    # Force rebuild to apply the optimized schema architecture
    vectorstore = build_or_load_vectorstore(force_rebuild=True)

    # Quick validation check on structural deployment
    sample = vectorstore.get(limit=3, include=["metadatas"])
    print("\nSample stored metadata attributes:")
    for meta in sample["metadatas"]:
        print(" ", meta)
    
    print('\n')