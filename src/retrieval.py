import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# 1. Connection to your pre-built Chroma database
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)
collection = vectorstore._collection

# 2. Utility: Cosine similarity (manual implementation for filtered retrieval to avoid Chroma's ANN limitations)
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a)
    b_norms = b / np.linalg.norm(b, axis=1, keepdims=True)
    return b_norms @ a_norm

# 3. Dense Retriever (with metadata filtering)
def dense_retrieve(query: str, k: int = 8, filter_category: str = None) -> list[Document]:
    query_vec = np.array(embedding_model.embed_query(query))

    # FILTERED BRANCH (Manual Cosine over exact category)
    if filter_category:
        where_clause = {"document_category": {"$eq": filter_category}}
        results = collection.get(
            where=where_clause,
            include=["documents", "metadatas", "embeddings"],
        )

        if not results["ids"]:
            return []

        embeddings = np.array(results["embeddings"])
        sims = _cosine_sim(query_vec, embeddings)
        sorted_indices = np.argsort(sims)[::-1]

        unique_docs = []
        # Removed 'seen_sources' constraint to allow multiple relevant slices from the same file
        for idx in sorted_indices[:k]:
            unique_docs.append(
                Document(
                    page_content=results["documents"][idx],
                    metadata=results["metadatas"][idx]
                )
            )
        return unique_docs

    # UNFILTERED BRANCH (Chroma Native ANN Search)
    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=k,    # Directly pull k items now that source constraint is lifted
    )

    unique_docs = []
    if results["documents"] and results["documents"][0]:
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            unique_docs.append(Document(page_content=text, metadata=meta))

    return unique_docs

# 4. Sparse Retriever (BM25) with metadata filtering

# Load ALL documents from Chroma once
all_db_data = collection.get(include=["documents", "metadatas"])
all_documents = [
    Document(page_content=text, metadata=meta)
    for text, meta in zip(all_db_data["documents"], all_db_data["metadatas"])
]

# Global BM25 index (unfiltered)
global_sparse_retriever = BM25Retriever.from_documents(all_documents)

def sparse_retrieve(query: str, k: int = 8, filter_category: str = None) -> list[Document]:
    
    # FILTERED BRANCH
    if filter_category:
        filtered_docs = [
            doc for doc in all_documents
            if doc.metadata.get("document_category") == filter_category
        ]

        if not filtered_docs:
            return []

        retriever = BM25Retriever.from_documents(filtered_docs)
        retriever.k = k
        return retriever.invoke(query)

    # UNFILTERED BRANCH
    global_sparse_retriever.k = k
    return global_sparse_retriever.invoke(query)

# 5. Hybrid Retriever (RRF) with metadata filtering
class CustomHybridRetriever:
    def __init__(self, dense_fn, sparse_fn):
        self.dense_fn = dense_fn
        self.sparse_fn = sparse_fn

    def invoke(self, query: str, k: int = 8, filter_category: str = None) -> list[Document]:
        dense_docs = self.dense_fn(query, k=k, filter_category=filter_category)
        sparse_docs = self.sparse_fn(query, k=k, filter_category=filter_category)

        rrf_scores = {}
        all_docs_map = {}

        # RRF scoring: 1 / (rank + 60)
        for rank, doc in enumerate(dense_docs):
            doc_id = doc.page_content
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + 60)
            all_docs_map[doc_id] = doc

        for rank, doc in enumerate(sparse_docs):
            doc_id = doc.page_content
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + 60)
            all_docs_map[doc_id] = doc

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [all_docs_map[doc_id] for doc_id, _ in sorted_docs[:k]]

# 6. Global Hybrid Instance
hybrid_retriever = CustomHybridRetriever(
    dense_fn=dense_retrieve,
    sparse_fn=sparse_retrieve
)