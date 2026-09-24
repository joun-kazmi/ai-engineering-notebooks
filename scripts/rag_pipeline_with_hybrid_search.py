#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
from dotenv import load_dotenv
load_dotenv()  # picks up .env from the repo root
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from rank_bm25 import BM25Okapi
import numpy as np
from openai import OpenAI

API_KEY = os.environ["NVIDIA_API_KEY"]
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=API_KEY)
MODEL = "openai/gpt-oss-20b"
EMBED_MODEL = "nvidia/nemotron-3-embed-1b"

def get_embedding(text, input_type="passage"):
    # "passage" for documents, "query" for search queries
    return client.embeddings.create(model=EMBED_MODEL, input=[text], extra_body={"input_type": input_type}).data[0].embedding

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Custom embedding for Chroma
class NIMEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [get_embedding(text) for text in input]

    def embed_query(self, input: Documents) -> Embeddings:
        # Chroma calls this for query_texts, so queries get the query-side encoding
        return [get_embedding(text, input_type="query") for text in input]

# Sample documents (in practice, you'd parse PDFs and chunk them)
documents = [
    "The capital of France is Paris. Paris is known for the Eiffel Tower.",
    "Python is a high-level programming language used for web development and data science.",
    "Machine learning models require training data to learn patterns.",
    "The Eiffel Tower is located in Paris and is a famous landmark.",
    "React is a JavaScript library for building user interfaces.",
]

# 1. Store in Chroma (vector DB)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="docs1", embedding_function=NIMEmbeddingFunction())
collection.add(documents=documents, ids=[str(i) for i in range(len(documents))])

# 2. Build BM25 index for hybrid search
tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# Precompute embeddings for documents (Chroma already stores them, but we need for hybrid)
doc_embeddings = [get_embedding(doc) for doc in documents]

# 3. Hybrid search function
def hybrid_search(query, alpha=0.5, k=2):
    # BM25 scores
    bm25_scores = bm25.get_scores(query.split())
    # Vector scores
    query_emb = get_embedding(query, input_type="query")
    vector_scores = [cosine_similarity(query_emb, emb) for emb in doc_embeddings]
    
    # Normalize
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)
    vector_scores = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + 1e-9)
    
    combined = alpha * vector_scores + (1 - alpha) * bm25_scores
    top_indices = np.argsort(combined)[::-1][:k]
    return [(documents[i], combined[i]) for i in top_indices]

# 4. Run a query
results = hybrid_search("What is the capital of France?", alpha=0.5)
for doc, score in results:
    print(f"Score {score:.3f}: {doc}")


# In[ ]:




