#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from openai import OpenAI

API_KEY = os.environ["NVIDIA_API_KEY"]
client = OpenAI(base_url = "https://integrate.api.nvidia.com/v1",api_key=API_KEY)


# In[2]:


EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"   # or any NIM embedding model
MODEL = "openai/gpt-oss-20b"

def get_embedding(text: str):
    """Return embedding vector for a single text."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        extra_body={"input_type": "query"}, # Use "query" or "document"
        input=[text]          # input is a list
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# In[3]:


### Hybrid Search BM25 + vector search
from rank_bm25 import BM25Okapi
import numpy as np

def hybrid_search(query, documents, embeddings, bm25_index, alpha=0.5):
    # BM25 scores
    tokenized_docs = [doc.split() for doc in documents]
    bm25_scores = bm25_index.get_scores(query.split())
    
    # Vector scores (cosine similarity)
    query_emb = get_embedding(query)
    vector_scores = [cosine_similarity(query_emb, emb) for emb in embeddings]
    
    # Normalize and combine
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)
    vector_scores = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + 1e-9)
    
    combined = alpha * vector_scores + (1 - alpha) * bm25_scores
    top_indices = np.argsort(combined)[::-1][:3]
    return [documents[i] for i in top_indices], combined[top_indices]


# In[4]:


### Query Expansion
def expand_query(user_query):
    prompt = f"Given the question, generate 3 alternative phrasings or keywords that might help in retrieval.\nQuestion: {user_query}\nAlternatives:"
    print('prompt ', prompt)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content


# In[5]:


### HyDE (Hypothetical Document Embeddings)
def hyde_query(user_query):
    prompt = f"Write a short paragraph that answers this question, even if you don't know exact details.\nQuestion: {user_query}"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    hypothetical_doc = response.choices[0].message.content
    return hypothetical_doc

# Then embed the hypothetical_doc instead of the original query


# In[6]:


#### Multi-Query Retrieval
def generate_multi_queries(user_query, n=3):
    prompt = f"Generate {n} different search queries related to: {user_query}"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    # Assume each line is a query
    queries = response.choices[0].message.content.strip().split('\n')
    return queries


# In[7]:


# Sample Data
documents = ["Apple plans to release a new phone", "Market analysts expect strong sales"]
embeddings = [get_embedding(doc) for doc in documents]
bm25_index = BM25Okapi([doc.split() for doc in documents])

# A. Call Hybrid Search
docs, scores = hybrid_search("Apple phone", documents, embeddings, bm25_index)
print("Hybrid Search Results:", docs)

# B. Call Query Expansion
expanded = expand_query("new phone")
print("Expanded Query:\n", expanded)

# C. Call HyDE
hypothetical = hyde_query("new phone")
print("HyDE Document:\n", hypothetical)

# D. Call Multi-Query Retrieval
queries = generate_multi_queries("new phone")
print("Multi-Queries:", queries)


# In[ ]:




