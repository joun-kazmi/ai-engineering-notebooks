#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
from openai import OpenAI

API_KEY = os.environ["NVIDIA_API_KEY"]
client = OpenAI(base_url = "https://integrate.api.nvidia.com/v1",api_key=API_KEY)


# In[7]:


EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"   # or any NIM embedding model

def get_embedding(text: str):
    """Return embedding vector for a single text."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        extra_body={"input_type": "query"}, # Use "query" or "document"
        input=[text]          # input is a list
    )
    return response.data[0].embedding


# In[12]:


import numpy as np
import chromadb
import time

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 1. Semantic search with numpy
sentences = [
    "I love programming in Python.",
    "The Python snake is a constrictor.",
    "Machine learning is fascinating.",
    "I enjoy hiking in the mountains.",
]
query = "How do you feel about coding?"

sentence_embs = [get_embedding(s) for s in sentences]
query_emb = get_embedding(query)


sims = [cosine_similarity(query_emb, emb) for emb in sentence_embs]
top_idx = np.argmax(sims)
print(f"Query: {query}")
print(f"Top match: {sentences[top_idx]} (similarity {sims[top_idx]:.3f})")


# 2. Same with Chroma
from chromadb import Documents, EmbeddingFunction, Embeddings

class NIMEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [get_embedding(text) for text in input]

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="test1", embedding_function=NIMEmbeddingFunction())
collection.add(documents=sentences, ids=[str(i) for i in range(len(sentences))])

results = collection.query(query_texts=[query], n_results=1)
print("Chroma top match:", results["documents"][0])


# In[ ]:




