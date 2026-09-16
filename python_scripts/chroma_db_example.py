#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from openai import OpenAI

API_KEY = os.environ["NVIDIA_API_KEY"]
client = OpenAI(base_url = "https://integrate.api.nvidia.com/v1",api_key=API_KEY)


# In[2]:


EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"   # or any NIM embedding model

def get_embedding(text: str):
    """Return embedding vector for a single text."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        extra_body={"input_type": "query"}, # Use "query" or "document"
        input=[text]          # input is a list
    )
    return response.data[0].embedding


# In[3]:


import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# Define a custom embedding function that calls NIM
class NIMEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # input is a list of strings
        embeddings = []
        for text in input:
            emb = get_embedding(text)
            embeddings.append(emb)
        return embeddings

# Initialize Chroma client
chroma_client = chromadb.Client()

# Create a collection with our custom embedding function
collection = chroma_client.create_collection(
    name="my_docs",
    embedding_function=NIMEmbeddingFunction()
)

# Add documents (Chroma will embed them automatically)
collection.add(
    documents=[
        "The capital of France is Paris.",
        "The Eiffel Tower is in Paris.",
        "Python is a programming language."
    ],
    ids=["1", "2", "3"]
)

# Query: find similar documents
results = collection.query(
    query_texts=["What is the capital of France?"],
    n_results=2
)
print(results["documents"])
# Expected: documents 1 and 2 (both about Paris)


# In[ ]:




