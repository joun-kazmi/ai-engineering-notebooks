#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[4]:


import numpy as np

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Generate embeddings for three sentences
dog_emb = get_embedding("dog")
puppy_emb = get_embedding("puppy")
car_emb = get_embedding("automobile")

sim_dog_puppy = cosine_similarity(dog_emb, puppy_emb)
sim_dog_car = cosine_similarity(dog_emb, car_emb)

print(f"dog–puppy similarity: {sim_dog_puppy:.4f}")   # usually 0.8+
print(f"dog–car similarity:   {sim_dog_car:.4f}")     # usually <0.5


# In[ ]:




