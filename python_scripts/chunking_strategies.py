#!/usr/bin/env python
# coding: utf-8

# In[1]:


def fixed_chunk(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# In[2]:


# Copy and run this cell in your notebook
text = """Apple plans to release a new phone next fall. The device will feature an upgraded battery and screen.

Meanwhile, market analysts expect strong sales in Asia. Supply chains are already preparing components."""

print(text)


# In[3]:


import re

def recursive_split(text, chunk_size=1000, chunk_overlap=200):
    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# In[6]:


chunk_fixed = fixed_chunk(text, chunk_size=55, overlap=0)
chunk_recursive = recursive_split(text, chunk_size=120)
for i in chunk_fixed:
    print('chunk_fixed part ', i)
for j in chunk_recursive:
    print('chunk_recursive part ', j)


# In[ ]:




