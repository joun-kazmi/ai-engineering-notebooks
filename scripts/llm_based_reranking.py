#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
from dotenv import load_dotenv
load_dotenv()  # picks up .env from the repo root
# Nemotron reasons out loud by default; these demos want direct answers
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
from openai import OpenAI

API_KEY = os.environ["NVIDIA_API_KEY"]
# Replace with your actual NIM key
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY
)
MODEL = "nvidia/nemotron-3-super-120b-a12b"

def llm_rerank(query, documents, top_k=3, verbose=True):
    """
    Scores each document's relevance to the query using the LLM.
    Returns top_k documents sorted by descending relevance score.
    """
    scored_docs = []
    for doc in documents:
        prompt = f"""On a scale of 1 to 10, how relevant is the following document to the query?
Query: {query}
Document: {doc}
Relevance score (1-10):"""

        response = client.chat.completions.create(
            model=MODEL,
            extra_body=NO_THINK,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0
        )

        print('response ', response)

        raw_score = response.choices[0].message.content.strip()
        try:
            score = float(raw_score)
        except ValueError:
            # If the model doesn't return a number, assign 0 and optionally log
            print(f"Warning: Could not parse score for doc: '{doc[:50]}...' -> '{raw_score}'")
            score = 0.0

        scored_docs.append((doc, score))
        if verbose:
            print(f"Score {score:4.1f} | {doc[:60]}...")

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in scored_docs[:top_k]]
    return top_docs

# ========== Test Data ==========
query = "What are the health benefits of regular exercise?"

documents = [
    "Regular physical activity can reduce the risk of heart disease and stroke.",
    "Exercise helps control weight, improves mental health, and strengthens bones.",
    "The history of the Olympic Games dates back to ancient Greece.",
    "A balanced diet includes fruits, vegetables, and whole grains.",
    "Cardiovascular exercises like running and swimming increase heart rate and lung capacity.",
    "The stock market can be volatile; diversification is key.",
    "Strength training builds muscle mass and boosts metabolism.",
    "Regular exercise is linked to better sleep and reduced anxiety.",
]

print("=== Testing LLM Reranker ===")
top_3 = llm_rerank(query, documents, top_k=3, verbose=True)

print("\n=== Top 3 Documents ===")
for i, doc in enumerate(top_3, 1):
    print(f"{i}. {doc}")


# In[ ]:




