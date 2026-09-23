#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
from dotenv import load_dotenv
load_dotenv()  # picks up .env from the repo root
# Nemotron reasons out loud by default; these demos want direct answers
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
from openai import OpenAI
import math

API_KEY = os.environ["NVIDIA_API_KEY"]
client = OpenAI(base_url = "https://integrate.api.nvidia.com/v1",api_key=API_KEY)

prompt = "The capital of France is"

# We ask the API to return the log probabilities
response = client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b",
    extra_body=NO_THINK,  # otherwise the first token is reasoning ("Okay"), not the answer
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1,
    logprobs=True,
    top_logprobs=5 # Ask for the top 5 probable next words
)

# Extract the logprobs
if not response.choices[0].logprobs:
        print("Error: The model/provider did not return logprobs. Try using a standard OpenAI model like 'gpt-4o-mini'.")
else:
    # FIX 2: Updated Pydantic structure for OpenAI SDK v1.x
    # content is a list of token objects, we grab the first one [0]
    top_logprobs_dict = response.choices[0].logprobs.content[0].top_logprobs

    print(f"Prompt: '{prompt}'\n")
    print("Top 5 Next Word Predictions (Mathematical Probabilities):")
    print("-" * 50)

    # Logprobs are returned as natural logs (base e). We convert them back to standard probabilities
    for token_info in top_logprobs_dict:
        token = token_info.token
        logprob = token_info.logprob
        
        # Math: Probability = e^(logprob)
        probability = math.exp(logprob) 
        print(f"Token: '{token:<10}' | Probability: {probability:.4f} ({probability*100:.2f}%)")

    # Optional: Sum them up to prove it's a probability distribution
    total_prob = sum(math.exp(t.logprob) for t in top_logprobs_dict)
    print("-" * 50)
    print(f"Sum of top 5 probs: {total_prob:.4f}")


# In[ ]:




