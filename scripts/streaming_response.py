#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from dotenv import load_dotenv
load_dotenv()  # picks up .env from the repo root
from openai import OpenAI
import math

API_KEY = os.environ["NVIDIA_API_KEY"]
client = OpenAI(base_url = "https://integrate.api.nvidia.com/v1",api_key=API_KEY)


# In[5]:


stream = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Explain quantum computing in 3 sentences."}],
    stream=True  # This is all you change
)

# Iterate over server-sent events
for chunk in stream:
     if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)


# In[ ]:




