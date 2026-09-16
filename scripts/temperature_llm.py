#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Exercise 1: Your first API call with both OpenAI and Anthropic
import os
from openai import OpenAI
from anthropic import Anthropic

API_KEY = os.environ["NVIDIA_API_KEY"]
# Store your keys in .env file
openai_client = OpenAI(base_url = "https://integrate.api.nvidia.com/v1",api_key=API_KEY)
prompt = "Write a one-sentence creative description of a sunset."

# OpenAI call
response = openai_client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1
)
print("OpenAI:", response.choices[0].message.content)



# In[ ]:




