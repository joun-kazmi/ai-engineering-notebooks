#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Exercise 1: Your first API call
import os
from dotenv import load_dotenv
load_dotenv()  # picks up .env from the repo root
# Nemotron reasons out loud by default; these demos want direct answers
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
from openai import OpenAI

API_KEY = os.environ["NVIDIA_API_KEY"]
# Store your keys in .env file
openai_client = OpenAI(base_url = "https://integrate.api.nvidia.com/v1",api_key=API_KEY)
prompt = "Write a one-sentence creative description of a sunset."

# OpenAI call
response = openai_client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b",
    extra_body=NO_THINK,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1
)
print("OpenAI:", response.choices[0].message.content)



# In[ ]:




