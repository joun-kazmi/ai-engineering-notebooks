#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
text = "Hello, world! How are you today?"
tokens = enc.encode(text)
print(f"Text: '{text}'")
print(f"Tokens ({len(tokens)}): {tokens}")
print(f"Decoded back: {enc.decode(tokens)}")
# Fun: try non-English text and see token count explode
hindi = "नमस्ते दुनिया"
hindi_tokens = enc.encode(hindi)
print(f"Hindi text '{hindi}' = {len(hindi_tokens)} tokens")


# In[3]:


import tiktoken

def count_tokens(text, model="gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

prompt = "What is the capital of France?"
print(f"Input tokens: {count_tokens(prompt)}")


# In[ ]:




