#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import random

# Example raw data (in real life, you'd gather from logs, surveys, etc.)
raw_data = [
    ("The product arrived broken and customer support was unhelpful.", "negative"),
    ("Absolutely love this! Fast shipping and great quality.", "positive"),
    ("It's okay, nothing special.", "neutral"),
    ("Works as expected, no complaints.", "positive"),
    ("Terrible experience, I want a refund.", "negative"),
    ("Not bad, but the colour is slightly off.", "neutral"),
]

# Shuffle and split into train/validation (90/10)
random.seed(42)
random.shuffle(raw_data)
split = int(0.9 * len(raw_data))
train_data = raw_data[:split]
val_data = raw_data[split:]

def convert_to_openai_format(data):
    jsonl = []
    for text, label in data:
        # System message: instruct the model to output only the label
        entry = {
            "messages": [
                {"role": "system", "content": "Classify the sentiment of the customer feedback as positive, negative, or neutral. Output only the label."},
                {"role": "user", "content": text},
                {"role": "assistant", "content": label}
            ]
        }
        jsonl.append(entry)
    return jsonl

train_jsonl = convert_to_openai_format(train_data)
val_jsonl = convert_to_openai_format(val_data)

# Write to files
with open("../data/train.jsonl", "w") as f:
    for item in train_jsonl:
        f.write(json.dumps(item) + "\n")

with open("../data/val.jsonl", "w") as f:
    for item in val_jsonl:
        f.write(json.dumps(item) + "\n")

print(f"Train examples: {len(train_jsonl)}, Validation examples: {len(val_jsonl)}")
print("Sample train entry:")
print(json.dumps(train_jsonl[0], indent=2))


# In[ ]:




