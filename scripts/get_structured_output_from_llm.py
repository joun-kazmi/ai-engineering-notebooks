#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from dotenv import load_dotenv
load_dotenv()  # picks up .env from the repo root
from openai import OpenAI
import math

API_KEY = os.environ["NVIDIA_API_KEY"]
client = OpenAI(base_url = "https://integrate.api.nvidia.com/v1",api_key=API_KEY)


# In[8]:


from pydantic import BaseModel

ticket_payload = {
  "jira_issue_key": "ESC-4092",
  "salesforce_case_id": "00982341",
  "customer_tier": "Enterprise - Platinum",
  "priority": "High",
  "status": "In Progress",
  "summary": "API Rate Limit Exhaustion During Peak Sync",
  "description": "Customer is reporting HTTP 429 'Too Many Requests' errors when hitting the /v2/sync endpoint. They claim they are well within their contractual limits of 5000 RPM. This is currently blocking their nightly data reconciliation.",
  "system_metadata": {
    "tenant_id": "t-88492A",
    "region": "ap-south-1",
    "last_deploy": "2026-08-10T08:00:00Z"
  },
  "support_comments": [
    {
      "author": "L1 Support",
      "timestamp": "2026-08-11T14:15:00Z",
      "text": "Verified the customer's tenant ID. Datadog logs show they hit 5050 RPM at 14:00 UTC. Escalating to engineering to check if the API Gateway rate limiter is counting internal retries erroneously."
    },
    {
      "author": "L2 Support",
      "timestamp": "2026-08-11T15:30:00Z",
      "text": "Customer is furious. We need to analyze the logs and confirm if the 5050 RPM includes the automated fallback loops."
    }
  ]
}

# 1. Define the exact schema you want back
class EscalationAnalysis(BaseModel):
    reasoning: str
    root_cause: str
    severity_level: str
    requires_engineering_sprint: bool

# 2. Pass it to the API
response = client.beta.chat.completions.parse(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "Analyze the escalation. Think step-by-step in the reasoning field."},
        {"role": "user", "content": f"<ticket_data>{ticket_payload}</ticket_data>"}
    ],
    temperature=0.0,
    response_format=EscalationAnalysis # Forces the LLM to output this exact structure
)

analysis = response.choices[0].message.parsed
print(analysis.severity_level)


# In[ ]:





# In[ ]:




