#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from dotenv import load_dotenv
load_dotenv()  # picks up .env from the repo root
# Nemotron reasons out loud by default; these demos want direct answers
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
from openai import OpenAI
import math

API_KEY = os.environ["NVIDIA_API_KEY"]
client = OpenAI(base_url = "https://integrate.api.nvidia.com/v1",api_key=API_KEY)


# In[3]:


import json


# ---------------------------------------------------------------------------
# 1. DEFINE LOCAL PYTHON FUNCTIONS (THE ACTUAL WORKERS)
# ---------------------------------------------------------------------------
def get_jira_issue_details(issue_key: str) -> str:
    """Simulates querying the Jira API for issue details."""
    # In production, this would be an HTTP request via 'requests' or 'httpx'
    database = {
        "ESC-4092": {
            "issue_key": "ESC-4092",
            "summary": "API Rate Limit Exhaustion During Peak Sync",
            "status": "IN_PROGRESS",
            "affected_customer": "Acme Corp",
            "rate_limit_rpm": 5000,
            "actual_peak_rpm": 5050,
            "error_log": "HTTP 429: Too Many Requests on endpoint /v2/sync"
        }
    }
    
    result = database.get(issue_key)
    if result:
        return json.dumps(result)
    return json.dumps({"error": f"Issue key '{issue_key}' not found."})


# Create a mapping dictionary to map function names from LLM strings to actual executable functions
available_tools = {
    "get_jira_issue_details": get_jira_issue_details,
}


# ---------------------------------------------------------------------------
# 2. DEFINE THE JSON TOOL SCHEMAS FOR THE LLM
# ---------------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_jira_issue_details",
            "description": "Fetches technical details and error logs for a specific Jira escalation issue key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "The Jira issue key, e.g., 'ESC-4092'"
                    }
                },
                "required": ["issue_key"]
            }
        }
    }
]


# ---------------------------------------------------------------------------
# 3. INITIAL LLM CALL WITH USER PROMPT + TOOL DEFINITIONS
# ---------------------------------------------------------------------------
messages = [
    {
        "role": "system",
        "content": "You are a senior technical support engineer. Use tools when necessary to retrieve ticket facts before answering."
    },
    {
        "role": "user",
        "content": "Can you check the details for ticket ESC-4092 and tell me if the customer exceeded their rate limit?"
    }
]

print("--- STEP 1: Sending initial prompt to LLM ---")
response = client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b",
    extra_body=NO_THINK,
    messages=messages,
    tools=tools,
    tool_choice="auto",  # Allows the LLM to choose whether to call a tool or reply directly
    temperature=0.0
)

response_message = response.choices[0].message
messages.append(response_message)  # Always append the assistant's message to conversation history


# ---------------------------------------------------------------------------
# 4. INTERCEPT & PROCESS THE TOOL CALLS
# ---------------------------------------------------------------------------
tool_calls = response_message.tool_calls

if tool_calls:
    print(f"\n--- STEP 2: LLM requested {len(tool_calls)} tool call(s) ---")
    
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"Executing local function: {function_name}(**{function_args})")
        
        # Look up the actual function and invoke it dynamically
        function_to_call = available_tools.get(function_name)
        if function_to_call:
            tool_output = function_to_call(**function_args)
            
            # ---------------------------------------------------------------
            # 5. PASS TOOL RESULTS BACK TO THE LLM
            # ---------------------------------------------------------------
            messages.append({
                "tool_call_id": tool_call.id,  # Matches result to the exact request
                "role": "tool",
                "name": function_name,
                "content": tool_output  # Function output MUST be a string
            })
            print(f"Tool Result returned: {tool_output}")

    # -----------------------------------------------------------------------
    # 6. FINAL LLM CALL FOR SYNTHESIS
    # -----------------------------------------------------------------------
    print("\n--- STEP 3: Sending tool output back to LLM for final answer ---")
    second_response = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b",
        extra_body=NO_THINK,
        messages=messages,
        temperature=0.0
    )
    
    final_answer = second_response.choices[0].message.content
    print("\n--- FINAL ANSWER ---")
    print(final_answer)

else:
    print("LLM did not call any tools. Direct Response:")
    print(response_message.content)


# In[ ]:




