#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
from dotenv import load_dotenv
load_dotenv()  # picks up .env from the repo root
from openai import OpenAI
import math
import json

API_KEY = os.environ["NVIDIA_API_KEY"]
client = OpenAI(base_url = "https://integrate.api.nvidia.com/v1",api_key=API_KEY)


# In[22]:


MODEL = "openai/gpt-oss-20b"

def get_current_time(city: str) -> str:
    return f"12:00 (simulated) in {city}"

FUNCTION_MAP = {
    "get_current_time": get_current_time
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current time for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}]

def run_agent_with_tools(user_input, verbose=True):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        msg = response.choices[0].message

        # If no tool calls, return the final text
        if not msg.tool_calls:
            if verbose:
                print("\n[Final answer]")
            return msg.content

        # Append assistant message (with tool calls) to history
        messages.append(msg)

        # Print tool call info
        if verbose:
            print("\n[Tool calls detected]")
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"  → Calling {func_name} with args: {args}")

        # Execute each tool call
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = FUNCTION_MAP[func_name](**args)

            # Append tool result
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

            if verbose:
                print(f"  ← Tool result: {result}")

        # Loop again

# Test it
final_answer = run_agent_with_tools("What time is it in Tokyo?")
print("Final answer:", final_answer)


# In[21]:


import json
from openai import OpenAI

MODEL = "openai/gpt-oss-20b"

# 1. Define the actual function(s)
def get_current_time(city: str) -> str:
    return f"12:00 (simulated) in {city}"

FUNCTION_MAP = {
    "get_current_time": get_current_time
}

# 2. Define the tool schema
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current time for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}]

def stream_agent_with_tools(user_input):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]

    while True:
        print("\n--- New stream call ---")
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            stream=True
        )

        # Accumulators
        full_text = ""
        tool_calls = {}  # index -> {id, name, arguments}

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Stream text (final answer or interim text)
            if delta and delta.content:
                full_text += delta.content
                print(delta.content, end="", flush=True)

            # Accumulate tool call parts
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": "",
                            "function": {"name": "", "arguments": ""}
                        }
                    acc = tool_calls[idx]

                    if tc.id:
                        acc["id"] += tc.id

                    if tc.function and tc.function.name:
                        acc["function"]["name"] += tc.function.name

                    if tc.function and tc.function.arguments:
                        arg = tc.function.arguments
                        if isinstance(arg, str):
                            acc["function"]["arguments"] += arg
                        # ignore ellipsis or non‑string

        # End of stream: print any accumulated tool calls
        if tool_calls:
            print("\n[Tool calls detected]")
            tool_messages = []
            for idx, call in tool_calls.items():
                func_name = call["function"]["name"]
                args_str = call["function"]["arguments"]
                # Validate JSON arguments (fallback to {})
                try:
                    args = json.loads(args_str) if args_str.strip() else {}
                except json.JSONDecodeError:
                    print(f"  ⚠ Warning: Malformed JSON for tool {func_name}: {args_str}")
                    args = {}
                print(f"  → Calling {func_name} with args: {args}")

                # Execute function
                func = FUNCTION_MAP.get(func_name)
                if func:
                    result = func(**args)
                else:
                    result = f"Error: Unknown function {func_name}"
                print(f"  ← Tool result: {result}")

                # Build tool message
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": str(result)
                })

            # Append assistant message (with tool_calls) to history
            # Note: The assistant message must include the tool_calls in the correct format
            assistant_msg = {
                "role": "assistant",
                "content": full_text or None,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": call["function"]["arguments"]
                        }
                    }
                    for call in tool_calls.values()
                ]
            }
            messages.append(assistant_msg)
            messages.extend(tool_messages)
            # Loop again to get the final answer
            continue
        else:
            # No tool calls -> we already streamed the final answer
            print("\n[Final answer streamed]")
            return full_text

# Test it
final = stream_agent_with_tools("What time is it in Tokyo?")
print("\n\nReturned final text:", final)


# In[ ]:




