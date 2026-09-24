#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json, os
from dotenv import load_dotenv
load_dotenv()  # picks up .env from the repo root
# Nemotron reasons out loud by default; these demos want direct answers
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)
MODEL = "nvidia/nemotron-3-super-120b-a12b"

SYSTEM_PROMPT = """You are an incident response assistant for a SaaS platform.
You investigate service issues by searching logs, checking recent deploys,
and reading metrics. Always investigate with tools before answering.
When you have enough evidence, state the likely root cause and next step."""


# In[2]:


# ---- Fake data sources (swap for real ones in the capstone) ----
FAKE_LOGS = {
    "payments-api": [
        "10:41:58 WARN  connection pool exhausted",
        "10:42:03 ERROR timeout connecting to postgres-primary",
        "10:42:05 ERROR timeout connecting to postgres-primary",
        "10:42:31 ERROR 503 returned for /charge",
    ],
    "auth-service": [
        "10:40:12 INFO  token refresh OK",
    ],
}
FAKE_DEPLOYS = [
    {"service": "payments-api", "version": "v2.14.3", "time": "10:35", "author": "dana@corp"},
]
FAKE_METRICS = {
    "payments-api": {"p95_latency_ms": 4200, "error_rate": 0.31, "db_conn_active": 100},
}

def search_logs(service: str, keyword: str, minutes: int = 30) -> dict:
    logs = FAKE_LOGS.get(service, [])
    hits = [l for l in logs if keyword.lower() in l.lower()]
    return {"service": service, "window_min": minutes, "matches": hits}

def get_recent_deploys(service: str) -> dict:
    return {"deploys": [d for d in FAKE_DEPLOYS if d["service"] == service]}

def get_metrics(service: str) -> dict:
    return FAKE_METRICS.get(service, {"error": f"no metrics for {service}"})

TOOL_REGISTRY = {
    "search_logs": search_logs,
    "get_recent_deploys": get_recent_deploys,
    "get_metrics": get_metrics,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_logs",
            "description": "Search recent logs of a service for a keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "e.g. payments-api"},
                    "keyword": {"type": "string", "description": "e.g. ERROR, timeout"},
                    "minutes": {"type": "integer", "description": "lookback window", "default": 30},
                },
                "required": ["service", "keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_deploys",
            "description": "List recent deployments for a service.",
            "parameters": {
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metrics",
            "description": "Get current latency, error rate and DB connections for a service.",
            "parameters": {
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
            },
        },
    },
]


# In[10]:


def run_agent(user_query: str, max_iterations: int = 8):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    for step in range(1, max_iterations + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            extra_body=NO_THINK,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",   # model decides: answer or call a tool
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump())   # echo assistant msg back verbatim

        # BASE CASE: no tool calls → final answer, exit
        if not msg.tool_calls:
            print(f"\n[step {step}] FINAL ANSWER")
            return msg.content

        # RECURSIVE CASE: run each requested tool, feed results back
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"[step {step}] TOOL CALL → {name}({args})")

            try:
                result = TOOL_REGISTRY[name](**args)
            except Exception as e:
                result = {"error": str(e)}   # never crash the loop; report to the model

            print(f"[step {step}] OBSERVED  → {json.dumps(result)[:200]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    return "⚠️ Stopped: max iterations reached without a final answer."

answer = run_agent("Users report payments are failing since ~10:40. Investigate payments-api.")
print(answer)


# In[3]:


#### Pattern #2: Plan-and-Execute

PLANNER_PROMPT = """You plan incident investigations. Given the alert, output a JSON list of steps.
Each step MUST have: {"tool": <name>, "args": {<tool-specific args>}, "reason": <why>}.

Available tools and their required args:
- search_logs: {"service": str, "keyword": str, "minutes": int (optional, default 30)}
- get_recent_deploys: {"service": str}
- get_metrics: {"service": str}

Example:
[
  {"tool": "search_logs", "args": {"service": "payments-api", "keyword": "ERROR"}, "reason": "find errors"},
  {"tool": "get_recent_deploys", "args": {"service": "payments-api"}, "reason": "check deploys"}
]

Max 5 steps. Output ONLY valid JSON, no markdown."""


def plan_and_execute(alert: str):
    # PHASE 1: plan
    plan_resp = client.chat.completions.create(
        model=MODEL,
        extra_body=NO_THINK,
        messages=[{"role": "user", "content": PLANNER_PROMPT + f"\n\nAlert: {alert}"}],
    )
    plan_text = plan_resp.choices[0].message.content
    # Strip markdown code fences if present
    if plan_text.startswith("```"):
        plan_text = plan_text.split("```")[1]
        if plan_text.startswith("json"):
            plan_text = plan_text[4:]
    plan = json.loads(plan_text)

    # PHASE 2: execute with validation
    findings = []
    for step in plan:
        tool_name = step["tool"]
        args = step.get("args", {})
        
        # Defensive: check required args
        if tool_name == "search_logs":
            args.setdefault("service", "payments-api")
            args.setdefault("keyword", "ERROR")  # fallback keyword
        elif tool_name in ("get_recent_deploys", "get_metrics"):
            args.setdefault("service", "payments-api")
        
        print(f"Executing: {tool_name}({args})")
        result = TOOL_REGISTRY[tool_name](**args)
        findings.append({"step": step["reason"], "tool": tool_name, "args": args, "result": result})

    # PHASE 3: synthesize
    return client.chat.completions.create(
        model=MODEL,
        extra_body=NO_THINK,
        messages=[{"role": "user", "content":
            f"Alert: {alert}\n\nFindings:\n{json.dumps(findings, indent=2)}\n\nWrite the root-cause summary."}],
    ).choices[0].message.content

answer = plan_and_execute("Users report payments are failing since ~10:40. Investigate payments-api.")
print( answer)


# In[4]:


####  Pattern #3: Reflection Loop

def with_reflection(task: str, max_rounds: int = 2):
    draft = client.chat.completions.create(
        model=MODEL, extra_body=NO_THINK, messages=[{"role": "user", "content": task}]
    ).choices[0].message.content

    for _ in range(max_rounds):
        critique = client.chat.completions.create(
            model=MODEL,
            extra_body=NO_THINK,
            messages=[{"role": "user", "content":
                f"Review this RCA draft for missing evidence, unsupported claims, or vague "
                f"recommendations. Reply ONLY 'APPROVED' if it's solid, else list issues.\n\n{draft}"}],
        ).choices[0].message.content

        if "APPROVED" in critique:
            return draft
        draft = client.chat.completions.create(
            model=MODEL,
            extra_body=NO_THINK,
            messages=[{"role": "user", "content": f"Original task: {task}\n\nYour draft:\n{draft}\n\n"
                                                  f"Fix these issues:\n{critique}"}],
        ).choices[0].message.content
    return draft


answer = with_reflection("Users report payments are failing since ~10:40. Investigate payments-api.")
print('answer : ', answer)


# In[ ]:




