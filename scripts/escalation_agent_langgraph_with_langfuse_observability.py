#!/usr/bin/env python
# coding: utf-8

# In[1]:


###  The Escalation Agent Scaffold
'''
(START)
    │
    ▼
┌────────┐  reads: alert
│ triage │  writes: parsed {service, severity}      [validated]
└───┬────┘
    ├──[SEV3]─────────────────────────┐
    ▼ [SEV1/SEV2]                     ▼
┌─────────────┐                 ┌────────────┐
│ investigate │◀─[revise & <3]──│ auto_close │──▶(END)
└──────┬──────┘                 └────────────┘
       ▼                          ▲   ▲
  ┌────────┐                      │   │
  │ verify │──[ok]──┐             │   │
  └────────┘        ▼             │   │
              ┌──────────┐        │   │
              │ write_rca│        │   │
              └────┬─────┘        │   │
        [no write action]─────────┘   │
              ▼ [write action]        │
        ┌────────────┐ ⏸              │
        │ human_gate │                │
        └──────┬─────┘                │
    [approved] ▼    [rejected]────────┘ → escalate_human → (END)
          ┌─────────┐
          │ execute │──▶(END)
          └─────────┘

 '''
# ══════════════════════════════════════════════════════════
# PART 3 — ESCALATION AGENT SCAFFOLD (complete, fixed)
# pip install -U langgraph langchain-openai pydantic
# export NVIDIA_API_KEY="nvapi-..."
# ══════════════════════════════════════════════════════════

# ── 0. IMPORTS ────────────────────────────────────────────
import inspect
import operator
import os
from dotenv import load_dotenv
load_dotenv()  # picks up .env from the repo root
# Nemotron reasons out loud by default; these demos want direct answers
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
from typing import Annotated, Literal, TypedDict
import json
import re
from pydantic import ValidationError
from pydantic import BaseModel, Field, ValidationError

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt

# ── 1. MODEL ──────────────────────────────────────────────
llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
    extra_body=NO_THINK,
)



def llm_structured(prompt: str, schema, max_retries: int = 3):
    """LLM → Pydantic schema, with retry-with-reflection.

    FIX: Inject the schema's JSON structure into the prompt so the model
    knows EXACTLY what shape to produce. Without this, retries just thrash
    on random shapes (which is the bug you hit).
    """
    # THE KEY FIX: show the model the target schema
    schema_json = json.dumps(schema.model_json_schema(), indent=2)

    feedback = ""
    for attempt in range(1, max_retries + 1):
        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with ONLY a raw JSON object matching this exact schema:\n"
            f"{schema_json}\n"
            f"No markdown, no code fences, no commentary — JSON only."
        )
        if feedback:
            full_prompt += (
                f"\n\nYour previous output FAILED validation:\n{feedback}\n"
                f"Fix these issues and return corrected JSON."
            )

        raw = llm.invoke(full_prompt).content or ""
        raw = raw.strip()
        # Strip fences defensively (reasoning models sometimes wrap output)
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        # If the model still added surrounding text, grab the JSON object
        if not raw.startswith("{"):
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)

        try:
            return schema.model_validate_json(raw)
        except ValidationError as e:
            feedback = "\n".join(
                f"- {'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()
            )
            print(f"  ⚠️ attempt {attempt}/{max_retries} failed: {feedback[:120]}")

    raise RuntimeError(f"{schema.__name__} validation failed after {max_retries} attempts")

# ── 3. TOOLS (mock data — swap for real sources in capstone) ──
@tool
def search_logs(service: str, keyword: str) -> dict:
    """Search recent logs of a service for a keyword."""
    return {"matches": [
        f"10:42:03 ERROR timeout connecting to postgres-primary ({service})",
        f"10:41:58 WARN connection pool exhausted ({service})",
    ]}

@tool
def get_recent_deploys(service: str) -> dict:
    """List recent deployments for a service."""
    return {"deploys": [{"service": service, "version": "v2.14.3",
                         "time": "10:35", "author": "dana@corp"}]}

@tool
def get_metrics(service: str) -> dict:
    """Get current p95 latency, error rate, and active DB connections for a service."""
    return {"p95_latency_ms": 4200, "error_rate": 0.31, "db_conn_active": 100}

@tool
def check_runbook(service: str) -> str:
    """Read the operations runbook for a service. Consult before recommending an action."""
    return ("1. Check postgres-primary connection pool (max 100). "
            "2. If error_rate > 10%, roll back the last deploy. "
            "3. Escalate to dana@corp.")

INVESTIGATION_TOOLS = [search_logs, get_recent_deploys, get_metrics, check_runbook]

# ── 4. INVESTIGATOR SUB-AGENT (version-safe) ──────────────
SYSTEM = ("You investigate incidents. Use your tools to gather evidence. "
          "When done, end with a 3-line RCA draft: root cause / evidence / suggested action.")

if "prompt" in inspect.signature(create_react_agent).parameters:      # LangGraph ≥ 1.0
    investigator = create_react_agent(llm, INVESTIGATION_TOOLS, prompt=SYSTEM)
else:                                                                  # LangGraph 0.2.x
    investigator = create_react_agent(llm, INVESTIGATION_TOOLS,
                                      state_modifier=SystemMessage(content=SYSTEM))

# ── 5. STATE & SCHEMAS ────────────────────────────────────
class EscalationState(TypedDict):
    alert: str
    parsed: dict
    evidence: Annotated[list, operator.add]
    attempts: int
    verdict: str
    feedback: str
    report: dict
    approved_by: str
    outcome: str

class ParsedAlert(BaseModel):
    service: str
    severity: Literal["SEV1", "SEV2", "SEV3"]
    symptom: str = Field(max_length=100)

class Verdict(BaseModel):
    verdict: Literal["ok", "revise"]
    feedback: str = ""

class RCAReport(BaseModel):
    root_cause: str = Field(min_length=20)
    evidence: list[str] = Field(min_length=1)
    severity: Literal["SEV1", "SEV2", "SEV3"]
    next_action: Literal["rollback_deploy", "page_oncall", "monitor", "restart_service"]

WRITE_ACTIONS = {"rollback_deploy", "page_oncall", "restart_service"}

# ── 6. NODES ──────────────────────────────────────────────
def triage(state):
    parsed = llm_structured(f"Parse this alert:\n{state['alert']}", ParsedAlert)
    print(f"🏷️ triage: {parsed.model_dump()}")
    return {"parsed": parsed.model_dump()}

def investigate(state):
    task = f"Investigate {state['parsed']['service']}: {state['alert']}"
    if state.get("verdict") == "revise":
        task += f"\nYour previous RCA was rejected. Fix: {state['feedback']}"
    result = investigator.invoke({"messages": [HumanMessage(content=task)]})
    draft = result["messages"][-1].content
    print(f"🔍 investigate (attempt {state.get('attempts', 0) + 1})")
    return {"evidence": [draft], "attempts": state.get("attempts", 0) + 1}

def verify(state):
    draft = state["evidence"][-1]
    v = llm_structured(
        f"EVIDENCE:\n{state['evidence']}\n\nRCA DRAFT:\n{draft}\n\n"
        f"'ok' only if every claim is backed by evidence. Else 'revise' with feedback.",
        Verdict)
    print(f"🧐 verify: {v.verdict}" + (f" — {v.feedback}" if v.feedback else ""))
    return {"verdict": v.verdict, "feedback": v.feedback}

def write_rca(state):
    report = llm_structured(
        f"Convert this into a final RCA report:\n{state['evidence'][-1]}", RCAReport)
    print(f"📝 write_rca: action={report.next_action}")
    return {"report": report.model_dump()}

def human_gate(state):
    decision = interrupt({
        "action": state["report"]["next_action"],
        "root_cause": state["report"]["root_cause"],
    })
    ok = decision.get("approved")
    print(f"🧑 human gate: {'APPROVED' if ok else 'REJECTED'}")
    return {"approved_by": decision.get("approver") if ok else None,
            "outcome": "approved" if ok else "rejected"}

def execute(state):
    print(f"🔧 EXECUTING {state['report']['next_action']} (approved by {state['approved_by']})")
    return {"outcome": f"Executed {state['report']['next_action']}"}

def escalate_human(state):
    return {"outcome": f"🙋 Human takeover. Evidence bundle: {state['evidence']}"}

def auto_close(state):
    return {"outcome": "🤖 Low severity — sent docs link, ticket closed."}

# ── 7. ROUTERS ────────────────────────────────────────────
def route_triage(s):
    return "investigate" if s["parsed"]["severity"] in ("SEV1", "SEV2") else "auto_close"

def route_verify(s):
    if s["verdict"] == "revise" and s["attempts"] < 3:   # loop with exit
        return "investigate"
    return "write_rca"

def route_rca(s):
    return "human_gate" if s["report"]["next_action"] in WRITE_ACTIONS else "auto_close"

def route_gate(s):
    return "execute" if s["outcome"] == "approved" else "escalate_human"

# ── 8. GRAPH ──────────────────────────────────────────────
b = StateGraph(EscalationState)
for name, fn in [("triage", triage), ("investigate", investigate), ("verify", verify),
                 ("write_rca", write_rca), ("human_gate", human_gate), ("execute", execute),
                 ("escalate_human", escalate_human), ("auto_close", auto_close)]:
    b.add_node(name, fn)

b.add_edge(START, "triage")
b.add_conditional_edges("triage", route_triage,
                        {"investigate": "investigate", "auto_close": "auto_close"})
b.add_edge("investigate", "verify")
b.add_conditional_edges("verify", route_verify,
                        {"investigate": "investigate", "write_rca": "write_rca"})
b.add_conditional_edges("write_rca", route_rca,
                        {"human_gate": "human_gate", "auto_close": "auto_close"})
b.add_conditional_edges("human_gate", route_gate,
                        {"execute": "execute", "escalate_human": "escalate_human"})
b.add_edge("execute", END)
b.add_edge("escalate_human", END)
b.add_edge("auto_close", END)

app = b.compile(checkpointer=InMemorySaver())
print("✅ Graph compiled successfully")


# In[2]:


# ── 9. SEV1 PATH: investigate → gate → approve ────────────
config = {"configurable": {"thread_id": "esc-1"}, "recursion_limit": 30}

app.invoke(
    {"alert": "payments-api error rate 31% since 10:42, users can't checkout"},
    config,
)
print("\n⏸️ Paused at human gate. Approving now…\n")

final = app.invoke(Command(resume={"approved": True, "approver": "you@corp"}), config)
print("\n🏁 FINAL OUTCOME:", final["outcome"])

# ── 10. SEV3 PATH: auto-close ─────────────────────────────
config3 = {"configurable": {"thread_id": "esc-2"}, "recursion_limit": 30}
final3 = app.invoke(
    {"alert": "auth-service latency slightly up, no user impact"}, config3
)
print("\n🏁 FINAL OUTCOME:", final3["outcome"])


# In[ ]:


import os

LANGFUSE_SECRET_KEY = os.environ["LANGFUSE_SECRET_KEY"]
LANGFUSE_PUBLIC_KEY = os.environ["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

from langfuse.langchain import CallbackHandler
from langfuse import Langfuse

# 1. Initialize the global client with credentials
Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_BASE_URL
)


# Now initialize without passing keys directly
langfuse_handler = CallbackHandler()

config = {
    "configurable": {"thread_id": "trace-test-1"},
    "recursion_limit": 30,
    "callbacks": [langfuse_handler]
}
# Run the agent
app.invoke({"alert": "payments-api error rate 31% since 10:42"}, config)


# In[ ]:




