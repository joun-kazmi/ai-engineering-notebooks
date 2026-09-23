#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from dotenv import load_dotenv
load_dotenv()  # picks up .env from the repo root
from typing import Literal, TypedDict
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

llm = ChatOpenAI(model="openai/gpt-oss-20b",
                 base_url="https://integrate.api.nvidia.com/v1",
                 api_key=os.environ["NVIDIA_API_KEY"])

# ── THE WHITEBOARD ─────────────────────────────────────────
class TicketState(TypedDict):
    ticket: str
    severity: str
    rca: str
    attempts: int
    verdict: str
    outcome: str

# ── THE BOXES (nodes) — each returns ONLY its updates ──────
class Triage(BaseModel):
    severity: Literal["critical", "minor"]

def classify(state: TicketState) -> dict:          # reads: ticket
    t = llm.with_structured_output(Triage).invoke(
        f"Classify this incident:\n{state['ticket']}")
    return {"severity": t.severity}                # writes: severity

def investigate(state: TicketState) -> dict:       # reads: ticket, rca
    prev = state.get("rca", "")
    draft = llm.invoke(
        f"Draft a 2-line RCA (root cause + action) for: {state['ticket']}\n"
        f"Previous draft: {prev or 'none'}. Improve it.")
    return {"rca": draft.content,
            "attempts": state["attempts"] + 1}     # writes: rca, attempts

class Judge(BaseModel):
    verdict: Literal["approve", "revise"]

def review(state: TicketState) -> dict:            # reads: rca
    v = llm.with_structured_output(Judge).invoke(
        f"Approve only if the RCA names a concrete root cause AND action:\n{state['rca']}")
    return {"verdict": v.verdict}                  # writes: verdict

def escalate(state: TicketState) -> dict:
    return {"outcome": f"📟 Paged on-call. RCA: {state['rca']}"}

def auto_reply(state: TicketState) -> dict:
    return {"outcome": "🤖 Sent docs link + canned reply."}

# ── THE LABELED-ARROW JUNCTIONS (routers) ──────────────────
# Routers are TINY and DETERMINISTIC: read state, return a label. No LLM calls!
def route_severity(state: TicketState) -> str:
    return state["severity"]            # "critical" | "minor"

def route_verdict(state: TicketState) -> str:
    if state["verdict"] == "revise" and state["attempts"] < 3:
        return "revise"                 # loop back
    return "approve"                    # EVERY loop needs this exit arrow

# ── DRAW THE PICTURE IN CODE ───────────────────────────────
b = StateGraph(TicketState)
b.add_node("classify", classify)        # ▭ boxes
b.add_node("investigate", investigate)
b.add_node("review", review)
b.add_node("escalate", escalate)
b.add_node("auto_reply", auto_reply)

b.add_edge(START, "classify")           # ● → first box
b.add_conditional_edges("classify", route_severity,
    {"critical": "investigate",         # →label→ arrows
     "minor": "auto_reply"})
b.add_edge("investigate", "review")     # → plain arrow
b.add_conditional_edges("review", route_verdict,
    {"revise": "investigate",           # the BACKWARD arrow (loop)
     "approve": "escalate"})
b.add_edge("escalate", END)             # → ◉
b.add_edge("auto_reply", END)

app = b.compile()
print(app.get_graph().draw_mermaid())   # ← prints the mermaid from 3b!


# In[2]:


for step in app.stream(
    {"ticket": "payments-api error rate 31% since 10:42, checkouts failing",
     "attempts": 0, "rca": "", "severity": "", "verdict": "", "outcome": ""},
    stream_mode="updates"):
    for node, updates in step.items():
        print(f"🟢 {node} wrote: {updates}")


# In[ ]:




