import json
import operator
import re
import sys
import uuid
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_engineering.config import NO_THINK, get_settings, make_chat_client

# ══════════════════════════════════════════════════════════
# 1. MODEL SETUP
# ══════════════════════════════════════════════════════════
_settings = get_settings()
llm_client = make_chat_client(_settings)
RAW_MODEL = _settings.resolved_model

# ══════════════════════════════════════════════════════════
# 2. RELIABILITY HELPER (Schema-injected)
# ══════════════════════════════════════════════════════════
def llm_structured(prompt: str, schema, max_retries: int = 3):
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    # Constrain decoding to the schema; the prompt alone is not enough for Nemotron
    # (it often wraps the answer in {"properties": ...} or adds trailing text)
    response_format = {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}
    feedback = ""
    for attempt in range(1, max_retries + 1):
        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with ONLY a raw JSON object matching this exact schema:\n"
            f"{schema_json}\n"
            f"No markdown, no code fences, no commentary — JSON only."
        )
        if feedback:
            full_prompt += f"\n\nYour previous output FAILED validation:\n{feedback}\nFix these issues."
        
        raw = llm_client.chat.completions.create(model=RAW_MODEL, extra_body=NO_THINK, messages=[{"role": "user", "content": full_prompt}], response_format=response_format).choices[0].message.content or ""
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        if not raw.startswith("{"):
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match: raw = match.group(0)
        try:
            return schema.model_validate_json(raw)
        except ValidationError as e:
            feedback = "\n".join(f"- {'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            print(f"  ⚠️ attempt {attempt}/{max_retries} failed: {feedback[:120]}")
    raise RuntimeError(f"{schema.__name__} validation failed after {max_retries} attempts")

# ══════════════════════════════════════════════════════════
# 3. TOOLS & INVESTIGATOR (raw SDK tool-calling loop)
# ══════════════════════════════════════════════════════════
SYSTEM = ("You investigate incidents. Use your tools to gather evidence. "
          "When done, end with a 3-line RCA draft: root cause / evidence / suggested action.")

def _search_logs(service, keyword="ERROR"):
    return {"matches": [f"10:42:03 ERROR timeout connecting to postgres-primary ({service})"]}
def _get_recent_deploys(service):
    return {"deploys": [{"version": "v2.14.3", "time": "10:35", "author": "dana@corp"}]}
def _get_metrics(service):
    return {"p95_latency_ms": 4200, "error_rate": 0.31, "db_conn_active": 100}
def _check_runbook(service):
    return "1. Check DB pool. 2. Rollback if error > 10%. 3. Page dana."

RAW_TOOL_REGISTRY = {
    "search_logs": _search_logs, "get_recent_deploys": _get_recent_deploys,
    "get_metrics": _get_metrics, "check_runbook": _check_runbook
}

RAW_TOOLS = [
    {"type": "function", "function": {"name": "search_logs", "description": "Search logs", 
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}, "keyword": {"type": "string"}}, "required": ["service"]}}},
    {"type": "function", "function": {"name": "get_recent_deploys", "description": "Get deploys",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}}},
    {"type": "function", "function": {"name": "get_metrics", "description": "Get metrics",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}}},
    {"type": "function", "function": {"name": "check_runbook", "description": "Read runbook",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}}},
]

def run_investigation(task: str, max_iterations: int = 8) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    for _ in range(max_iterations):
        resp = llm_client.chat.completions.create(model=RAW_MODEL, extra_body=NO_THINK, messages=messages, tools=RAW_TOOLS, tool_choice="auto")
        msg = resp.choices[0].message
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]
        messages.append(assistant_msg)
        if not msg.tool_calls: return msg.content or ""
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            try: result = RAW_TOOL_REGISTRY[tc.function.name](**args)
            except Exception as e: result = {"error": str(e)}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return "Max iterations reached."

# ══════════════════════════════════════════════════════════
# 4. SCHEMAS & STATE
# ══════════════════════════════════════════════════════════
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

class GenerateRequest(BaseModel):
    prompt: str


WRITE_ACTIONS = {"rollback_deploy", "page_oncall", "restart_service"}

# ══════════════════════════════════════════════════════════
# 5. NODES & ROUTERS
# ══════════════════════════════════════════════════════════
def triage(state):
    parsed = llm_structured(f"Parse this alert:\n{state['alert']}", ParsedAlert)
    return {"parsed": parsed.model_dump()}

def investigate(state):
    task = f"Investigate {state['parsed']['service']}: {state['alert']}"
    if state.get("verdict") == "revise": task += f"\nFix: {state['feedback']}"
    draft = run_investigation(task)
    return {"evidence": [draft], "attempts": state.get("attempts", 0) + 1}

def verify(state):
    v = llm_structured(f"EVIDENCE:\n{state['evidence']}\n\nRCA DRAFT:\n{state['evidence'][-1]}\n\n'ok' if backed by evidence. Else 'revise' with feedback.", Verdict)
    return {"verdict": v.verdict, "feedback": v.feedback}

def write_rca(state):
    report = llm_structured(f"Convert this into a final RCA report:\n{state['evidence'][-1]}", RCAReport)
    return {"report": report.model_dump()}

def human_gate(state):
    # This pauses the graph and returns control to FastAPI
    decision = interrupt({"action": state["report"]["next_action"], "root_cause": state["report"]["root_cause"]})
    ok = decision.get("approved")
    return {"approved_by": decision.get("approver") if ok else None, "outcome": "approved" if ok else "rejected"}

def execute(state): return {"outcome": f"Executed {state['report']['next_action']}"}
def escalate_human(state): return {"outcome": f"Human takeover. Evidence: {state['evidence']}"}
def auto_close(state): return {"outcome": "Low severity — closed."}

def route_triage(s): return "investigate" if s["parsed"]["severity"] in ("SEV1", "SEV2") else "auto_close"
def route_verify(s): return "investigate" if s["verdict"] == "revise" and s["attempts"] < 3 else "write_rca"
def route_rca(s): return "human_gate" if s["report"]["next_action"] in WRITE_ACTIONS else "auto_close"
def route_gate(s): return "execute" if s["outcome"] == "approved" else "escalate_human"

# ══════════════════════════════════════════════════════════
# 6. COMPILE THE LANGGRAPH APP (This fixes your error!)
# ══════════════════════════════════════════════════════════
b = StateGraph(EscalationState)
for name, fn in [("triage", triage), ("investigate", investigate), ("verify", verify),
                 ("write_rca", write_rca), ("human_gate", human_gate), ("execute", execute),
                 ("escalate_human", escalate_human), ("auto_close", auto_close)]:
    b.add_node(name, fn)

b.add_edge(START, "triage")
b.add_conditional_edges("triage", route_triage, {"investigate": "investigate", "auto_close": "auto_close"})
b.add_edge("investigate", "verify")
b.add_conditional_edges("verify", route_verify, {"investigate": "investigate", "write_rca": "write_rca"})
b.add_conditional_edges("write_rca", route_rca, {"human_gate": "human_gate", "auto_close": "auto_close"})
b.add_conditional_edges("human_gate", route_gate, {"execute": "execute", "escalate_human": "escalate_human"})
b.add_edge("execute", END)
b.add_edge("escalate_human", END)
b.add_edge("auto_close", END)

# THIS IS THE MISSING PIECE! We define 'app' right here.
app = b.compile(checkpointer=InMemorySaver())

# ══════════════════════════════════════════════════════════
# 7. FASTAPI WRAPPER
# ══════════════════════════════════════════════════════════
app_fastapi = FastAPI(title="Escalation Agent API")
active_threads = {}

class AlertPayload(BaseModel):
    alert_text: str

class ApprovalPayload(BaseModel):
    thread_id: str
    approved: bool
    approver: str



@app_fastapi.post("/generate")
async def generate(req: GenerateRequest):
    response = llm_client.chat.completions.create(
        model=RAW_MODEL,
        extra_body=NO_THINK,
        messages=[{"role": "user", "content": req.prompt}],
        max_tokens=200,
        temperature=0.3
    )
    return {"response": response.choices[0].message.content}
    
@app_fastapi.post("/alert")
def receive_alert(payload: AlertPayload):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    
    try:
        # Run until it hits the human_gate interrupt
        state = app.invoke({"alert": payload.alert_text, "attempts": 0, "evidence": []}, config)
        active_threads[thread_id] = config
        
        # Check if it auto-closed or hit the gate
        if state.get("outcome") and "Low severity" in state["outcome"]:
            return {"status": "completed", "outcome": state["outcome"]}
            
        return {
            "status": "awaiting_approval",
            "thread_id": thread_id,
            "proposed_action": state.get("report", {}).get("next_action"),
            "root_cause": state.get("report", {}).get("root_cause")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app_fastapi.post("/approve")
def approve_action(payload: ApprovalPayload):
    config = active_threads.get(payload.thread_id)
    if not config:
        raise HTTPException(status_code=404, detail="Thread not found")
        
    resume_data = {"approved": payload.approved, "approver": payload.approver}
    final_state = app.invoke(Command(resume=resume_data), config)
    
    del active_threads[payload.thread_id]
    return {"status": "finalized", "outcome": final_state.get("outcome")}