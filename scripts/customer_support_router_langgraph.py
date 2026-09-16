#!/usr/bin/env python
# coding: utf-8

# In[6]:


from typing import TypedDict

class AgentState(TypedDict):
    query: str
    category: str
    response: str


# In[7]:


def classify(state):
    # In a real app, you'd call an LLM here.
    # For demo, we'll just check keywords.
    query = state["query"].lower()
    if "price" in query or "buy" in query:
        category = "sales"
    else:
        category = "support"
    return {"category": category}

def sales_agent(state):
    # Simulate a sales response
    return {"response": f"Sales: We have great deals! You asked: {state['query']}"}

def support_agent(state):
    # Simulate a support response
    return {"response": f"Support: We're here to help! You asked: {state['query']}"}


# In[8]:


def route_after_classify(state):
    if state["category"] == "sales":
        return "sales_agent"
    else:
        return "support_agent"


# In[9]:


from langgraph.graph import StateGraph, END

# Create graph builder
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("classify", classify)
builder.add_node("sales_agent", sales_agent)
builder.add_node("support_agent", support_agent)

# Set entry point
builder.set_entry_point("classify")

# Add conditional edges from "classify"
builder.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "sales_agent": "sales_agent",
        "support_agent": "support_agent"
    }
)

# Add edges from agents to END
builder.add_edge("sales_agent", END)
builder.add_edge("support_agent", END)

# Compile the graph
app = builder.compile()


# In[10]:


# Example input
result = app.invoke({"query": "I want to buy a laptop"})
print(result)
# Output: {'query': 'I want to buy a laptop', 'category': 'sales', 'response': 'Sales: We have great deals! You asked: I want to buy a laptop'}


# In[ ]:




