import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from typing import TypedDict, Annotated, List, Sequence
import operator
from tools import TOOLS

class AgentState(TypedDict):
    messages: Annotated[Sequence[dict], operator.add]
    action_required: dict
    user_role: str

def create_agent(groq_api_key: str, model_name: str = "llama3-8b-8192"):
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not provided.")
    
    llm = ChatGroq(model=model_name, groq_api_key=groq_api_key)
    llm_with_tools = llm.bind_tools(TOOLS)

    def should_continue(state: AgentState):
        messages = state['messages']
        last_message = messages[-1]
        
        if not last_message.tool_calls:
            return "end"
            
        return "continue"
        
    def call_model(state: AgentState):
        messages = list(state['messages']) # Make a copy
        user_role = state.get('user_role', 'Internal Agent')
        
        account_context = ""
        if "Northstar" in user_role:
            account_context = "Your account ID is ACCT-001 (Northstar Logistics)."
        elif "LumenWorks" in user_role:
            account_context = "Your account ID is ACCT-002 (LumenWorks)."
        else:
            account_context = "You are an internal ParcelPilot operations agent with access to all accounts."
            
        sys_msg_content = f'''You are an AI support agent for ParcelPilot. 
Your current active user role is: {user_role}.
{account_context}

Your goal is to investigate customer issues, answer support questions, and help operations staff.
You have tools to search documents and query operational data.
You can also escalate and update tickets, but these actions require user confirmation.
Always explain your reasoning based on the provided documents and data.
Prioritize 'CURRENT' policies over 'DEPRECATED' policies unless asked otherwise.
If a customer's specific agreement (like Northstar Logistics) contradicts a general policy, follow the customer's agreement.
If the user asks about "my orders", "my account", or "my tickets", automatically use their account ID to query the data without asking them for it.
'''
        
        # Add or replace system prompt
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=sys_msg_content)
        else:
            messages.insert(0, SystemMessage(content=sys_msg_content))
            
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
        
    def call_tool(state: AgentState):
        messages = state['messages']
        last_message = messages[-1]
        
        import tools as tools_module
        tools_module.CURRENT_ROLE = state.get('user_role', 'Internal Agent')
        
        tool_node = ToolNode(TOOLS)
        # Execute tools
        action_req = None
        
        # We need to manually execute to catch action required
        # For simplicity, if any tool returns ACTION_REQUIRED, we pause
        tool_responses = []
        for tool_call in last_message.tool_calls:
            tool = next(t for t in TOOLS if t.name == tool_call['name'])
            res = tool.invoke(tool_call['args'])
            if isinstance(res, str) and res.startswith("ACTION_REQUIRED:"):
                parts = res.split(":", 2)
                action_type = parts[1]
                args_str = parts[2]
                try:
                    import json
                    args = json.loads(args_str)
                except:
                    args = {"raw": args_str}
                    
                action_req = {
                    "tool_call_id": tool_call['id'],
                    "action_type": action_type,
                    "args": args,
                    "original_call": tool_call
                }
                tool_responses.append(ToolMessage(content="Action pending user confirmation.", tool_call_id=tool_call['id']))
                break # Only one action at a time
            else:
                tool_responses.append(ToolMessage(content=str(res), tool_call_id=tool_call['id']))
                
        if action_req:
            return {"messages": tool_responses, "action_required": action_req}
            
        return {"messages": tool_responses}
        
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)
    
    workflow.add_edge(START, "agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END
        }
    )
    
    workflow.add_edge("action", "agent")
    
    return workflow.compile()
