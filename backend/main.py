from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
from agent import create_agent
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ParcelPilot Support Agent")

# Mount static files for frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ConfirmActionRequest(BaseModel):
    tool_call_id: str
    action_approved: bool
    session_id: str = "default"

# In-memory session state for simplicity
sessions = {}
agents = {}

def get_agent(api_key: str, model_name: str = "llama3-8b-8192"):
    try:
        return create_agent(api_key, model_name)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    if not os.path.exists("static/index.html"):
        return {"error": "Frontend not built yet"}
    return FileResponse("static/index.html")

@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set in the .env file")
        
    user_role = request.headers.get("x-user-role", "Internal Agent")
    agent = get_agent(api_key, model_name)
    
    if req.session_id not in sessions:
        sessions[req.session_id] = {"messages": [], "user_role": user_role}
    else:
        sessions[req.session_id]["user_role"] = user_role
        
    state = sessions[req.session_id]
    
    from langchain_core.messages import HumanMessage
    state["messages"].append(HumanMessage(content=req.message))
    
    # Run agent
    try:
        old_len = len(state["messages"])
        result = agent.invoke(state)
        # update state
        sessions[req.session_id] = result
        
        new_msgs = result["messages"][old_len:]
        
        # Check if action required
        action_req = result.get("action_required")
        if action_req:
            return JSONResponse({
                "status": "action_required",
                "action": action_req,
                "messages": [{"role": m.type, "content": m.content, "tool_calls": getattr(m, "tool_calls", [])} for m in new_msgs]
            })
            
        return JSONResponse({
            "status": "success",
            "messages": [{"role": m.type, "content": str(m.content), "tool_calls": getattr(m, "tool_calls", [])} for m in new_msgs]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/confirm_action")
def confirm_action(req: ConfirmActionRequest, request: Request):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    state = sessions[req.session_id]
    action_req = state.get("action_required")
    
    if not action_req or action_req["tool_call_id"] != req.tool_call_id:
        raise HTTPException(status_code=400, detail="Invalid tool call ID or no action pending")
        
    from langchain_core.messages import ToolMessage
    # Replace the pending tool message
    messages = state["messages"]
    for i, msg in enumerate(messages):
        if getattr(msg, "tool_call_id", None) == req.tool_call_id:
            if req.action_approved:
                messages[i] = ToolMessage(content=f"Action '{action_req['action_type']}' APPROVED and executed successfully.", tool_call_id=req.tool_call_id)
            else:
                messages[i] = ToolMessage(content=f"Action '{action_req['action_type']}' DENIED by user.", tool_call_id=req.tool_call_id)
            break
            
    # Clear action required
    state["action_required"] = None
    
    # Continue agent
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set in the .env file")
        
    user_role = request.headers.get("x-user-role", "Internal Agent")
    agent = get_agent(api_key, model_name)
    
    # Update role in case it changed
    state["user_role"] = user_role
    
    old_len = len(state["messages"])
    result = agent.invoke(state)
    sessions[req.session_id] = result
    
    new_msgs = result["messages"][old_len:]
    
    return JSONResponse({
        "status": "success",
        "messages": [{"role": m.type, "content": str(m.content), "tool_calls": getattr(m, "tool_calls", [])} for m in new_msgs]
    })
