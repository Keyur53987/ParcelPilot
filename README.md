# ParcelPilot AI Support Agent

This repository contains the solution for the **CalQuity AI Engineer First-Round Assessment**. 

It implements an **Internal Support/Operations Chatbot** that helps authorized ParcelPilot staff investigate customer issues, answer support questions, and interact safely with operational data. It also supports mocked multi-tenant **Customer-facing** roles to demonstrate strict data segregation.

## 🚀 Setup and Run Instructions

### Prerequisites
- Python 3.10+
- A valid Groq API Key.

### Installation
1. Clone this repository.
2. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the `backend` directory with your Groq credentials:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=openai/gpt-oss-20b
   ```
5. Build the local ChromaDB vector store (ingests the PDF Data Pack):
   ```bash
   python build_vectorstore.py
   ```

### Running the Application
1. Start the FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
2. Open your browser and navigate to: **http://127.0.0.1:8000/**
3. In the UI header:
   - Select your desired **Role** (e.g., Internal Agent vs. Customer: Northstar).
   - Start chatting!

---

## 🏗️ Architecture Note

### 1. Agent Design
The agent is built using a **LangGraph State Machine** implementing a stateful **ReAct (Reason + Act)** pattern. Rather than a simple zero-shot execution, LangGraph allows the agent to think, use a tool, observe the result, and iterate in a loop until it reaches a confident answer. The graph maintains a strict `AgentState` containing the conversation history, the active user role, and any pending action intercepts.

### 2. Tool Design
The system utilizes three distinct tool categories:
1. `search_documents(query)`: Performs semantic RAG over the policy PDFs.
2. `query_operational_data(query)`: A deterministic lookup tool that scans the in-memory pandas DataFrames for Account, Order, and Ticket IDs. 
3. `escalate_ticket` & `update_ticket`: State-changing action tools. When invoked, these tools **do not execute immediately**. Instead, they pause the LangGraph state machine by returning an `ACTION_REQUIRED` payload, returning control to the frontend UI for human confirmation.

### 3. Document and Structured-Data Handling
- **Documents (Unstructured):** Processed via `PyPDFLoader`, chunked using `RecursiveCharacterTextSplitter`, and embedded into a local `ChromaDB` vector store using lightweight `all-MiniLM-L6-v2` embeddings for blazing fast, localized semantic search.
- **Structured Data:** The `ParcelPilot_Assessment_Data.xlsx` file is loaded into Pandas DataFrames on server boot. This provides fast, deterministic lookups using exact ID matching (e.g., `ORD-1001`), ensuring the agent doesn't hallucinate tabular data.

### 4. Source Reliability and Conflict Handling
The system resolves source conflicts deterministically via strict System Prompting:
- **Contract vs. Policy:** The prompt explicitly instructs the LLM that specific Enterprise Agreements (e.g., Northstar) strictly override general SOPs.
- **Freshness:** The agent is instructed to prioritize policies marked `CURRENT` over those marked `DEPRECATED`.
- **Data Segregation:** Role-based access control is actively enforced at the **Python Tool Layer** (bypassing the LLM). When the LLM attempts to query operational data, the tool cross-references the targeted `account_id` with the current `user_role` injected into the agent's state. If a customer attempts to query another customer's data, the tool hard-blocks the request and returns an `[ACCESS DENIED]` string.

### 5. Major Technical Trade-offs
- **Vanilla JS Frontend vs. React:** Chose a monolithic Vanilla JS/Tailwind frontend served directly via FastAPI to ensure maximum portability and ease of testing for evaluators (zero Node.js dependencies required).
- **In-Memory Data vs. Database:** Opted for Pandas DataFrames reading the Excel file directly into memory rather than spinning up a PostgreSQL instance, optimizing for setup simplicity while maintaining fast query performance for the dataset size.

---

## 💡 Product Note

### 1. Additional Client Problem Addressed
**Problem 2: Trust and Reliability.** 
A major concern was the fear of a confidently incorrect AI taking irreversible actions or giving out incorrect policies. This was addressed through three core product decisions:
- **Human-in-the-Loop Intercepts:** State-changing actions are strictly intercepted. The agent prepares the escalation payload but forces a distinct UI modal requiring explicit human approval.
- **Deterministic Data Isolation:** Customer data silos are strictly enforced at the lowest backend tier. The agent literally cannot leak data between customers because the Python function denies the read request before the LLM even sees it.
- **Conflict Guidelines:** The model is explicitly tuned to acknowledge document deprecation statuses and hierarchy (Agreements > SOPs).

### 2. Anything else you would build for ParcelPilot
If development continued, I would build out **Problem 1: Proactive Issue Detection**. We could deploy a scheduled background cron job (or an asynchronous Celery task) that uses an LLM to sweep incoming tickets every 15 minutes, cluster them semantically, and fire Slack alerts to the Operations Team if a sudden spike of identical product issues occurs, rather than waiting for humans to manually notice the trend.

### 3. What was intentionally left out of the submission
- **Production Authentication:** True JWT/OAuth authentication was omitted in favor of a UI dropdown to allow evaluators to easily test the Multi-Tenant access control features without needing to create mock user accounts or manage database migrations.
- **Persistent Chat History:** Chat history is kept in local memory (`sessions` dictionary). In a real environment, this would be backed by Redis or PostgreSQL to persist across server restarts.

### 4. Metric for Success
**Action Acceptance Rate:** 
The percentage of state-changing actions (escalations/updates) proposed by the agent that are **Approved** vs. **Rejected** by the human operator via the UI modal. A high acceptance rate indicates the agent is accurately reasoning through policies and proposing correct solutions, directly reflecting high Trust and Reliability.

---

## 🤖 AI Tool Usage
During this project, AI coding assistants were leveraged to maximize development speed and handle boilerplate generation:
- **Google Antigravity (Agentic Coding Assistant):** Used as a highly autonomous pair-programmer to scaffold the FastAPI backend, construct the LangGraph state machine, parse the initial PDFs into ChromaDB, and build the Tailwind CSS UI components. It was actively guided through architectural decisions, debugging dependency version conflicts, and implementing the precise context-injected data silo logic for the Role-Based Access Control requirement.
