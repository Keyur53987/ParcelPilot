import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool
from pydantic import BaseModel, Field

CURRENT_ROLE = "Internal Agent"

DATA_PATH = "../AI Agent Assessment - Candidate Pack/ParcelPilot_Assessment_Data.xlsx"

try:
    accounts_df = pd.read_excel(DATA_PATH, sheet_name="accounts")
    orders_df = pd.read_excel(DATA_PATH, sheet_name="orders")
    tickets_df = pd.read_excel(DATA_PATH, sheet_name="tickets")
except Exception as e:
    print(f"Failed to load Excel data: {e}")

try:
    client = chromadb.PersistentClient(path="./chroma_db")
    sentence_transformer_ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_collection(name="parcelpilot_docs", embedding_function=sentence_transformer_ef)
except Exception as e:
    print(f"Failed to load ChromaDB: {e}")
    collection = None

@tool
def search_documents(query: str, include_deprecated: bool = False):
    """
    Search the ParcelPilot policies, SOPs, and agreements for answers.
    Set include_deprecated to True only if specifically asked about old policies.
    """
    if not collection:
        return "Vector database not initialized."
    
    where_clause = {}
    if not include_deprecated:
        where_clause = {"is_deprecated": False}
        
    results = collection.query(
        query_texts=[query],
        n_results=4,
        where=where_clause if where_clause else None
    )
    
    docs = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        docs.append(f"Source: {meta['source']}\nContent: {doc}")
        
    return "\n\n---\n\n".join(docs)

@tool
def query_operational_data(query: str):
    """
    Query operational data (accounts, orders, tickets).
    Pass the ID of the entity you want to look up, e.g., 'ORD-1001' or 'ACCT-001' or 'TKT-501'.
    """
    role = CURRENT_ROLE
    allowed_account = None
    if "Northstar" in role:
        allowed_account = "ACCT-001"
    elif "LumenWorks" in role:
        allowed_account = "ACCT-002"
        
    query = query.upper()
    result = ""
    if "ORD-" in query:
        order_id = [word for word in query.split() if "ORD-" in word][0]
        order = orders_df[orders_df['order_id'] == order_id]
        if not order.empty:
            if allowed_account and order.iloc[0]['account_id'] != allowed_account:
                return f"[ACCESS DENIED] You do not have permission to view data for order {order_id}."
            result += f"Order Data:\n{order.iloc[0].to_json()}\n"
    if "ACCT-" in query:
        acct_id = [word for word in query.split() if "ACCT-" in word][0]
        acct = accounts_df[accounts_df['account_id'] == acct_id]
        if not acct.empty:
            if allowed_account and acct_id != allowed_account:
                return f"[ACCESS DENIED] You do not have permission to view data for account {acct_id}."
            result += f"Account Data:\n{acct.iloc[0].to_json()}\n"
    if "TKT-" in query:
        tkt_id = [word for word in query.split() if "TKT-" in word][0]
        tkt = tickets_df[tickets_df['ticket_id'] == tkt_id]
        if not tkt.empty:
            if allowed_account and tkt.iloc[0]['account_id'] != allowed_account:
                return f"[ACCESS DENIED] You do not have permission to view data for ticket {tkt_id}."
            result += f"Ticket Data:\n{tkt.iloc[0].to_json()}\n"
            
    if not result:
        return f"No data found for the query: {query}. Make sure to provide a valid ID like ORD-1001."
    return result

class EscalateTicketInput(BaseModel):
    ticket_id: str = Field(description="The ticket ID to escalate")
    reason: str = Field(description="Reason for escalation")

@tool("escalate_ticket", args_schema=EscalateTicketInput)
def escalate_ticket(ticket_id: str, reason: str):
    """
    Escalates a ticket. Requires user confirmation.
    """
    return f"ACTION_REQUIRED:escalate_ticket:{json.dumps({'ticket_id': ticket_id, 'reason': reason})}"

class UpdateTicketInput(BaseModel):
    ticket_id: str = Field(description="The ticket ID to update")
    status: str = Field(description="New status for the ticket")

@tool("update_ticket", args_schema=UpdateTicketInput)
def update_ticket(ticket_id: str, status: str):
    """
    Updates a ticket's status. Requires user confirmation.
    """
    return f"ACTION_REQUIRED:update_ticket:{json.dumps({'ticket_id': ticket_id, 'status': status})}"

TOOLS = [search_documents, query_operational_data, escalate_ticket, update_ticket]
