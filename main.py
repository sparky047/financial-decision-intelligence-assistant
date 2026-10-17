import os
import re
import time
import sqlite3
import numpy as np
import pandas as pd
from typing import List, Optional, Any
from pandas.errors import DatabaseError
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types

# 1. Initialize API Client
client = genai.Client()

app = FastAPI(
    title="Financial Decision Intelligence Assistant API",
    description="Hybrid Text-to-SQL and RAG assistant with read-only SQLite guardrails.",
    version="1.0.0"
)

# 2. Database Schema Context
DB_PATH = "financial_data.db"
DB_SCHEMA_CONTEXT = """
Database Schema:
1. companies (company_id INTEGER PRIMARY KEY, ticker TEXT UNIQUE, name TEXT, sector TEXT);
2. quarterly_financials (financial_id INTEGER PRIMARY KEY, company_id INTEGER, quarter TEXT, fiscal_year INTEGER, revenue_usd REAL, net_income_usd REAL, operating_expense_usd REAL);
3. credit_risk (risk_id INTEGER PRIMARY KEY, company_id INTEGER, rating_agency TEXT, credit_score INTEGER, risk_category TEXT, default_probability REAL);
"""

SYSTEM_PROMPT = f"""
You are a specialized SQL assistant for a financial analytics database.
Given a user's question, output ONLY a valid SQLite read-only query using the provided schema.

Rules:
1. Return ONLY the raw SQL query. No markdown formatting (no ```sql or ```).
2. Only write SELECT statements. Never write INSERT, UPDATE, DELETE, DROP, or ALTER.
3. Correctly perform JOIN operations on `company_id`.

{DB_SCHEMA_CONTEXT}
"""

# 3. Policy Document Repository
DOCUMENTS = [
    {
        "doc_id": "DOC-101",
        "company": "Novastone Logistics",
        "title": "Q3 2025 Credit Risk & Debt Covenant Review",
        "content": "Novastone Logistics maintains a credit rating of 720 (Moody's). The company has complied with all debt covenants for Q3 2025. Working capital facility renewals are scheduled for Q1 2026, with interest rate margins pegged at SOFR + 180 bps."
    },
    {
        "doc_id": "DOC-102",
        "company": "Apex Cloud Systems",
        "title": "FY2025 Enterprise Cloud Expansion Policy",
        "content": "Apex Cloud Systems holds an S&P rating of 810 (Minimal Risk). All capital expenditures exceeding $5M require dual-signoff from the Risk Committee. Current liquidity ratios exceed covenant minimums by 2.4x."
    },
    {
        "doc_id": "DOC-103",
        "company": "Vantage Retail Group",
        "title": "Turnaround Strategy & Inventory Impairment Audit",
        "content": "Vantage Retail Group was rated 590 by Fitch (Moderate Risk) following Q2 operating losses. The advisory board mandated a 12% reduction in non-essential SG&A expenses and initiated renegotiation of long-term store lease agreements."
    }
]

# 4. Core Processing Helpers
def get_text_embedding(text: str) -> np.ndarray:
    models_to_try = ["gemini-embedding-001", "gemini-embedding-2"]
    for m in models_to_try:
        try:
            res = client.models.embed_content(model=m, contents=text)
            if hasattr(res, "embeddings") and res.embeddings:
                return np.array(res.embeddings[0].values)
            elif hasattr(res, "embedding"):
                return np.array(res.embedding.values)
        except Exception:
            continue
    raise RuntimeError("Failed to generate embedding.")

# Precompute document vectors on startup
for doc in DOCUMENTS:
    doc["embedding"] = get_text_embedding(doc["content"])

def execute_safe_query(query: str, db_path: str = DB_PATH):
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.execute("PRAGMA query_only = ON;")
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except (sqlite3.OperationalError, DatabaseError, Exception) as e:
        raise HTTPException(status_code=400, detail=f"Database Guardrail Violation: {e}")

# USING THE EXACT MODELS THAT WORKED IN YOUR NOTEBOOK
WORKING_MODELS = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-pro"]

def generate_sql(question: str) -> str:
    last_err = None
    for m in WORKING_MODELS:
        try:
            res = client.models.generate_content(
                model=m,
                contents=question,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.0)
            )
            sql = res.text.strip()
            sql = re.sub(r"^```(sql)?\s*", "", sql, flags=re.IGNORECASE)
            return re.sub(r"\s*```$", "", sql).strip()
        except Exception as e:
            last_err = e
            time.sleep(1)
            continue
    raise HTTPException(status_code=503, detail=f"Text-to-SQL error: {last_err}")

def retrieve_top_document(query: str):
    q_vec = get_text_embedding(query).reshape(1, -1)
    scored = []
    for doc in DOCUMENTS:
        d_vec = doc["embedding"].reshape(1, -1)
        sim = cosine_similarity(q_vec, d_vec)[0][0]
        scored.append((sim, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

def synthesize_grounded_answer(user_question: str, db_data: list, doc: dict) -> str:
    prompt = f"""
You are a Financial Decision Intelligence Assistant.
Synthesize a concise, grounded executive summary answering the question using ONLY the provided evidence.

Structured DB Evidence:
{db_data}

Unstructured Policy Evidence:
Document [{doc['doc_id']}] {doc['title']}: {doc['content']}

Question:
{user_question}

Instructions:
- Provide a direct answer.
- Explicitly cite database metrics and the Document ID.
- Do not extrapolate beyond the provided text.
"""
    last_err = None
    for m in WORKING_MODELS:
        try:
            res = client.models.generate_content(
                model=m,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            return res.text.strip()
        except Exception as e:
            last_err = e
            time.sleep(1)
            continue
    raise HTTPException(status_code=503, detail=f"Synthesis error: {last_err}")

# 5. Pydantic Schemas
class QueryRequest(BaseModel):
    question: str

class StructuredMetric(BaseModel):
    name: Optional[str] = None
    revenue_usd: Optional[float] = None
    net_income_usd: Optional[float] = None
    credit_score: Optional[int] = None
    risk_category: Optional[str] = None
    rating_agency: Optional[str] = None
    default_probability: Optional[float] = None

class EvidenceSource(BaseModel):
    source_type: str
    source_id: str
    details: Any

class QueryResponse(BaseModel):
    query: str
    status: str = "success"
    structured_metrics: List[StructuredMetric]
    retrieved_policy_id: str
    evidence_trail: List[EvidenceSource]
    executive_summary: str

# 6. API Endpoints
@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Financial Decision Intelligence Assistant API"}

@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    # Step A: SQL Generation & Execution
    sql_query = generate_sql(request.question)
    df_result = execute_safe_query(sql_query)
    records = df_result.to_dict(orient="records")

    # Step B: Policy Vector Retrieval
    best_doc = retrieve_top_document(request.question)

    # Step C: Synthesize Grounded Output
    executive_summary = synthesize_grounded_answer(request.question, records, best_doc)

    # Step D: Package into Pydantic Contract
    metrics = [StructuredMetric(**r) for r in records]
    evidence = [
        EvidenceSource(source_type="relational_db", source_id="sqlite_financial_data", details={"sql": sql_query}),
        EvidenceSource(source_type="unstructured_policy", source_id=best_doc["doc_id"], details={"title": best_doc["title"]})
    ]

    return QueryResponse(
        query=request.question,
        structured_metrics=metrics,
        retrieved_policy_id=best_doc["doc_id"],
        evidence_trail=evidence,
        executive_summary=executive_summary
    )
