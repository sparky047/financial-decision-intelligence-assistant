# Financial Decision Intelligence Assistant

A hybrid decision-support system integrating **Relational Text-to-SQL**, **Semantic Document Retrieval (RAG)**, and **Strict Pydantic Validation** to provide evidence-grounded financial analytics with zero hallucination.

---

## Architecture Overview
User Natural Language Query
│
├──► Text-to-SQL Engine (Gemini API) ──► Read-Only SQLite DB (Safeguarded)
│                                              │
├──► Semantic Search (Cosine Similarity) ──────► Corporate Credit Policies
│                                              │
▼                                              ▼
Hybrid Synthesis Engine ◄────────────────────────────────┘
│
▼
Pydantic Validation Layer ──► Typed JSON Response with Evidence Audit Trail

---

## Core Capabilities

- **Database-Level Read-Only Safeguards:** Enforces strict execution safety via SQLite URI connection flags (`mode=ro`) and `PRAGMA query_only = ON` to block any write, update, or drop statements at the driver level.
- **Dynamic Text-to-SQL:** Generates SQL queries across multi-table schemas (`companies`, `quarterly_financials`, `credit_risk`) to compute metrics such as revenue, margins, and credit scores.
- **Semantic Vector Retrieval (RAG):** Embeds unstructured corporate policy memos and retrieves relevant governance covenants using cosine similarity.
- **Evidence-Grounded Synthesis:** Summarizes findings by explicitly linking statements directly to database rows and document citations without extrapolation.
- **FastAPI Backend & Pydantic Contracts:** Serves validated JSON outputs containing structured metrics, the generated SQL audit log, and source document IDs.

---

## Tech Stack

- **Framework:** FastAPI, Pydantic v2, Uvicorn
- **Data & Analytics:** SQLite3, Pandas, NumPy
- **Vector Search:** Scikit-learn (Cosine Similarity)
- **LLM & Embeddings:** Google GenAI SDK (`gemini-3.8-flash`, `gemini-embedding-001`)

---

## Sample Validated Output

```json
{
  "query": "What was Apex Cloud Systems' revenue and credit standing in Q3 2025, and what are their rules regarding capital expenditures?",
  "status": "success",
  "structured_metrics": [
    {
      "name": "Apex Cloud Systems",
      "revenue_usd": 14200000.0,
      "credit_score": 810,
      "risk_category": "Minimal Risk",
      "rating_agency": "S&P",
      "default_probability": 0.005
    }
  ],
  "retrieved_policy_id": "DOC-102",
  "evidence_trail": [
    {
      "source_type": "relational_db",
      "source_id": "sqlite_financial_data",
      "details": {
        "sql": "SELECT c.name, qf.revenue_usd, cr.credit_score, cr.risk_category, cr.rating_agency, cr.default_probability FROM companies c JOIN quarterly_financials qf ON c.company_id = qf.company_id LEFT JOIN credit_risk cr ON c.company_id = cr.company_id WHERE c.name = 'Apex Cloud Systems' AND qf.quarter = 'Q3' AND qf.fiscal_year = 2025;"
      }
    },
    {
      "source_type": "unstructured_policy",
      "source_id": "DOC-102",
      "details": {
        "title": "FY2025 Enterprise Cloud Expansion Policy"
      }
    }
  ],
  "executive_summary": "Apex Cloud Systems reported Q3 2025 revenue of $14,200,000 USD with an S&P credit score of 810 (Minimal Risk). All capital expenditures exceeding $5M require dual-signoff from the Risk Committee."
}
