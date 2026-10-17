# Financial Decision Intelligence Assistant

A hybrid decision-support analytics engine combining **Text-to-SQL**, **Cosine-Similarity Semantic Document Retrieval (RAG)**, and **Strict Pydantic Validation** to evaluate corporate financial performance and debt covenant compliant

## 1. System Architecture

The pipeline processes user queries across two parallel retrieval channels before synthesizing a validated answer:

1. **Structured Retrieval (Text-to-SQL):**
   - User question is translated into SQLite syntax via Gemini using database schema context.
   - Query executes on `financial_data.db` under driver-level read-only protection (`mode=ro`, `PRAGMA query_only = ON`).
   - Computes deterministic metrics (revenue, margins, credit score) directly in the database.

2. **Unstructured Retrieval (Semantic RAG):**
   - Question is embedded via `gemini-embedding-001` and matched against policy memos using cosine similarity.
   - **Relevance Gate (Threshold = 0.60):** If cosine similarity is below `0.60`, the system marks the policy as `NOT_RETRIEVED` rather than forcing an irrelevant document.

3. **Synthesis & API Layer:**
   - Gemini synthesizes an executive summary citing both numerical database metrics and the specific Document ID.
   - Output is validated and served via FastAPI using strict Pydantic v2 schemas with an auditable evidence trail.

## 2. Core Technical Capabilities

### A. Database Guardrails & Execution Security
- **Driver-Level Immutability:** Uses SQLite URI parameters (`file:financial_data.db?mode=ro`) combined with `PRAGMA query_only = ON` to physically reject destructive operations (`DROP`, `DELETE`, `INSERT`, `UPDATE`) at the database engine level.
- **Relational Integrity:** Queries across multi-table schema (`companies`, `quarterly_financials`, `credit_risk`) joined via foreign keys on `company_id`.

### B. Semantic Vector Retrieval with Abstention Policy
- **Dense Embeddings:** Embeds unstructured policy memos (debt covenants, capex review guidelines) via the Google GenAI SDK.
- **Relevance Gate (Threshold = 0.60):** Unlike baseline RAG systems that force an arbitrary top match, queries falling below a cosine similarity score of `0.60` trigger a controlled fallback (`NOT_RETRIEVED`), preventing cross-domain hallucinations on unsupported queries (e.g., HR or employee leave policies).

### C. Typed Data Contracts (Pydantic v2)
- **Field Retention:** Extends `StructuredMetric` schemas across core relational fields (`ticker`, `quarter`, `fiscal_year`, `operating_expense_usd`, `revenue_usd`, `credit_score`, `risk_category`) to ensure database columns are preserved in API payloads.
- **Auditable Evidence Trail:** Emits structured metadata containing executed SQL queries and matched document IDs alongside the executive briefing.

---

## 3. Database Schema Overview

The relational SQLite database (`financial_data.db`) models synthetic multi-entity performance metrics:

1. **`companies`**: `company_id` (PK), `ticker` (UNIQUE), `name`, `sector`
2. **`quarterly_financials`**: `financial_id` (PK), `company_id` (FK), `quarter`, `fiscal_year`, `revenue_usd`, `net_income_usd`, `operating_expense_usd`
3. **`credit_risk`**: `risk_id` (PK), `company_id` (FK), `rating_agency`, `credit_score`, `risk_category`, `default_probability`


## 4. API Specification

### Endpoint: `POST /query`

#### Request Payload

{
  "question": "What was Apex Cloud Systems' revenue and credit standing in Q3 2025, and what are their rules regarding capital expenditures?"
}

#### Validated Response Payload

{
  "query": "What was Apex Cloud Systems' revenue and credit standing in Q3 2025, and what are their rules regarding capital expenditures?",
  "status": "success",
  "structured_metrics": [
    {
      "name": "Apex Cloud Systems",
      "ticker": null,
      "quarter": "Q3",
      "fiscal_year": 2025,
      "revenue_usd": 14200000.0,
      "net_income_usd": 3100000.0,
      "operating_expense_usd": null,
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
        "sql": "SELECT c.name, q.quarter, q.fiscal_year, q.revenue_usd, q.net_income_usd, r.credit_score, r.risk_category, r.rating_agency, r.default_probability FROM companies c JOIN quarterly_financials q ON c.company_id = q.company_id JOIN credit_risk r ON c.company_id = r.company_id WHERE c.name = 'Apex Cloud Systems' AND q.quarter = 'Q3' AND q.fiscal_year = 2025;"
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
  "executive_summary": "Apex Cloud Systems reported a Q3 2025 revenue of $14,200,000 USD and net income of $3,100,000 USD, maintaining an S&P credit score of 810 (Minimal Risk). Under policy DOC-102, all capital expenditures exceeding $5M require dual-signoff from the Risk Committee."
}

## 5. Verification & Testing Suite

The project includes systematic manual and integration checks located under Section D of the notebook:

* **Test 1 (Schema Field Retention):** Verifies that `ticker`, `quarter`, and `operating_expense_usd` exist inside `StructuredMetric.model_fields`.
* **Test 2 (Relevance Abstention):** Asserts that relevant capex questions retrieve `DOC-102` while out-of-domain questions (e.g., employee vacation policies) correctly yield `None`.
* **Test 3 (End-to-End Pipeline):** Executes hybrid query flows through the FastAPI endpoint handler, checking typed JSON construction.
* **Test 4 (Database Exception Handling):** Confirms that invalid SQL queries against non-existent tables throw structured HTTP 400 exceptions.
* **Test 5 (Read-Only Enforcement):** Confirms that write queries (`DELETE FROM companies`) trigger SQLite read-only driver errors.

---

## 6. Installation & Local Setup

### 1. Clone Repository

git clone [https://github.com/sparky047/financial-decision-intelligence-assistant.git](https://github.com/sparky047/financial-decision-intelligence-assistant.git)
cd financial-decision-intelligence-assistant


### 2. Install Dependencies

pip install fastapi uvicorn pydantic pandas scikit-learn numpy google-genai

### 3. Configure Credentials
export GEMINI_API_KEY="your-api-key-here"
### On Windows Command Prompt:
set GEMINI_API_KEY="your-api-key-here"
### 4. Run the API Server
uvicorn main:app --reload
Interactive Swagger API documentation will be available at `http://127.0.0.1:8000/docs`.

## 7. Known Architectural Trade-Offs & Roadmap

* **Vector Storage:** Currently operates over in-memory document vectors for targeted corporate policies; future iterations can integrate persistent stores (e.g., ChromaDB, pgvector).
* **Tool Routing:** Uses concurrent execution of relational retrieval and vector matching; an intent classification router can be introduced to conditionally invoke single modalities.
* **SQL Parsing:** While driver-level read-only flags prevent modifications, production scale-out would benefit from an AST query parser (e.g., `sqlglot`) to enforce query execution timeouts and internal metadata restrictions.


