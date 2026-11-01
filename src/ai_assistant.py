# src/ai_assistant.py
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


SCHEMA_CONTEXT = """
You are a Credit Risk SQL & Analytics Assistant. You have access to a single DuckDB table named `portfolio` (14,700 loan accounts, 1 row per account_id) joined from 5 source tables:
- loan_accounts.csv: account_id, customer_id, loan_type, loan_purpose, loan_amount, interest_rate_pct, tenure_months, origination_date, disbursal_channel, collateral_flag, target_default (0 or 1)
- customers.csv: age, age_band, gender, marital_status, education_level, employment_type, occupation_sector, years_employed, annual_income, income_band, region, city_tier, num_dependents, housing_type, years_at_residence
- bureau_credit.csv (aggregated): total_accounts, active_accounts, total_sanctioned, total_outstanding, bureau_max_dpd, count_overdue_loans, utilization_rate, max_line_utilization, has_bureau_history
- repayment_history.csv (aggregated): months_observed, repay_max_dpd, avg_dpd, latest_dpd, count_late, count_missed, repayment_ratio, total_shortfall, balance_reduction_pct, has_dpd_30plus, recent_3m_max_dpd, recent_3m_missed_count
- collections_events.csv (aggregated): coll_event_cnt, coll_total_recovered
- Model & Anomaly outputs: pd_logreg, pd_lgbm, risk_tier ('Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk'), loan_to_income_ratio, actual_dti_ratio, is_anomaly (0 or 1), anomaly_reason
"""


def _get_api_key() -> tuple[str | None, str | None]:
    """Detect Gemini or OpenAI API key from environment or Streamlit secrets."""
    for env_name, provider in [("GEMINI_API_KEY", "gemini"), ("OPENAI_API_KEY", "openai")]:
        val = os.environ.get(env_name)
        if val and val.strip() and not val.strip().lower().startswith("your_"):
            return val.strip(), provider
    try:
        import streamlit as st
        for sec_name, provider in [("GEMINI_API_KEY", "gemini"), ("OPENAI_API_KEY", "openai")]:
            val = str(st.secrets.get(sec_name, "")).strip()
            if val and not val.lower().startswith("your_"):
                return val, provider
    except Exception:
        pass
    return None, None


def _call_llm(prompt: str, api_key: str, provider: str, timeout: float = 30.0) -> str:
    """Call Gemini or OpenAI API with low temperature for deterministic SQL grounding."""
    if provider == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        # model="gemini-3.8-flash"
        model="gemini-3.5-flash"
        config = types.GenerateContentConfig(
            temperature=0.0,
        )
        resp = client.models.generate_content(model=model, contents=prompt, config=config)
        return resp.text
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=timeout)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content


def ask_portfolio(question: str, portfolio_df: pd.DataFrame, api_timeout: float = 30.0) -> dict:
    """Execute grounded Text-to-DuckDB Q&A over portfolio_df with zero hallucinated numbers."""
    api_key, provider = _get_api_key()
    if not api_key:
        return {
            "answer": "Declined: No API Key Set",
            "sql": None,
            "data": pd.DataFrame(),
            "source": "None",
            "mode": "Error: No API Key Set",
        }

    con = duckdb.connect(database=":memory:")
    con.register("portfolio", portfolio_df)

    sql_prompt = f"""{SCHEMA_CONTEXT}
User Question: "{question}"

Return ONLY a JSON object with keys:
- "answerable": true if the question can be answered using the columns in `portfolio`, false otherwise.
- "decline_reason": if answerable is false, explain what data is missing.
- "sql": a valid read-only DuckDB SELECT query over `portfolio` (limit to 20 rows max).
- "source_tables": string naming which of the 5 CSV tables hold the queried columns.
"""
    try:
        raw_json = _call_llm(sql_prompt, api_key, provider, timeout=api_timeout)
        clean_json = re.sub(r"```(?:json)?|```", "", raw_json).strip()
        parsed = json.loads(clean_json)
        answerable = bool(parsed.get("answerable", True))
        sql_query = parsed.get("sql", "").strip()
        source_tables = parsed.get("source_tables", "loan_accounts.csv, customers.csv")
        if not answerable:
            return {
                "answer": f"Declined (Out of Scope): {parsed.get('decline_reason', 'This question cannot be answered from the provided portfolio tables.')}",
                "sql": None,
                "data": pd.DataFrame(),
                "source": "Guardrail Declined",
                "mode": f"Live LLM ({provider.upper()})",
            }
    except Exception as e:
        return {
            "answer": f"Error generating SQL query: {str(e)}",
            "sql": None,
            "data": pd.DataFrame(),
            "source": "Error",
            "mode": f"Live LLM ({provider.upper()})",
        }

    # if not sql_query.upper().lstrip().startswith("SELECT"):
    #     return {
    #         "answer": "Declined: The generated query is not a valid SELECT statement.",
    #         "sql": None,
    #         "data": pd.DataFrame(),
    #         "source": "Guardrail Declined",
    #         "mode": f"Live LLM ({provider.upper()})",
    #     }

    try:
        result_df = con.execute(sql_query).df()
    except Exception as e:
         return {
            "answer": f"Error executing SQL query: {str(e)}",
            "sql": sql_query,
            "data": pd.DataFrame(),
            "source": "Error",
            "mode": f"Live LLM ({provider.upper()})",
        }

    synth_prompt = f"""You are a Credit Risk Analyst. Answer the user's question in 2-4 concise sentences using ONLY the exact numbers in the SQL result table below. Do NOT invent or extrapolate any numbers. Explicitly mention the source tables ({source_tables}).

Question: {question}
SQL Executed: {sql_query}
Exact Result Data:
{result_df.to_markdown(index=False)}
"""
    try:
        answer_text = _call_llm(synth_prompt, api_key, provider, timeout=api_timeout)
        mode_label = f"Live LLM ({provider.upper()})"
    except Exception as e:
        answer_text = f"Based on `{source_tables}`, here are the exact figures from the portfolio:\n\n" + result_df.to_string(index=False)
        mode_label = "Fallback to Dataframe String"

    return {
        "answer": answer_text,
        "sql": sql_query,
        "data": result_df,
        "source": source_tables,
        "mode": mode_label,
    }
