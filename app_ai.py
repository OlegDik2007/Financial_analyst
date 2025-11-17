# app_ai.py
# ------------------------------------------------------------
# AI-assisted categorization for bank statements in Streamlit.
# - Merchant normalization (noise stripping + mapping)
# - Local rule store (learn-as-you-edit)
# - Caching to avoid repeated AI calls
# - Batch LLM classification (plug your provider)
# - Confidence-driven autofill
# ------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# ==========================
# 1) Category Taxonomy
# ==========================
DEFAULT_TAXONOMY: Dict[str, List[str]] = {
    "Income": ["Salary/W2", "Business Revenue", "Refund/Rebate", "Interest/Dividends", "Transfer-In"],
    "Housing": ["Rent/Mortgage", "Property Tax", "HOA", "Home Insurance", "Utilities"],
    "Transport": ["Fuel", "Ride-share/Taxi", "Public Transit", "Parking/Tolls", "Auto Insurance", "Maintenance"],
    "Groceries": ["Supermarket", "Warehouse Club"],
    "Dining": ["Restaurant", "Cafe/Fast Food"],
    "Health": ["Pharmacy", "Doctor/Dental", "Insurance Premium"],
    "Shopping": ["Online Retail", "Clothing", "Electronics"],
    "Entertainment": ["Streaming", "Events", "Subscriptions"],
    "Cash & Transfers": ["ATM Withdrawal", "Zelle/Venmo", "Transfer-Out"],
    "Fees & Charges": ["Bank Fee", "Late Fee", "FX Fee"],
    "Other": ["Uncategorized"],
}

# ==========================
# 2) Merchant Normalization
# ==========================
NOISE_PATTERNS = [
    r"\bPOS\s*\d+\b",
    r"\b#\d+\b",
    r"\bTERM\s*\d+\b",
    r"\bSTORE\s*\d+\b",
    r"\b[A-Z]{2}\s\d{2}/\d{2}\b",  # e.g., "IL 11/04"
    r"\d{4}-\d{4}-\d{4}",          # masked card
    r"\bID[:\s]*\w+\b",
]

COMMON_MERCHANT_MAP = {
    "AMZN": "Amazon",
    "AMAZON": "Amazon",
    "WAL-MART": "Walmart",
    "WALMART": "Walmart",
    "MCDONALD": "McDonalds",
    "MCDONALD'S": "McDonalds",
    "UBER *TRIP": "Uber",
    "UBER TRIP": "Uber",
    "LYFT RIDE": "Lyft",
    "COSTCO WHSE": "Costco",
    "COSTCO": "Costco",
    "STARBUCKS": "Starbucks",
    "TARGET": "Target",
    "WALGREENS": "Walgreens",
    "CVS": "CVS",
}


def normalize_merchant(text: str) -> str:
    """Normalize merchant names by stripping noise and mapping common variants."""
    t = (text or "").upper()
    for pat in NOISE_PATTERNS:
        t = re.sub(pat, "", t)
    t = re.sub(r"\s+", " ", t).strip()
    for k, v in COMMON_MERCHANT_MAP.items():
        if k in t:
            return v
    return t.title()


# ==========================
# 3) Rule Store (learn-as-you-edit)
# ==========================
@st.cache_resource
def get_rules_store() -> Dict[tuple, Dict[str, str]]:
    """
    Returns a mutable dict stored across reruns.
    Keys: (normalized_merchant, 'inflow'|'outflow')
    Values: {"category":..., "subcategory":..., "source":...}
    """
    return {}


def sign_bucket(amount: float) -> str:
    return "inflow" if float(amount) > 0 else "outflow"


def apply_rules(merchant: str, amount: float, rules_store: Dict) -> Optional[Dict[str, str]]:
    return rules_store.get((merchant, sign_bucket(amount)))


def learn_rule(
    merchant: str,
    amount: float,
    cat: str,
    sub: str,
    src: str,
    rules_store: Optional[Dict] = None,
) -> None:
    rules_store = rules_store or get_rules_store()
    rules_store[(merchant, sign_bucket(amount))] = {
        "category": cat,
        "subcategory": sub,
        "source": src,
    }


# ==========================
# 4) Caching AI Calls
# ==========================
@st.cache_resource
def _ai_cache() -> Dict[str, Dict[str, Any]]:
    """Simple in-memory cache for AI results."""
    return {}


def cache_get(key: str) -> Optional[Dict[str, Any]]:
    return _ai_cache().get(key)


def cache_set(key: str, value: Dict[str, Any]) -> None:
    _ai_cache()[key] = value


def make_cache_key(row: Dict[str, Any]) -> str:
    # Cache key buckets by (normalized_merchant, amount, year-month)
    base = f"{row.get('normalized_merchant','')}|{float(row.get('Amount',0)):.2f}|{str(row.get('Date'))[:7]}"
    return hashlib.sha1(base.encode()).hexdigest()


# ==========================
# 5) LLM Client Adapter
# ==========================
def call_llm_classify(
    batch_rows: List[Dict[str, Any]],
    taxonomy: Dict[str, List[str]] | None = None,
    provider: str = "openai",
) -> List[Dict[str, Any]]:
    """
    Classify transactions into (category, subcategory, source, confidence).
    Input items look like:
      {"description": str, "amount": float, "date": "YYYY-MM-DD"}

    Returns list of dicts (same order as input):
      {"category": str, "subcategory": str, "source": str, "confidence": float}

    By default this uses a lightweight heuristic "simulator".
    Swap the inner block with your LLM of choice for production.
    """
    taxonomy = taxonomy or DEFAULT_TAXONOMY

    # --------------- Real LLM (OpenAI) example ---------------
    # Uncomment to use. Make sure you set OPENAI_API_KEY in Streamlit secrets.
    #
    # try:
    #     from openai import OpenAI
    #     client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    #     system_prompt = (
    #         "You are a bank transaction categorizer.\n"
    #         "Return ONLY JSON with a list of objects (one per transaction) with keys: "
    #         "category, subcategory, source, confidence (0..1).\n"
    #         "Use this taxonomy strictly:\n" + json.dumps(taxonomy) + "\n"
    #         "Rules: inflows -> Income.* unless clearly refunds/transfers; "
    #         "groceries vs dining by merchant; ride-hail -> Transport.Ride-share/Taxi; "
    #         "subscriptions -> Entertainment.Subscriptions; "
    #         "unknown -> Other.Uncategorized with low confidence."
    #     )
    #     payload = {"transactions": batch_rows}
    #     resp = client.responses.create(
    #         model="gpt-4o-mini",
    #         input=[
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": json.dumps(payload)},
    #         ],
    #         response_format={"type": "json_object"},
    #     )
    #     content = resp.output[0].content[0].text  # type: ignore
    #     parsed = json.loads(content)
    #     # Expect parsed like {"results":[{...}, ...]} or just a list
    #     results = parsed.get("results", parsed)
    #     return results
    # except Exception as e:
    #     st.warning(f"LLM call failed, using heuristic fallback. Error: {e}")

    # --------------- Heuristic Fallback (Simulator) ---------------
    results: List[Dict[str, Any]] = []
    for r in batch_rows:
        desc = (r.get("description") or "").upper()
        amt = float(r.get("amount") or 0.0)

        # Simple income detection
        if amt > 0 and ("PAYROLL" in desc or "SALARY" in desc):
            results.append(
                {
                    "category": "Income",
                    "subcategory": "Salary/W2",
                    "source": "Payroll/Employer",
                    "confidence": 0.95,
                }
            )
            continue

        if amt > 0 and ("LLC" in desc or "LTD" in desc or "INC" in desc):
            results.append(
                {
                    "category": "Income",
                    "subcategory": "Business Revenue",
                    "source": "Business Payout",
                    "confidence": 0.9,
                }
            )
            continue

        # Common merchants
        if "AMZN" in desc or "AMAZON" in desc:
            results.append(
                {
                    "category": "Shopping",
                    "subcategory": "Online Retail",
                    "source": "Amazon",
                    "confidence": 0.85,
                }
            )
            continue

        if "UBER" in desc or "LYFT" in desc:
            results.append(
                {
                    "category": "Transport",
                    "subcategory": "Ride-share/Taxi",
                    "source": "Uber/Lyft",
                    "confidence": 0.9,
                }
            )
            continue

        if "COSTCO" in desc or "WALMART" in desc or "TARGET" in desc:
            results.append(
                {
                    "category": "Groceries",
                    "subcategory": "Supermarket",
                    "source": normalize_merchant(r.get("description", "")),
                    "confidence": 0.7,
                }
            )
            continue

        # Default: unknown/other
        results.append(
            {
                "category": "Other",
                "subcategory": "Uncategorized",
                "source": normalize_merchant(r.get("description", "")),
                "confidence": 0.4,
            }
        )

    return results


# ==========================
# 6) Main Entry: Categorize a DataFrame
# ==========================
def ai_categorize_df(
    df: pd.DataFrame,
    desc_col: str = "Description",
    amt_col: str = "Amount",
    date_col: str = "Date",
    taxonomy: Dict[str, List[str]] | None = None,
    min_conf_to_autofill: float = 0.75,
    batch_size: int = 25,
) -> pd.DataFrame:
    """
    Add AI-assisted categorization columns to a transaction DataFrame.

    New columns:
      - normalized_merchant
      - AI Category, AI Subcategory, AI Source, AI Confidence
      - Category, Subcategory, Source  (defaults to AI predictions, can be edited in UI)
    """
    taxonomy = taxonomy or DEFAULT_TAXONOMY

    df = df.copy()
    if desc_col not in df.columns or amt_col not in df.columns:
        raise ValueError(f"Missing required columns: '{desc_col}' and '{amt_col}' must exist.")

    if date_col not in df.columns:
        df[date_col] = ""

    # Normalize merchants once
    df["normalized_merchant"] = df[desc_col].astype(str).apply(normalize_merchant)

    rules_store = get_rules_store()
    predictions: Dict[int, Dict[str, Any]] = {}
    todo_batch: List[Dict[str, Any]] = []
    idx_batch: List[int] = []

    # Iterate rows
    for i, row in df.iterrows():
        try:
            amt = float(row[amt_col])
        except Exception:
            amt = 0.0

        nm = row["normalized_merchant"]

        # 1) Local rule hit?
        rule = apply_rules(nm, amt, rules_store)
        if rule:
            predictions[i] = {**rule, "confidence": 1.0}
            continue

        # 2) Cache hit?
        key = make_cache_key(
            {"normalized_merchant": nm, "Amount": amt, "Date": row.get(date_col)}
        )
        cached = cache_get(key)
        if isinstance(cached, dict):
            predictions[i] = cached
            continue

        # 3) Queue for AI
        todo_batch.append(
            {
                "description": str(row[desc_col]),
                "amount": float(amt),
                "date": str(row.get(date_col, ""))[:10],
            }
        )
        idx_batch.append(i)

        # Flush if batch is ready
        if len(todo_batch) >= batch_size:
            _flush_llm_batch(
                todo_batch,
                idx_batch,
                df,
                amt_col,
                date_col,
                taxonomy,
                min_conf_to_autofill,
                predictions,
                rules_store,
            )
            todo_batch, idx_batch = [], []

    # Flush remaining
    if todo_batch:
        _flush_llm_batch(
            todo_batch,
            idx_batch,
            df,
            amt_col,
            date_col,
            taxonomy,
            min_conf_to_autofill,
            predictions,
            rules_store,
        )

    # Attach predictions to DataFrame
    pred_df = pd.DataFrame.from_dict(predictions, orient="index").reindex(df.index)

    df["AI Category"] = pred_df.get("category", pd.Series(index=df.index))
    df["AI Subcategory"] = pred_df.get("subcategory", pd.Series(index=df.index))
    df["AI Source"] = pred_df.get("source", pd.Series(index=df.index))
    df["AI Confidence"] = (
        pred_df.get("confidence", pd.Series(index=df.index, dtype=float))
        .astype(float)
        .round(2)
    )

    # Final editable columns (default to AI values if not already present)
    if "Category" not in df.columns:
        df["Category"] = df["AI Category"]
    if "Subcategory" not in df.columns:
        df["Subcategory"] = df["AI Subcategory"]
    if "Source" not in df.columns:
        df["Source"] = df["AI Source"]

    return df


# ==========================
# 7) Helper: Flush LLM Batch
# ==========================
def _flush_llm_batch(
    todo_batch: List[Dict[str, Any]],
    idx_batch: List[int],
    df: pd.DataFrame,
    amt_col: str,
    date_col: str,
    taxonomy: Dict[str, List[str]],
    min_conf_to_autofill: float,
    predictions: Dict[int, Dict[str, Any]],
    rules_store: Dict,
) -> None:
    """Send a batch to the classifier, then cache + learn high-confidence rules."""
    results = call_llm_classify(todo_batch, taxonomy)

    for di, res in zip(idx_batch, results):
        res = {
            "category": res.get("category", "Other"),
            "subcategory": res.get("subcategory", "Uncategorized"),
            "source": res.get("source", df.at[di, "normalized_merchant"]),
            "confidence": float(res.get("confidence", 0.0)),
        }

        predictions[di] = res

        key2 = make_cache_key(
            {
                "normalized_merchant": df.at[di, "normalized_merchant"],
                "Amount": float(df.at[di, amt_col]),
                "Date": df.at[di, date_col],
            }
        )
        cache_set(key2, res)

        # Learn rule if confident and not just "Other"
        if res["confidence"] >= min_conf_to_autofill and res["category"] != "Other":
            learn_rule(
                df.at[di, "normalized_merchant"],
                float(df.at[di, amt_col]),
                res["category"],
                res["subcategory"],
                res["source"],
                rules_store,
            )


# ==========================
# Optional: tiny utility to add a correction (call from UI)
# ==========================
def record_user_correction(
    description: str,
    amount: float,
    chosen_category: str,
    chosen_subcategory: str,
    chosen_source: str = "Manual",
) -> None:
    """
    Call this from your UI after the user edits a row to persist a new rule.
    """
    merchant = normalize_merchant(description)
    learn_rule(
        merchant,
        float(amount),
        chosen_category,
        chosen_subcategory,
        chosen_source,
    )
