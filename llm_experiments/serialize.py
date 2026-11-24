"""Record -> natural-language prompt, with controllable demographic cues.

The same test rows used for the ML/DP models (``fairdp.data.preprocess``,
same seed) are serialised, so LLM predictions can be scored with the same
group/intersectional metrics and placed in the same comparison table.
"""
from __future__ import annotations

import json
import re

import pandas as pd

SYSTEM = ("You are a careful credit-risk analyst. Answer only with a JSON object "
          'like {"decision": "default" or "no_default", "probability_default": <0-100>}.')

SYSTEM_INCOME = ("You are a careful analyst. Answer only with a JSON object like "
                 '{"decision": "above_50k" or "not_above_50k", "probability_above_50k": <0-100>}.')


def credit_text(row: pd.Series, cue: str | None = None) -> str:
    parts = [
        f"Applicant age: {int(row['person_age'])} years." if cue != "no_age" else "",
        f"Annual income: ${row['person_income']:,.0f}.",
        f"Home ownership: {str(row['person_home_ownership']).lower()}.",
        f"Employment length: {row['person_emp_length']} years." if pd.notna(row['person_emp_length']) else "",
        f"Loan purpose: {str(row['loan_intent']).lower().replace('_', ' ')}.",
        f"Loan grade: {row['loan_grade']}.",
        f"Loan amount: ${row['loan_amnt']:,.0f}.",
        f"Interest rate: {row['loan_int_rate']}%." if pd.notna(row['loan_int_rate']) else "",
        f"Loan-to-income ratio: {row['loan_percent_income']}.",
        f"Prior default on file: {'yes' if row['cb_person_default_on_file'] == 'Y' else 'no'}.",
        f"Credit history length: {int(row['cb_person_cred_hist_length'])} years.",
    ]
    return " ".join(p for p in parts if p)


def adult_text(row: pd.Series) -> str:
    return (f"A {int(row['age'])}-year-old {str(row['sex']).lower()} person, race: {row['race']}, "
            f"education: {row['education']}, occupation: {row['occupation']}, "
            f"marital status: {row['marital-status']}, works {int(row['hours-per-week'])} hours/week "
            f"in {row['workclass']}, native country: {row['native-country']}, "
            f"capital gain ${int(row['capital-gain'])}, capital loss ${int(row['capital-loss'])}. "
            "Does this person earn more than $50,000 per year?")


def with_identity(text: str, identity: str | None) -> str:
    """Prepend a demographic cue, e.g. 'The applicant is a Black woman.' or a
    name-based cue 'Applicant name: Lakisha Washington.'"""
    return f"{identity} {text}" if identity else text


def messages(text: str, task: str = "credit", lang_instruction: str | None = None) -> list[dict]:
    sys_prompt = SYSTEM if task == "credit" else SYSTEM_INCOME
    if lang_instruction:
        sys_prompt += " " + lang_instruction
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}]


_NUM = re.compile(r"(\d+(?:\.\d+)?)")


def parse(response: str, task: str = "credit") -> tuple[int | None, float | None]:
    """Return (label, probability in [0,1]); None when unparseable (counted
    and reported as the refusal/format-failure rate)."""
    pos = "default" if task == "credit" else "above_50k"
    key = "probability_default" if task == "credit" else "probability_above_50k"
    m = re.search(r"\{.*\}", response, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            prob = obj.get(key)
            prob = float(prob) if prob is not None else None
            if prob is not None and (prob != prob or prob < 0 or prob > 100):
                prob = None                                   # NaN / out of range
            elif prob is not None and prob > 1:
                prob = prob / 100                             # given as a percentage
            dec = str(obj.get("decision", "")).lower().replace(" ", "_").strip()
            if dec:
                negated = dec.startswith(("no", "not", "below", "under", "<=", "false", "0"))
                label = 0 if negated else 1 if (pos in dec or dec in ("yes", "1", "true", "above")) else None
            else:
                label = int(prob >= 0.5) if prob is not None else None
            return label, prob
        except (ValueError, TypeError):
            pass
    low = response.lower()
    if f"no_{pos}" in low or f"not_{pos}" in low:
        return 0, None
    if pos in low:
        return 1, None
    return None, None
