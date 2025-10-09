import re
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


# ----------------------------
# Config
# ----------------------------
OUTPUT_JSONL = "role_matrix_sentences.jsonl"
APP_LABEL = ["BoardPAC", "BoardPAC Web", "BoardPAC  Device"]  # shown in the generated sentences
INCLUDE_SHEETS = {"Web", "Device", "BA", "SA"}

# OPTIONAL: if a sheet needs a forced header row index, specify it here (0-based)
# e.g., {"SA": 0, "BA": 0}
FORCED_HEADER_ROWS: Dict[str, int] = {
    # "SA": 0,
    # "BA": 0,
    # "ALL": 0,
}

# Tokens that mean allow / deny
ALLOW_TOKENS = {"allow", "allowed", "✓", "✔", "√", "yes", "y", "true", "1"}
DENY_TOKENS  = {"notallow", "not_allowed", "deny", "denied",
                "✗", "×", "x", "no", "n", "false", "0", "forbid", "forbidden",
                "notallowed"}


# ----------------------------
# Helpers
# ----------------------------
def _clean_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    # normalize weird spaces
    s = s.replace("\u00A0", " ").replace("\u200B", "")
    s = s.strip()
    s = re.sub(r"[ \t]+", " ", s)
    return s


def _is_permission_token(x: str) -> bool:
    if not x:
        return False
    v = re.sub(r"[\s_\-]+", "", x.lower())
    return (v in ALLOW_TOKENS) or (v in DENY_TOKENS) or v in {"allow", "notallow"}


def _normalize_permission(value: str) -> Optional[bool]:
    if not value:
        return None
    v = re.sub(r"[\s_\-]+", "", value.lower())
    if v in ALLOW_TOKENS or v == "allow":
        return True
    if v in DENY_TOKENS or v in {"notallow"}:
        return False
    return None


def _guess_header_row(df: pd.DataFrame) -> int:
    """
    Robust header row detector for your workbook:
      - Require >=3 non-empty tail cells (role columns).
      - Strongly penalize rows whose tail already contains permission tokens.
      - Bonus if first cell is empty (your header row has blank first col).
      - Reward if the *next few rows* contain permission tokens (real data rows).
    """
    cleaned = df.copy()
    for c in cleaned.columns:
        cleaned[c] = cleaned[c].apply(_clean_str)

    best_idx = 0
    best_score = -10**9
    max_scan = min(len(df), 30)

    for i in range(max_scan):
        row = cleaned.iloc[i].tolist()
        tail = row[1:]
        non_empty_tail = sum(1 for x in tail if x)
        if non_empty_tail < 3:
            continue  # needs enough role labels

        # Disqualify/penalize rows that already contain permission tokens
        perms_in_row = sum(1 for v in tail if _is_permission_token(v))
        if perms_in_row > 0:
            base = -1000 * perms_in_row
        else:
            base = 50

        # Bonus: your header row usually has empty first col
        base += 30 if not _clean_str(row[0]) else 0

        # Reward contiguous filled tail cells
        contiguous = 0
        cur = 0
        for cell in tail:
            if cell:
                cur += 1
                contiguous = max(contiguous, cur)
            else:
                cur = 0
        base += contiguous * 2 + non_empty_tail * 3

        # Lookahead: next 1..5 rows—count permission tokens
        lookahead = 0
        for j in range(i + 1, min(i + 6, len(cleaned))):
            tail2 = cleaned.iloc[j].tolist()[1:]
            perm_count = sum(1 for v in tail2 if _is_permission_token(v))
            lookahead += perm_count * 5

        score = base + lookahead
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def _is_category_row(row_values: List[str]) -> bool:
    """
    Category row = first cell has text; all other cells empty.
    Captures 'Login Controllers', 'Settings', etc.
    """
    if not row_values:
        return False
    first = _clean_str(row_values[0])
    rest = [_clean_str(v) for v in row_values[1:]]
    if not first:
        return False
    return all(v == "" for v in rest)


def _read_sheet_with_header(book_path: str, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(book_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    if sheet_name in FORCED_HEADER_ROWS:
        header_row = FORCED_HEADER_ROWS[sheet_name]
    else:
        header_row = _guess_header_row(raw)
    df = pd.read_excel(book_path, sheet_name=sheet_name, header=header_row, engine="openpyxl")
    # Ensure first column has a usable name
    cols = list(df.columns)
    if not cols:
        return df
    if not _clean_str(cols[0]) or str(cols[0]).lower().startswith("unnamed"):
        cols[0] = "Action"
        df.columns = cols
    return df


def sheet_to_sentences(df: pd.DataFrame, sheet_name: str, app_label: str = APP_LABEL[0]) -> List[Dict[str, Any]]:
    if df.shape[1] < 2:
        return []

    # Clean column names
    cols = [(_clean_str(c) or f"Col{i}") for i, c in enumerate(df.columns)]
    df.columns = cols

    action_col = cols[0]
    role_cols = cols[1:]

    # Drop entirely empty role columns
    keep_roles = []
    for c in role_cols:
        series = df[c].apply(_clean_str)
        if series.replace("", pd.NA).notna().any():
            keep_roles.append(c)
    role_cols = keep_roles

    out: List[Dict[str, Any]] = []
    current_category: Optional[str] = None

    for _, row in df.iterrows():
        row_strs = [_clean_str(row.get(c, "")) for c in df.columns]

        # Update category if needed
        if _is_category_row(row_strs):
            current_category = row_strs[0]
            continue

        action = _clean_str(row.get(action_col, ""))
        if not action:
            continue
        if action.lower() == "legend":
            continue
        for role in role_cols:
            raw_perm = _clean_str(row.get(role, ""))
            perm = _normalize_permission(raw_perm)
            if perm is None:
                continue

            can_cannot = "can" if perm else "cannot"
            if sheet_name == "Web":
                app_label = APP_LABEL[1]
            elif sheet_name == "Device":
                app_label = APP_LABEL[2]
            else:
                app_label = APP_LABEL[0]
            
            text = (
                f"For {app_label}, the role '{role}' {can_cannot} perform the action '{action}'."
                + (f" (Category: {current_category})." if current_category else "")
                + f" [Sheet: {sheet_name}]"
            )
            meta = {
                "app": app_label,
                "sheet": sheet_name,
                "category": current_category,
                "action": action,
                "role": role,
                "permission": "ALLOW" if perm else "NOTALLOW",
            }
            out.append({"text": text, "metadata": meta})

    return out


def excel_to_vector_sentences(xlsx_path: str) -> List[Dict[str, Any]]:
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    all_out: List[Dict[str, Any]] = []
    for sheet in xls.sheet_names:
        if sheet.strip().lower() not in {s.lower() for s in INCLUDE_SHEETS}:
            continue
        try:
            df = _read_sheet_with_header(xlsx_path, sheet)
            all_out.extend(sheet_to_sentences(df, sheet_name=sheet))
        except Exception as e:
            print(f"[WARN] Skipping sheet '{sheet}' due to error: {e}")
    return all_out


def write_jsonl(records: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python excel_to_vectors.py <excel_file.xlsx>")
        sys.exit(1)

    excel_path = sys.argv[1]
    assert Path(excel_path).exists(), f"File not found: {excel_path}"

    records = excel_to_vector_sentences(excel_path)
    print(f"Generated {len(records)} vector sentences.")
    write_jsonl(records, OUTPUT_JSONL)
    print(f"Wrote JSONL to: {OUTPUT_JSONL}")

    # Show a few examples
    for r in records[:10]:
        print(r["text"], "|", r["metadata"])


if __name__ == "__main__":
    main()
