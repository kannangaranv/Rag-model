import re
import io
from typing import Any, Dict, List, Optional,Union
import pandas as pd

APP_LABEL = ["BoardPAC", "BoardPAC Web", "BoardPAC  Device"]
INCLUDE_SHEETS = {"Web", "Device", "BA", "SA"}
ALLOW_TOKENS = {"allow", "allowed", "✓", "✔", "√", "yes", "y", "true", "1"}
DENY_TOKENS  = {"notallow", "not_allowed", "deny", "denied",
                "✗", "×", "x", "no", "n", "false", "0", "forbid", "forbidden",
                "notallowed"}

# clean a string value from a cell
def _clean_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("\u00A0", " ").replace("\u200B", "")
    s = s.strip()
    s = re.sub(r"[ \t]+", " ", s)
    return s

# check if a string is a permission token
def _is_permission_token(x: str) -> bool:
    if not x:
        return False
    v = re.sub(r"[\s_\-]+", "", x.lower())
    return (v in ALLOW_TOKENS) or (v in DENY_TOKENS) or v in {"allow", "notallow"}

# normalize a permission token to True/False/None
def _normalize_permission(value: str) -> Optional[bool]:
    if not value:
        return None
    v = re.sub(r"[\s_\-]+", "", value.lower())
    if v in ALLOW_TOKENS or v == "allow":
        return True
    if v in DENY_TOKENS or v in {"notallow"}:
        return False
    return None

# guess the header row index in a dataframe
def _guess_header_row(df: pd.DataFrame) -> int:
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
            continue  
        perms_in_row = sum(1 for v in tail if _is_permission_token(v))
        if perms_in_row > 0:
            base = -1000 * perms_in_row
        else:
            base = 50
        base += 30 if not _clean_str(row[0]) else 0
        contiguous = 0
        cur = 0
        for cell in tail:
            if cell:
                cur += 1
                contiguous = max(contiguous, cur)
            else:
                cur = 0
        base += contiguous * 2 + non_empty_tail * 3
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

# check if a row is a category row (first cell non-empty, rest empty)
def _is_category_row(row_values: List[str]) -> bool:
    if not row_values:
        return False
    first = _clean_str(row_values[0])
    rest = [_clean_str(v) for v in row_values[1:]]
    if not first:
        return False
    return all(v == "" for v in rest)

def _read_sheet_with_header(src: Union[str, bytes, io.BytesIO], sheet_name: str) -> pd.DataFrame:
    if isinstance(src, (bytes, bytearray)):
        bio = io.BytesIO(src)
        open_ctx = pd.ExcelFile(bio, engine="openpyxl")
    elif isinstance(src, io.BytesIO):
        open_ctx = pd.ExcelFile(src, engine="openpyxl")
    else:
        # assume filesystem path
        open_ctx = pd.ExcelFile(src, engine="openpyxl")

    with open_ctx as xls:
        raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        header_row = _guess_header_row(raw)
        df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row)

    cols = list(df.columns)
    if cols:
        first = str(cols[0])
        if not _clean_str(first) or first.lower().startswith("unnamed"):
            cols[0] = "Action"
            df.columns = cols
    return df
# convert a dataframe sheet to sentences with metadata
def sheet_to_sentences(df: pd.DataFrame, doc_id: str, sheet_name: str, app_label: str = APP_LABEL[0]) -> List[Dict[str, Any]]:
    if df.shape[1] < 2:
        return []
    
    cols = [(_clean_str(c) or f"Col{i}") for i, c in enumerate(df.columns)]
    df.columns = cols

    action_col = cols[0]
    role_cols = cols[1:]

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
                "doc_id": doc_id,
                "app": app_label,
                "sheet": sheet_name,
                "category": current_category,
                "action": action,
                "role": role,
                "permission": "ALLOW" if perm else "NOTALLOW",
            }
            out.append({"text": text, "metadata": meta})

    return out

def excel_to_vector_sentences(xlsx_path: str, doc_id: str) -> List[Dict[str, Any]]:
    all_out: List[Dict[str, Any]] = []
    with pd.ExcelFile(xlsx_path, engine="openpyxl") as xls:
        sheet_names = xls.sheet_names

    for sheet in sheet_names:
        if sheet.strip().lower() not in {s.lower() for s in INCLUDE_SHEETS}:
            continue
        try:
            df = _read_sheet_with_header(xlsx_path, sheet)
            all_out.extend(sheet_to_sentences(df, doc_id=doc_id, sheet_name=sheet))
        except Exception as e:
            print(f"[WARN] Skipping sheet '{sheet}' due to error: {e}")

    return all_out


