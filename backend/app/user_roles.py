from __future__ import annotations

from typing import Optional

# Canonical roles from role_matrix_sentences.jsonl.
ROLE_NAMES = [
    "Super Admin",
    "Actionee",
    "Admin Assistant- Access given to assigned subcategory data only",
    "Board Administrator",
    "Guest",
    "Invitee",
    "Member",
    "Member - Comment",
    "Member - View",
    "Secretary",
    "Secretary- Assistant",
    "Secretary- Confirm",
    "Secretary- Upload",
    "System Administrator",
]

# Backward-compatible aliases for input normalization.
ROLE_ALIASES = {
    "admin assistant": "Admin Assistant- Access given to assigned subcategory data only",
    "admin assistant- access given to assigned subcategory data only": "Admin Assistant- Access given to assigned subcategory data only",
    "system admin": "System Administrator",
}

# Legacy users might have old 1..6 levels; keep a stable interpretation.
LEGACY_LEVEL_TO_ROLE = {
    1: "System Administrator",
    2: "Board Administrator",
    3: "Secretary",
    4: "Member",
    5: "Invitee",
    6: "Guest",
}


def normalize_role_name(role: str) -> Optional[str]:
    raw = (role or "").replace("\r", " ").replace("\n", " ").strip()
    if not raw:
        return None
    folded = " ".join(raw.split()).lower()

    for name in ROLE_NAMES:
        if folded == name.lower():
            return name
    if folded in ROLE_ALIASES:
        return ROLE_ALIASES[folded]
    return None


def allowed_roles() -> list[str]:
    return list(ROLE_NAMES)


def role_to_level_code(role: str) -> int:
    canonical = normalize_role_name(role)
    if canonical is None:
        raise ValueError("Invalid role")
    # Store a deterministic integer code in existing Users.Level column.
    return ROLE_NAMES.index(canonical) + 1


def level_code_to_role(level: Optional[int]) -> str:
    if level is None:
        return "Member"
    try:
        n = int(level)
    except Exception:
        return "Member"

    if 1 <= n <= len(ROLE_NAMES):
        return ROLE_NAMES[n - 1]
    if n in LEGACY_LEVEL_TO_ROLE:
        return LEGACY_LEVEL_TO_ROLE[n]
    return "Member"
