"""Explicit reviewer-selected account masks, independent of field names."""


def account_mask_candidate(mode):
    expressions = {
        "keep_last_four": "'*' * (len(str(x)) - 4) + str(x)[-4:] if len(str(x)) > 4 else '*' * len(str(x))",
        "mask_last_four": "str(x)[:-4] + '*' * min(4, len(str(x)))",
        "mask_all": "'*' * len(str(x))",
    }
    labels = {
        "keep_last_four": "Show only the last four characters; fully mask shorter values",
        "mask_last_four": "Mask the last four characters, or the whole value if shorter",
        "mask_all": "Mask every character",
    }
    if mode not in expressions:
        raise ValueError("Choose a supported account masking rule.")
    code = f"lambda x: ({expressions[mode]}) if pd.notna(x) else x"
    return {
        "method": "MASK",
        "parameters": {
            "strategy": mode,
            "mask_char": "*",
            "unit": "character",
            "preserve_missing": True,
        },
        "code_snippet": code,
        "example_implementation": code,
        "description": labels[mode]
        + ". Shared account rule selected by the reviewer; pending final approval.",
        "legal_evidence": "Reviewer-selected handling rule; legal applicability requires review.",
        "applicability": "High",
    }
