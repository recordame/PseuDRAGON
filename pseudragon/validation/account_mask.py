"""Compile reviewer-selected masking rules independently of field names."""

import ast


MASKING_MODES = frozenset(
    {"keep_last", "keep_first", "mask_last", "mask_first", "keep_edges", "mask_all"}
)


def _mask_count(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 64:
        raise ValueError(f"{label} must be an integer from {minimum} to 64.")
    return value


def masking_candidate(mode, count=4, prefix_count=1, suffix_count=4, mask_char="*"):
    """Build a pending MASK candidate with a verifiable, fixed-form program.

    Counts refer to characters, including separators. Keeping zero characters
    masks the entire value; masking zero characters is rejected. Missing values
    remain missing, and short values are never fully exposed by a keep rule.
    """
    if not isinstance(mode, str) or mode not in MASKING_MODES:
        raise ValueError("Choose a supported masking rule.")
    if (
        not isinstance(mask_char, str)
        or len(mask_char) != 1
        or not mask_char.isprintable()
        or mask_char.isspace()
    ):
        raise ValueError("The mask character must be one visible, printable character.")

    rule = {"mode": mode, "mask_char": mask_char}
    char = repr(mask_char)
    full_mask = f"{char} * len(str(x))"
    expression = full_mask
    label = "Mask every character"
    if mode in {"keep_last", "keep_first", "mask_last", "mask_first"}:
        count = _mask_count(count, "Character count", minimum=1 if mode.startswith("mask_") else 0)
        rule["count"] = count
        if mode == "keep_last" and count:
            expression = (
                f"{char} * (len(str(x)) - {count}) + str(x)[-{count}:] "
                f"if len(str(x)) > {count} else {full_mask}"
            )
            label = f"Show only the last {count} characters; fully mask shorter values"
        elif mode == "keep_first" and count:
            expression = (
                f"str(x)[:{count}] + {char} * (len(str(x)) - {count}) "
                f"if len(str(x)) > {count} else {full_mask}"
            )
            label = f"Show only the first {count} characters; fully mask shorter values"
        elif mode == "mask_last":
            expression = f"str(x)[:-{count}] + {char} * min({count}, len(str(x)))"
            label = f"Mask the last {count} characters, or the whole value if shorter"
        elif mode == "mask_first":
            expression = f"{char} * min({count}, len(str(x))) + str(x)[{count}:]"
            label = f"Mask the first {count} characters, or the whole value if shorter"
    elif mode == "keep_edges":
        prefix_count = _mask_count(prefix_count, "Leading character count")
        suffix_count = _mask_count(suffix_count, "Trailing character count")
        rule.update(prefix_count=prefix_count, suffix_count=suffix_count)
        total = prefix_count + suffix_count
        if total:
            prefix = f"str(x)[:{prefix_count}]" if prefix_count else "''"
            suffix = f"str(x)[-{suffix_count}:]" if suffix_count else "''"
            expression = (
                f"{prefix} + {char} * (len(str(x)) - {total}) + {suffix} "
                f"if len(str(x)) > {total} else {full_mask}"
            )
            label = (
                f"Show the first {prefix_count} and last {suffix_count} characters; "
                "fully mask values that leave no middle characters to hide"
            )

    parameters = {"strategy": mode, **{key: value for key, value in rule.items() if key != "mode"}}
    parameters.update(unit="character", preserve_missing=True)
    if mode.startswith("keep_"):
        parameters["short_value"] = "fully_masked"
    code = f"lambda x: ({expression}) if pd.notna(x) else x"
    return {
        "method": "MASK",
        "masking_rule": rule,
        "parameters": parameters,
        "code_snippet": code,
        "example_implementation": code,
        "description": label + ". Rule selected by the reviewer; pending final approval.",
        "legal_evidence": "Reviewer-selected handling rule; legal applicability requires review.",
        "applicability": "High",
    }


def validate_masking_candidate(candidate):
    """Verify compiled metadata and the complete AST without executing input code."""
    if "masking_rule" not in candidate:
        return False
    rule = candidate["masking_rule"]
    if not isinstance(rule, dict):
        raise ValueError("The masking rule must be a configuration object.")
    try:
        expected = masking_candidate(**rule)
    except TypeError as exc:
        raise ValueError("The masking rule contains missing or unsupported options.") from exc
    if rule != expected["masking_rule"]:
        raise ValueError("The masking rule contains options unrelated to the selected method.")
    if candidate.get("method", candidate.get("action")) != "MASK" or (
        "action" in candidate and candidate["action"] != "MASK"
    ):
        raise ValueError("The compiled masking rule must use the MASK method.")
    if candidate.get("parameters") != expected["parameters"]:
        raise ValueError("The MASK parameters do not match the selected masking rule.")
    if not isinstance(candidate.get("code_snippet"), str):
        raise ValueError("The masking program must be source text.")
    try:
        actual_ast = ast.dump(
            ast.parse(candidate.get("code_snippet", "")), include_attributes=False
        )
        expected_ast = ast.dump(ast.parse(expected["code_snippet"]), include_attributes=False)
    except (SyntaxError, TypeError, ValueError) as exc:
        raise ValueError("The masking program must contain valid source code.") from exc
    if actual_ast != expected_ast:
        raise ValueError(
            "The MASK program does not match the selected masking rule. Apply the rule again."
        )
    return True


def masking_preview(candidate, samples):
    """Return sample outputs from a validated, internally regenerated template."""
    if not validate_masking_candidate(candidate):
        raise ValueError("Select a compiled masking rule before previewing it.")
    import pandas as pd

    trusted = masking_candidate(**candidate["masking_rule"])
    transform = eval(
        trusted["code_snippet"],
        {"__builtins__": {"str": str, "len": len, "min": min}, "pd": pd},
    )
    return [transform(value) for value in samples]


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
