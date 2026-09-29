"""Bind supported snippet forms to accurate method metadata without execution.

This is deliberately a small structural recognizer, not general Python semantic
inference. Unknown edited code needs an explicit user declaration and remains
marked unverified. Input records are never modified in place.
"""

import ast
import copy
import hashlib


METHODS = frozenset(
    {"KEEP", "DELETE", "HASH", "MASK", "TOKENIZE", "GENERALIZE", "ROUND", "ENCRYPT"}
)
INTERVAL_1000 = 'lambda x: f"{(int(x)//1000)*1000}~{(int(x)//1000)*1000+999}" if pd.notna(x) and str(x).isdigit() else str(x)'
INTERVAL_PARAMETERS = {
    "strategy": "integer_interval",
    "interval_width": 1000,
    "interval_origin": 0,
    "separator": "~",
    "non_numeric": "stringify",
}
MASK_LAST_FOUR_PARAMETERS = {"mask_char": "*", "mask_last": 4, "short_value_mask_length": 4}


class MethodContractError(ValueError):
    """An edited snippet cannot be approved with its current declaration."""


def _structure(code):
    try:
        return ast.dump(ast.parse(code), include_attributes=False)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise MethodContractError("Code snippet must be valid Python before approval.") from exc


def _declared_contract(declaration):
    if not isinstance(declaration, dict) or set(declaration) != {"method", "parameters"}:
        raise MethodContractError(
            "Provide both an explicit method and a parameters object for edited custom code."
        )
    method = declaration["method"]
    parameters = declaration["parameters"]
    if (
        not isinstance(method, str)
        or method.upper() not in METHODS
        or not isinstance(parameters, dict)
    ):
        raise MethodContractError(
            "Custom-code method must be supported and parameters must be a JSON object."
        )
    if method.upper() == "MASK" and not parameters:
        raise MethodContractError("A MASK declaration needs explicit nonempty parameters.")
    return method.upper(), copy.deepcopy(parameters)


def normalize_method_metadata(selected, column, code_modified=False, declared_contract=None):
    """Return a detached candidate with a truthful code/metadata contract.

    Supported full-AST forms: the 1,000-wide interval lambda and the
    generated suffix-four masking assignment for this exact column. Whitespace,
    comments and quote style are immaterial; extra statements, changed operands,
    predicates and other columns do not match. The code itself is never changed.

    Unknown unchanged snippets retain their prior metadata; they are not labeled
    verified. MASK candidates require nonempty parameters before selection.
    Unknown edited snippets require an explicit structured declaration.
    Such a declaration is recorded as user_declared_unverified, not as a proof.
    """
    if not isinstance(selected, dict) or not isinstance(column, str) or not column:
        raise MethodContractError("A method object and nonempty column name are required.")
    result = copy.deepcopy(selected)
    # Template identity verifies execution metadata, not legal appropriateness.
    from pseudragon.validation.display_requirement import validate_display_candidate

    try:
        compiled_display = validate_display_candidate(result)
    except ValueError as exc:
        raise MethodContractError(str(exc)) from exc
    if compiled_display:
        declaration = (
            _declared_contract(declared_contract) if declared_contract is not None else None
        )
        if declaration is not None and declaration != ("MASK", result["parameters"]):
            raise MethodContractError("The declaration disagrees with the compiled display rule.")
        result["contract_verification"] = "compiled_display_requirement"
        result["contract_code_sha256"] = hashlib.sha256(result["code_snippet"].encode()).hexdigest()
        return result
    code = result.get("code_snippet", "")
    if not isinstance(code, str):
        raise MethodContractError("Code snippet must be a string.")
    if not code.strip():
        if result.get("method", result.get("action", "")).upper() == "MASK":
            raise MethodContractError(
                "A MASK selection needs executable code and explicit parameters."
            )
        if code_modified:
            raise MethodContractError(
                "Edited code is empty; select KEEP or DELETE explicitly instead."
            )
        return result
    try:
        tree = _structure(code)
    except MethodContractError:
        if (
            code_modified
            or declared_contract is not None
            or result.get("method", result.get("action", "")).upper() == "MASK"
        ):
            raise
        return result
    recognized = None
    if tree == _structure(INTERVAL_1000):
        recognized = ("GENERALIZE", INTERVAL_PARAMETERS)
    if column.isidentifier():
        mask = f"masked_{column} = str(record[{column!r}])[:-4] + '****' if len(str(record[{column!r}])) > 4 else '****'"
        if tree == _structure(mask):
            recognized = ("MASK", MASK_LAST_FOUR_PARAMETERS)
    for visible in range(1, 65):
        keep_suffix = f"lambda x: '*' * (len(str(x)) - {visible}) + str(x)[-{visible}:] if pd.notna(x) and len(str(x)) > {visible} else str(x)"
        if tree == _structure(keep_suffix):
            recognized = (
                "MASK",
                {"mask_char": "*", "visible_suffix": visible, "short_value": "unchanged"},
            )
            break
    keep_first = "lambda x: str(x)[0] + '*' * (len(str(x)) - 1) if pd.notna(x) and len(str(x)) > 0 else str(x)"
    if tree == _structure(keep_first):
        recognized = ("MASK", {"mask_char": "*", "visible_prefix": 1, "empty_value": "unchanged"})
    from pseudragon.validation.account_mask import account_mask_candidate

    for mode in ("keep_last_four", "mask_last_four", "mask_all"):
        candidate = account_mask_candidate(mode)
        if tree == _structure(candidate["code_snippet"]):
            recognized = ("MASK", candidate["parameters"])
            break
    declaration = _declared_contract(declared_contract) if declared_contract is not None else None
    if recognized is not None:
        method, parameters = recognized
        if declaration is not None and declaration != recognized:
            raise MethodContractError(
                "The explicit method/parameters disagree with the supported code pattern."
            )
        verification = "supported_ast_pattern"
    elif code_modified or declaration is not None:
        if declaration is None:
            raise MethodContractError(
                "Edited custom code is outside the supported patterns. Confirm its method and parameters "
                "in Custom-code contract before approval; this declaration will remain unverified."
            )
        method, parameters = declaration
        verification = "user_declared_unverified"
    else:
        if result.get("method", result.get("action", "")).upper() == "MASK" and not result.get(
            "parameters"
        ):
            raise MethodContractError(
                "This MASK candidate has no declared parameters. Select a source-compiled candidate "
                "or provide an explicit custom-code contract before approval."
            )
        return result
    key = "method" if "method" in result else "action"
    previous = {
        "method": result.get(key),
        "parameters": copy.deepcopy(result.get("parameters", {})),
    }
    result[key] = method
    if "method" in result and "action" in result:
        result["action"] = method
    result["parameters"] = copy.deepcopy(parameters)
    result["contract_verification"] = verification
    result["contract_code_sha256"] = hashlib.sha256(code.encode()).hexdigest()
    if previous != {"method": method, "parameters": parameters}:
        result["previous_contract"] = previous
    return result


def prepare_method_selection(selected, column, selection):
    """Merge UI edits then validate, without mutating the candidate or payload."""
    result = copy.deepcopy(selected)
    if not isinstance(selection, dict):
        return normalize_method_metadata(result, column)
    code_changed = False
    if selection.get("code_modified", False):
        if "code_snippet" not in selection:
            raise MethodContractError("Missing edited code snippet.")
        code_changed = selection["code_snippet"] != result.get("code_snippet", "")
        result["code_snippet"] = selection["code_snippet"]
        result["example_implementation"] = selection["code_snippet"]
    if selection.get("description_modified", False):
        if not isinstance(selection.get("description"), str):
            raise MethodContractError("Edited description must be a string.")
        result["description"] = selection["description"]
    return normalize_method_metadata(
        result, column, code_changed, selection.get("declared_contract")
    )
