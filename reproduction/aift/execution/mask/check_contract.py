import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "contract"))
from pseudragon.validation.method_contract import MethodContractError, normalize_method_metadata


def main():
    inputs = json.loads((HERE / "inputs.json").read_text())
    rows = []
    for candidate in inputs["raw_candidates"]:
        assert hashlib.sha256(candidate["code"].encode()).hexdigest() == candidate["code_sha256"]
        method = {
            "method": "MASK",
            "parameters": candidate["parameters"],
            "code_snippet": candidate["code"],
        }
        try:
            selected = normalize_method_metadata(method, candidate["column"])
            row = {
                "candidate_id": candidate["candidate_id"],
                "accepted": True,
                "verification": selected.get("contract_verification", "unverified"),
            }
        except MethodContractError as exc:
            row = {"candidate_id": candidate["candidate_id"], "accepted": False, "reason": str(exc)}
        rows.append(row)
    report = {
        "scope": "Fresh original metadata-normalizer execution on unchanged raw candidates",
        "candidates": len(rows),
        "accepted": sum(r["accepted"] for r in rows),
        "rejected": sum(not r["accepted"] for r in rows),
        "rows": rows,
        "compiled_contract_fresh_revalidation": False,
        "compiled_contract_limitation": "Exact original validation requires private source text; compiled contracts are retained historical evidence.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
