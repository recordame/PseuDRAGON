import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def call(script, *args):
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(script), *map(str, args)],
        cwd=script.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(script.name + " failed:\n" + result.stderr + result.stdout)
    return json.loads(result.stdout)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Replay finite synthetic execution checks and distinguish historical record checks."
    )
    parser.add_argument("--output", default="results/first")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = HERE / output
    if output.exists():
        raise SystemExit("Output directory already exists; choose a new --output path.")
    manifest = read(HERE / "runtime_manifest.json")["files"]
    for name, expected in manifest.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="work-", dir=output) as work:
        work = Path(work)
        for name in ["mask", "linked_records"]:
            shutil.copytree(HERE / name, work / name)
        mask = work / "mask"
        call(mask / "run_raw.py", "first.json")
        call(mask / "run_raw.py", "repeat.json")
        call(mask / "verification/independent_check.py")
        mask_report = read(mask / "verification/independent_report.json")
        assert mask_report["status"] == "pass"
        raw_contract = call(mask / "check_contract.py")
        raw = read(mask / "first.json")
        by_kind = {}
        for kind in ["raw", "compiled"]:
            groups = {}
            for row in raw["rows"]:
                if row["kind"] == kind:
                    groups.setdefault(row["candidate_id"], []).append(row)
            by_kind[kind] = {
                "candidates": len(groups),
                "three_primary_pass": sum(
                    all(r["position_pass"] for r in rows if r["case_index"] in [0, 1, 2])
                    for rows in groups.values()
                ),
                "all_seventeen_pass": sum(
                    all(
                        r["position_pass"] if r["position_pass"] is not None else r["missing_pass"]
                        for r in rows
                    )
                    for rows in groups.values()
                ),
            }
        inputs = read(mask / "inputs.json")
        compiled = inputs["compiled_recorded_rows"]
        compiled_contract = sum(
            row.get("program_executed") is True
            and row.get("status") == "pass"
            and bool(row.get("selected_contract_sha256"))
            and row.get("review_status") == "pending"
            and row.get("stage3_violations") == []
            for row in compiled
        )
        table8 = {
            "fresh_raw_output_execution": by_kind["raw"],
            "fresh_raw_contract_validation": {
                "accepted": raw_contract["accepted"],
                "candidates": raw_contract["candidates"],
            },
            "compiled_archived_output_rescoring": by_kind["compiled"],
            "compiled_archived_contract_records": {
                "accepted_and_executed_records": compiled_contract,
                "candidates": len(compiled),
            },
            "compiled_fresh_execution": False,
            "compiled_fresh_contract_validation": False,
            "fixture_definition": {
                "primary_case_indices": [0, 1, 2],
                "all_cases": 17,
                "digit_bearing": 12,
                "zero_digit_preservation": 2,
                "missing_preservation": 3,
            },
            "independent_checker_status": mask_report["status"],
            "first_repeat_byte_identical": mask_report["first_repeat_byte_identical"],
        }
        assert by_kind["raw"] == {
            "candidates": 33,
            "three_primary_pass": 32,
            "all_seventeen_pass": 0,
        }
        assert by_kind["compiled"] == {
            "candidates": 66,
            "three_primary_pass": 66,
            "all_seventeen_pass": 66,
        }
        assert raw_contract["accepted"] == 0 and compiled_contract == 66
        write(output / "table8.json", table8)
        write(output / "raw_contract.json", raw_contract)
        for name, source in [
            ("mask_rows.json", mask / "first.json"),
            ("mask_independent.json", mask / "verification/independent_report.json"),
        ]:
            shutil.copyfile(source, output / name)
        linked = work / "linked_records"
        call(linked / "run_check.py", "first")
        call(linked / "run_check.py", "repeat")
        linked_report = call(linked / "verification/independent_check.py")
        assert linked_report["status"] == "pass"
        write(output / "linked_independent.json", linked_report)
        shutil.copyfile(linked / "first.json", output / "linked_rows.json")
    cells = call(HERE / "cells/verify_execution.py", "--bundle-root", HERE / "cells")
    assert cells["status"] == "PASS"
    write(output / "cells.json", cells)
    report = {
        "status": "pass",
        "runtime": {"python": sys.version.split()[0]},
        "table8": table8,
        "synthetic_cells": cells["fresh_function_replay"],
        "historical_standalone_records": cells["historical_standalone_records"],
        "linked_records": linked_report,
        "scope": "Fresh raw MASK, raw contract, retained synthetic dataframe functions, and linked-record functions; archived compiled MASK outputs and historical standalone runs are record checks.",
    }
    write(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
