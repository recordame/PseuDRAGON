#!/usr/bin/env python3
"""Replay versioned offline studies in a new directory, then check their outputs."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def semantic(value):
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items() if k != "protocol_sha256"}
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def execute(command, cwd, log):
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        LLM_AUDIT_ENABLED="false",
    )
    with log.open("w") as stream:
        subprocess.run(
            [sys.executable, *command],
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def prepare_study(name, target, definition):
    origin = HERE / "rq1/studies" / name
    shutil.copytree(origin, target)
    compressed = list((target / "inputs").glob("*.gz"))
    compressed += list((target / "sources").glob("*.py.gz"))
    compressed += list(target.glob("*.py.gz"))
    for p in compressed:
        p.with_suffix("").write_bytes(gzip.decompress(p.read_bytes()))
        p.unlink()
    protocol = read(target / "runtime_settings.json")
    frozen = {
        str(p.relative_to(target)): sha(p)
        for directory in ("inputs", "sources")
        for p in (target / directory).glob("*")
        if p.is_file()
    }
    for name_ in (definition["runner"], "replay_helpers.py"):
        if (target / name_).exists():
            frozen[name_] = sha(target / name_)
    if name in ("family_name_exclusion", "equal_maintenance"):
        manifest = {rel: digest for rel, digest in frozen.items() if rel.startswith("inputs/")}
        write(target / "input_manifest.json", manifest)
        frozen["input_manifest.json"] = sha(target / "input_manifest.json")
        protocol["frozen_files"] = frozen
    else:
        protocol["files"] = frozen
        protocol["runner_sha256"] = sha(target / definition["runner"])
    write(target / "protocol.json", protocol)


def verify_binding():
    binding = read(HERE / "rq1/source_binding.json")
    for row in binding["files"]:
        data = (HERE / row["destination"]).read_bytes()
        if row["encoding"] == "gzip":
            data = gzip.decompress(data)
        assert hashlib.sha256(data).hexdigest() == row["included_payload_sha256"]
    return len(binding["files"])


def check_outputs(name, target, definition):
    expected = read(target / "expected/output_hashes.json")
    checked = {}
    for path, digest in expected.items():
        actual = sha(target / path)
        assert actual == digest, (name, path, "raw derived output differs from archived bytes")
        checked[path] = actual
    for path in definition["expected_results"]:
        original = json.loads(
            gzip.decompress((target / "expected" / (Path(path).name + ".gz")).read_bytes())
        )
        assert semantic(read(target / path)) == semantic(original), (name, path, "result mismatch")
    return {
        "raw_derived_output_hashes_match": checked,
        "all_result_fields_match_except_portable_protocol_hash": True,
    }


def run_study(name, output, definition, repeat):
    target = output / "runtime" / name
    prepare_study(name, target, definition)
    runner = definition["runner"]
    print("Running " + name, flush=True)
    if name == "family_name_exclusion":
        execute([runner, "run", "--run-id", "first"], target, output / (name + "-first.log"))
        if repeat:
            execute([runner, "run", "--run-id", "repeat"], target, output / (name + "-repeat.log"))
            execute([runner, "verify-repeat"], target, output / (name + "-compare.log"))
    elif name == "equal_maintenance":
        execute([runner, "run", "--run-name", "first"], target, output / (name + "-first.log"))
        if repeat:
            execute(
                [runner, "run", "--run-name", "repeat"], target, output / (name + "-repeat.log")
            )
            execute([runner, "compare"], target, output / (name + "-compare.log"))
    else:
        execute([runner, "run"], target, output / (name + "-first.log"))
        execute([definition["verifier"]], target, output / (name + "-independent.log"))
        if repeat:
            second = output / "repeat-runtime" / name
            prepare_study(name, second, definition)
            execute([runner, "run"], second, output / (name + "-repeat.log"))
            execute([definition["verifier"]], second, output / (name + "-repeat-independent.log"))
            for p in ("results.json", "snapshots.jsonl"):
                assert sha(target / p) == sha(second / p), (name, p, "repeat differs")
    report = check_outputs(name, target, definition)
    report.update({"fresh_frozen_learner_execution": True, "repeat_run": repeat})
    return report


def gate(output, repeat):
    """Override directory bindings, while executing the archived source unchanged."""
    source = output / "runtime/growth"
    target = output / "runtime/conservative_gate"
    shutil.copytree(HERE / "rq1/conservative_gate", target)
    for path in list(target.glob("*.py.gz")):
        path.with_suffix("").write_bytes(gzip.decompress(path.read_bytes()))
        path.unlink()
    protocol = json.loads(gzip.decompress((target / "protocol.json.gz").read_bytes()))
    protocol["source_files"] = {name: sha(source / name) for name in protocol["source_files"]}
    protocol["packaging"] = "Post-result path relocation; source code and intervention unchanged."
    protocol["source"] = str(source.relative_to(target.parents[2]))
    write(target / "protocol.json", protocol)
    spec = importlib.util.spec_from_file_location("portable_gate", target / "run_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE, module.SOURCE = target, source
    module.run("first")
    if repeat:
        module.run("repeat")
        execute(["verify_and_report.py"], target, output / "conservative-gate-independent.log")
    original = json.loads(gzip.decompress((target / "first-results.json.gz").read_bytes()))
    assert semantic(read(target / "first-results.json")) == semantic(original)
    assert (
        sha(target / "first-routes.json")
        == read(target / "expected_hashes.json")["first-routes.json"]
    )
    return {
        "source_unchanged": True,
        "routes_byte_equal_to_archived": True,
        "results_equal_except_portable_protocol_hash": True,
        "repeat_run": repeat,
        "mode": "Recorded-store rerouting on freshly regenerated growth snapshots",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New empty results directory")
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=[
            "all",
            "family_name_exclusion",
            "equal_maintenance",
            "growth",
            "sensitivity",
            "historical",
            "execution",
        ],
        default=["all"],
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Run fixed RQ1 studies twice and compare normalized outputs",
    )
    parser.add_argument("--with-conservative-gate", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    assert not output.exists(), "Use a new output directory to preserve prior runs."
    output.mkdir(parents=True)
    definitions = read(HERE / "rq1/studies.json")
    studies = (
        list(definitions) + ["historical", "execution"] if "all" in args.studies else args.studies
    )
    report = {
        "source_binding_files_verified": verify_binding(),
        "new_model_calls": 0,
        "new_human_decisions": 0,
        "python": sys.version.split()[0],
        "studies": {},
    }
    for name in studies:
        if name in definitions:
            report["studies"][name] = run_study(name, output, definitions[name], args.repeat)
    if args.with_conservative_gate:
        assert "growth" in studies, "The conservative gate requires freshly replayed growth."
        report["studies"]["conservative_gate"] = gate(output, args.repeat)
    if "historical" in studies or any(
        n in studies for n in ["family_name_exclusion", "equal_maintenance"]
    ):
        execute(
            [
                str(HERE / "rq1/verify.py"),
                "--output",
                str(output),
                *(["--historical"] if "historical" in studies else []),
            ],
            HERE,
            output / "rq1-independent.log",
        )
        report["independent_rq1"] = read(output / "rq1-independent.json")
    if "execution" in studies:
        execute(
            [str(HERE / "execution/run.py"), "--output", str(output / "execution")],
            HERE / "execution",
            output / "execution.log",
        )
        report["studies"]["execution"] = {
            "see": "execution",
            "mode": "Raw MASK execution, archival compiled-output checks, and synthetic function/linked-record execution",
        }
    report["status"] = "PASS"
    write(output / "verification.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "studies": list(report["studies"]),
                "source_binding_files_verified": report["source_binding_files_verified"],
            }
        )
    )


if __name__ == "__main__":
    main()
