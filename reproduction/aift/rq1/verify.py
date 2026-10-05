"""Recompute metrics from routes; do not import experiment runners or learners."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_audit_namespace = {"__name__": "frozen_audit_common"}
_audit_path = HERE / "audit_common.py.gz"
exec(compile(gzip.decompress(_audit_path.read_bytes()), str(_audit_path), "exec"), _audit_namespace)
independent_route = _audit_namespace["independent_route"]
audit_feedback = _audit_namespace["audit_feedback"]
audit_rule_support = _audit_namespace["audit_rule_support"]
metric_rows = _audit_namespace["metric_rows"]
split_ids = _audit_namespace["split_ids"]


def read(p):
    return json.loads(Path(p).read_text())


def family(folder):
    records = read(folder / "inputs/records.json")
    by_id = {r["id"]: r for r in records}
    groups = read(folder / "inputs/groups.json")["groups"]
    all_rows = read(folder / "runs/first/routes.json")
    snapshots = read(folder / "runs/first/snapshots.json")
    initial = read(folder / "inputs/initial_fallbacks.json")
    n = 0
    for snapshot in snapshots:
        group, excluded = snapshot["group"], snapshot["excluded_name"]
        held, train_ids, removed = split_ids(records, groups, group, excluded)
        assert train_ids == set(snapshot["training_ids"])
        assert removed == set(snapshot["removed_same_name_ids_from_family_train"])
        train = [by_id[k] for k in snapshot["training_ids"]]
        rules = snapshot["rules"]["heuristics"]
        audit_rule_support(rules, train, excluded)
        prediction_set = snapshot["prediction_set"]
        if prediction_set.startswith("fresh_"):
            audit_feedback(snapshot["store"], train, int(prediction_set[-1]))
        for row in snapshot["routes"]:
            assert row["id"] in held
            assert set(row["support_ids"]) <= train_ids
            expected_fallback = (
                initial[row["id"]]
                if prediction_set == "initial_secondary"
                else by_id[row["id"]]["fallbacks"][prediction_set[-1]]
            )
            assert row["fallback"] == expected_fallback
            if excluded:
                assert row["column"].casefold() == excluded.casefold()
            arm = row["condition"]
            answer = independent_route(
                snapshot["store"],
                rules,
                row["column"],
                row["fallback"],
                cache=arm != "guarded_history_rules",
                allow_rules=arm != "name_cache",
            )
            kind = "cache" if row["route"] == "name_cache" else row["route"]
            assert (row["prediction"], kind, row["pattern"]) == (
                answer["prediction"],
                answer["kind"],
                answer["pattern"],
            )
            n += 1
    collected = defaultdict(list)
    for row in all_rows:
        collected[row["prediction_set"], row["mode"], row["condition"]].append(row)
    reported = read(folder / "runs/first/results.json")["sets"]
    table = []
    for (prediction_set, mode, arm), rows in sorted(collected.items()):
        assert len(rows) == len({r["id"] for r in rows}) == 92
        refs = by_id
        repetition = prediction_set[-1] if prediction_set.startswith("fresh_") else "1"
        if prediction_set == "initial_secondary":
            refs = {k: dict(r, fallbacks={"1": initial[k]}) for k, r in by_id.items()}
        metrics = metric_rows(rows, refs, repetition)
        original = reported[prediction_set][mode][arm]
        for name in [
            "TP",
            "FP",
            "FN",
            "TN",
            "n",
            "reused",
            "correct_reuse",
            "wrong_reuse",
            "errors",
            "F1",
        ]:
            assert metrics[name] == original["field_micro"][name]
        assert metrics["distinct_reused_names"] == original["name_macro"]["names_with_any_reuse"]
        for name in ["reused", "correct_reuse", "wrong_reuse", "errors"]:
            field = {
                "reused": "coverage",
                "correct_reuse": None,
                "wrong_reuse": "wrong_reuse_rate",
                "errors": "error_rate",
            }[name]
            if field:
                assert (
                    abs(metrics["name_equal_weighted_rates"][name] - original["name_macro"][field])
                    < 1e-12
                )
        if prediction_set.startswith("fresh_"):
            table.append(
                dict(
                    prediction_set=prediction_set,
                    mode=mode,
                    arm=arm,
                    exact=sum(r["route"] == "name_cache" for r in rows),
                    rule=sum(r["route"] == "rule" for r in rows),
                    fallback=metrics["fallback"],
                    reused=metrics["reused"],
                    wrong_reuse=metrics["wrong_reuse"],
                    positive_reuse=metrics["reused_positive"],
                    negative_reuse=metrics["reused_negative"],
                    reused_names=metrics["distinct_reused_names"],
                    F1=metrics["F1"],
                    per_name=metrics["per_name"],
                )
            )
    return {
        "status": "PASS",
        "freshly_reconstructed_routes": n,
        "stores": len(snapshots),
        "family_and_query_name_exclusion_checked": True,
        "table4": table,
    }


def maintenance(folder):
    records = read(folder / "inputs/records.json")
    labels = {r["id"]: r["label"] for r in records}
    baseline = read(folder / "runs/first/baseline_routes.json")
    tallies = defaultdict(Counter)
    clean = defaultdict(Counter)
    arms = ["exact_cache", "guarded_history_rules", "guarded_cache_rules"]
    for block in baseline:
        for arm, rows in block["routes"].items():
            clean[block["rep"], arm]["reused"] += sum(r["route"] != "fallback" for r in rows)
            clean[block["rep"], arm]["wrong_reuse"] += sum(
                r["route"] != "fallback" and r["prediction"] != labels[r["id"]] for r in rows
            )
    case_count = 0
    identities = Counter()
    for line in (folder / "runs/first/case_routes.jsonl").open():
        case = json.loads(line)
        for stage, choices in case["stages"].items():
            vectors = [{r["id"]: r["prediction"] for r in choices[arm]} for arm in arms]
            assert vectors[0] == vectors[1] == vectors[2]
            for arm, rows in choices.items():
                wrong = sum(
                    r["route"] != "fallback" and r["prediction"] != labels[r["id"]] for r in rows
                )
                tallies[case["rep"], stage, arm]["wrong_cases"] += wrong > 0
                tallies[case["rep"], stage, arm]["wrong_reuse_events"] += wrong
        for arm in arms:
            assert case["stages"]["withdrawn_rebuilt"][arm] == case["fresh_remaining"][arm]
            assert case["stages"]["corrected_rebuilt"][arm] == case["stages"]["clean"][arm]
            identities[arm] += 1
        case_count += 1
    assert case_count == 1104
    assert all(
        c["wrong_cases"] == 2 and c["wrong_reuse_events"] == 4
        for (rep, stage, arm), c in tallies.items()
        if stage == "wrong_approval"
    )
    assert all(
        c["wrong_reuse_events"] == 0
        for (rep, stage, arm), c in tallies.items()
        if stage in ("withdrawn_rebuilt", "corrected_rebuilt")
    )
    return {
        "status": "PASS",
        "cases": case_count,
        "all_heldout_label_vectors_including_fallback_equal_across_arms": True,
        "each_arm_withdrawal_and_restoration_route_identities": dict(identities),
        "clean": [dict(rep=k[0], arm=k[1], **v) for k, v in sorted(clean.items())],
        "per_prediction_set": [
            dict(rep=k[0], stage=k[1], arm=k[2], **v) for k, v in sorted(tallies.items())
        ],
    }


def historical():
    with gzip.open(HERE / "studies/family_name_exclusion/inputs/records.json.gz", "rt") as stream:
        labels = {r["id"]: r["label"] for r in json.load(stream)}
    table = []
    for name in ["history_guard", "cache_blocker"]:
        counters = defaultdict(Counter)
        cases = Counter()
        with gzip.open(HERE / "historical" / (name + "-case_routes.jsonl.gz"), "rt") as stream:
            for line in stream:
                block = json.loads(line)
                rep = int(block["case"].split(":", 1)[0][1:])
                cases[rep] += 1
                for stage in [
                    "damaged",
                    "feedback_only_stale",
                    "removed_and_rebuilt",
                    "full_corrected",
                ]:
                    for arm, rows in block[stage].items():
                        wrong = sum(
                            r["route"] != "fallback" and r["prediction"] != labels[r["id"]]
                            for r in rows
                        )
                        counters[rep, stage, arm]["wrong_cases"] += wrong > 0
                        counters[rep, stage, arm]["wrong_reuse_events"] += wrong
        assert dict(cases) == {1: 368, 2: 368, 3: 368}
        for (rep, stage, arm), values in sorted(counters.items()):
            table.append(dict(source=name, rep=rep, stage=stage, arm=arm, **values))
        if name == "history_guard":
            expected = {
                "current_history_rules": (208, 322),
                "guarded_history_rules": (2, 4),
                "canonical_lookup_archived_rules": (2, 4),
            }
        else:
            expected = {"blocker_on": (2, 4), "blocker_off_last": (48, 75)}
        for rep in [1, 2, 3]:
            for arm, pair in expected.items():
                c = counters[rep, "damaged", arm]
                assert (c["wrong_cases"], c["wrong_reuse_events"]) == pair
    return {
        "status": "PASS",
        "mode": "Archived raw-route arithmetic, not a fresh learner execution",
        "table5": table,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--historical", action="store_true")
    args = parser.parse_args()
    result = {"production_code_imports": False}
    if (args.output / "runtime/family_name_exclusion").exists():
        result["family_name_exclusion"] = family(args.output / "runtime/family_name_exclusion")
    if (args.output / "runtime/equal_maintenance").exists():
        result["equal_maintenance"] = maintenance(args.output / "runtime/equal_maintenance")
    if args.historical:
        result["historical"] = historical()
    result["status"] = "PASS"
    (args.output / "rq1-independent.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: {"status": value.get("status", "PASS")} if isinstance(value, dict) else value
                for key, value in result.items()
            }
        )
    )


if __name__ == "__main__":
    main()
