# Offline reproduction of the reuse and execution evaluations

This versioned package links Tables 4–8 to fixed inputs, frozen source snapshots, raw records, and executable checks. It does not call a model, contact a service, collect human approvals, or change a product installation. The 92 fields are reconstructed schema metadata from the same evaluation corpus, with saved labels and model predictions. They are not new institutions or independent datasets. Synthetic values are used for transformation execution.

## Run

Use Python 3.11 or 3.12 in an isolated environment. Install the two pinned dependencies:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -B verify_manifest.py
python -B run.py --output /tmp/pseudragon-reproduction --studies all --repeat --with-conservative-gate
```

The output directory must not exist. Choose a new name for another run. No network access is used after dependency installation. Full replay creates temporary learned stores and can require about 1 GB of working disk; the distributed package contains only the minimal inputs and evidence. `verification.json` is the top-level result. The independent checks are under `rq1-independent.json`, the generated growth/sensitivity studies, and `execution/`.

For a shorter selected run:

```sh
python -B run.py --output /tmp/pseudragon-table4 --studies family_name_exclusion --repeat
python -B run.py --output /tmp/pseudragon-table5 --studies equal_maintenance historical --repeat
python -B run.py --output /tmp/pseudragon-tables6-7 --studies growth sensitivity --repeat --with-conservative-gate
python -B execution/run.py --output /tmp/pseudragon-execution
```

## What each check establishes

| Paper evidence | Included check | Meaning of the output |
|---|---|---|
| Table 4: managed exact-name cache, managed cache plus rules; additional query-name exclusion | `rq1/studies/family_name_exclusion/` | The original frozen learner rebuilds empty stores. Family holdout precedes removal of every remaining record with the query name. Three fixed fresh-prediction sets give 75 exact routes versus 75 exact + 8 rule routes; name exclusion gives 0 cache routes versus 39 rule routes, including 8 positive and 31 negative fields over 12 names. All reused labels in these complete-store runs are correct. The separately labeled `initial_secondary` set is the original saved prediction set. |
| Table 5: equally maintained cache/history/rule methods and approval lifecycle | `rq1/studies/equal_maintenance/` | The original guarded manager/learner rebuilds rules for 1,104 replayed approval perturbations, 368 positions per prediction set. Each method is compared with its own clean or freshly rebuilt reference. Across methods, the full held-out prediction vectors, including fallbacks, agree at each stage; routes need not agree across methods. |
| Table 5: historical unguarded history-first and blocker ablations | `rq1/historical/` and `rq1/verify.py` | Independent arithmetic over preserved per-case route records. The damaged-stage counts are 208 cases/322 wrong reuse events for history-first, 2/4 after the guard, and 48/75 for cache with its blocker disabled. These historical arms are **not freshly relearned** by this package. Counts are per 368 perturbation positions and prediction set, not per 92 unique fields. |
| Table 6: partial-knowledge accumulation | `rq1/studies/growth/` | Fresh execution of the unchanged frozen learner for 20 orders, five family folds, five knowledge fractions, and three saved prediction sets. The separate verifier reconstructs routing, scores, transitions, and full-store endpoint comparisons without importing learner code. The mean combined reuse is 62.20 at 25% and 83 at 100%. At 25%, orders 7 and 19 each produce two false-negative reuses of the same `WD_ACNO` name. |
| Table 7: parameter sensitivity | `rq1/studies/sensitivity/` | Fresh learner execution at the default and eight fixed factor combinations, then independent reconstruction of routes and metrics. This retains the original candidate-score application and exact-to-suffix merge semantics; it does not impose a new uniform threshold on every rule path. |
| Optional conservative gate discussed with Table 6 | `rq1/conservative_gate/` | The preserved gate source reroutes the newly replayed growth stores, deferring only learned Non-PII routes for names absent from training to the same saved fallback. At 25%, mean reuse changes from 62.20 to 51.35 and added false negatives from 0.20 to 0; at full knowledge reuse changes from 83 to 77. This changes routing, not the learner. |
| Table 8: raw MASK candidates | `execution/mask/` | Actual execution of the original AST-accepted snippets and separate contract checks: 32/33 candidates pass all three primary inputs, 0/33 pass all 17 inputs, and 0/33 pass the raw parameter contract. Missing-value checks have a separate denominator from display-position checks. |
| Table 8: compiled MASK candidates | `execution/mask/` | Re-scoring of 66 saved compiled outputs on 17 inputs and checking 66 saved accepted/executed contract records. It does **not** rerun the compiler or the full source-linked contract validator, which require restricted institutional source text. |
| Synthetic execution and linked-record checks | `execution/cells/`, `execution/linked_records/` | Fresh execution of retained transformation functions on 92,000 synthetic cells, 1,288 boundary cells, and 276 missing-value cells; output hashing against the 27 recorded standalone-run outputs; and the linked-record/control checks. The historical 276,000 cell evaluations are records from three original runs, not 276,000 independent fields or a new three-run compiler benchmark. |

## Frozen version and provenance

The versioned learner has SHA256 `93d97d9cb7eb34c4e2ff0d84617a09c21560163e71ddb3e37e8bdb297b47fd10`. The guarded manager has SHA256 `47ef0f77f9e601786a29fd43324d2cdec61442cdbb553ecf423384cc93eb4935`; growth/sensitivity retain their original manager `2cb7856c88793fd15ca59a22a16653cc3826d373779e285bdcf5e96c134db11f`. These are experiment snapshots and do not replace current product code.

`rq1/source_binding.json` records original repository-relative paths and uncompressed source hashes. Every bound RQ1 Python snapshot is stored as deterministic gzip, including runners, helpers, learners, and frozen checkers. This protects the original bytes from repository formatters; compression changes storage only. The wrapper unpacks snapshots into the new runtime directories, and the separate arithmetic checker loads its frozen helper from the verified compressed bytes in memory. No snapshot is unpacked over the distributed package. In one input file, the unused `provenance_directory` field is converted from a private absolute path to a repository-relative path in 1,104 records; every other JSON field is unchanged, and both original and public-payload hashes are recorded. A separate wrapper reconstructs relative input manifests and portable runtime protocols because the original input manifests referenced the research checkout. The original protocol hash remains recorded. This is post-result packaging, not a new preregistration. The optional gate's directory bindings are supplied by the wrapper without changing its source bytes.

The public run checks generated raw-route/store-snapshot hashes against the original recorded hashes and compares every result field except the relocated protocol hash. Repeat execution compares all normalized route/metric outputs; only the original runners' timestamp normalization is used. Independent arithmetic/routing checks do not import the production learner. These are executable consistency checks, not independent new reference annotation.

## Serving environment

The deployment operator confirmed that institutional, source-ordered, and paired-timing inference used `gpt-oss:120b` in the same institutional network and NVIDIA H100 (80 GB) serving environment. Staff-facing aliases identify internally served model access. The timing result measures automated Stage 1 wall time; recorded timings and request counts were independently recomputed. `serving_environment.json` records this clarification. Frozen request identifiers and protocols retain their original bytes.

## Scope of the public material

The package excludes institutional policy bodies, customer records, author-specific filesystem paths, deployment logs, model request transcripts, and credentials. It does not reproduce practitioner measurements, initial/live model calls, retrieval-context histories, or the reported 82.95% automated live timing reduction. Those results belong to their separately described experiments. Compiled-output and historical-route checks are explicitly distinguished from fresh execution above.

The source-binding reports establish what was compared locally before packaging. The included hashes identify omitted restricted sources without making those sources runtime dependencies. See `coverage.json` for a machine-readable evidence map, `MANIFEST.json` for the distributed file hashes, and `verification_record.json` for the isolated execution record. Formatting conventions apply to new portable wrappers; the frozen research snapshots retain their original formatting so their byte hashes remain verifiable.
