This package covers Table 8 and the finite execution checks in Section 4.4.
It uses synthetic input values and retained code/output records. It does not
require the application checkout, database, institutional policy corpus, network,
model service, or new participant decisions.

From this directory, create a Python environment and run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -I -B run.py --output results/first
.venv/bin/python -I -B run.py --output results/repeat
```

Python 3.11 or later is required. Dependencies are pinned in requirements.txt.
Run the second command set with a new output directory; existing results are
preserved. The script is independent of the current working directory. Relative
output paths resolve below this package. Temporary execution copies are created
inside the requested output directory and removed after the checks.

The report separates these checks:

| Evidence | What is recomputed |
| --- | --- |
| Raw MASK outputs | The 33 unchanged raw candidates, comprising 8 distinct snippets, run on the original 17 fixtures. A separate checker independently executes and scores them. The first 3 fixtures yield 32/33 passing candidates; all 17 yield 0/33. |
| Raw MASK contracts | The frozen metadata normalizer runs on each unchanged code/parameter pair: 0/33 accepted. |
| Compiled MASK outputs | The 66 recorded candidate outputs are rescored on all 17 fixtures: 66/66. No compiled program is regenerated or executed by this package. |
| Compiled MASK contracts | The package aggregates 66 historical accepted-and-executed records and their selected-contract hashes. It does not rerun the source-linked validator, which requires the excluded exact source text. |
| Synthetic cell execution | The 9 retained dataframe functions execute on 92,000 unique cells in the 9 original synthetic table exports, plus 1,288 boundary cells and 276 missing cells. An independent scalar reference checks every value. |
| Historical standalone runs | The 27 retained records total 276,000 comparisons including repeats. Fresh function outputs are serialized and compared with every recorded CSV hash. This is not 27 new database-backed standalone executions. |
| Linked records | The 2 selected functions execute on 84 synthetic transaction rows for 12 fictitious customers. All 144 matching row-pair identities and customer counts/sums are checked. A deliberate wrong key preserves the join count but introduces 3 missing and 3 incorrect pairs. A separate checker independently executes the functions and evaluates these relations. |

The MASK suite distinguishes 12 digit-bearing strings, 2 zero-digit preservation
cases, and 3 missing-value cases. The 3-input and 17-input results use different
input scopes; they are not a before/after change to the raw candidates.

`source_binding.json` records the original relative source paths, byte hashes,
and packaging transformations. Source text and full production-policy records
are excluded. `runtime_manifest.json` binds the packaged inputs and programs.
Only executable dataframe function ASTs are retained from the original standalone
programs; their executable structure was matched during packaging. The public
checker validates the supplied functions and records, without independently
authenticating inaccessible source artifacts.

`mask/verification/independent_check.py`,
`cells/verify_execution.py`, and
`linked_records/verification/independent_check.py` provide separate output
checking logic. The combined report is written to the requested output directory,
together with row-level MASK and linked-record results and component reports.

The distributed execution evidence is summarized in `evidence/summary.json` and
`portability_verification.json`. Full generated result directories are excluded;
the commands above regenerate them from the retained inputs and code.
