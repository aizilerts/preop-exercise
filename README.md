# Pre-Op Triage Take-Home

## Objective

Implement `triage_submission(...)` in `core.py` - it is a pre-op triage function for a single submission package. It is currently a naive LLM-based solution that makes a real model API call. The starter implementation intentionally does not follow some best practices in using the OpenAI API. You may use whatever file structure makes sense for your solution.

Your output must match this schema:

- `decision`: `READY | NEEDS_FOLLOW_UP | NOT_CLEARED`
- `issues[]`: `category` + evidence (how you design this is up to you)
- `explanation`

## What Is Provided

- `data/patients_sample_50.jsonl` includes:
  - `case_id`
  - `submission`
  - `label` (what a human labeled this case as)
- `run_baseline.py` runs your `triage_submission` implementation and writes outputs.
- `run_evals.py` scores outputs against provided `label` and can run determinism checks.

## Completion

Note: this exercise is evaluated on engineering judgment. You may not reach a 100% score, and that is OK! We are looking to understand how you approached the problem and designed a working solution.

## Setup

1. Confirm `uv` is installed.

```bash
uv --version
```

2. Set your OpenAI API key.

```bash
export OPENAI_API_KEY="<your_api_key>"
```

## Recommended Workflow

### Windows PowerShell
(These were added so I could run the tests with uv instead of make)

`make` is a separate build utility and is not included with Windows PowerShell by
default. You can run the equivalent project commands directly with `uv`:

1. Run the unit tests, which mock the API and test the policy rules locally:

```powershell
uv run --with 'openai>=2.0.0' --with 'pydantic>=2.8.0' --with 'pytest>=8.0.0' python -m pytest tests
```

2. Run baseline outputs (this sends each submission's document text to the API):

```powershell
uv run run_baseline.py --input data/patients_sample_50.jsonl --output data/baseline_outputs.jsonl --model gpt-4.1-mini
```

3. Run eval scoring:

```powershell
uv run run_evals.py --input data/patients_sample_50.jsonl --outputs data/baseline_outputs.jsonl --report data/eval_report.json
```

4. Run determinism check (this makes repeated API calls for one case):

```powershell
uv run run_evals.py --determinism --input data/patients_sample_50.jsonl --model gpt-4.1-mini --report data/determinism_report.json
```

5. Print score:

```powershell
uv run python -c "import json; r=json.load(open('data/eval_report.json', encoding='utf-8')); s=(r.get('primary_score') or {}).get('value_pct'); print(s if s is not None else r.get('local_metrics_summary', {}).get('aggregate_local_score_pct', 0.0))"
```

6. View the interactive report (TUI):

```powershell
uv run view_report.py --report data/eval_report.json
```

This opens a terminal UI (`view_report.py`) that shows per-case results side-by-side with oracle expectations. You can browse records, see metric pass/fail status, and inspect submission data. Press `f` on a metric row to filter the case list to failures. Press `q` to quit.

### macOS/Linux (or a shell with GNU Make)

The `make` shortcuts below use the same commands and default paths:

```bash
make test
make baseline
make evals
make determinism
make score
make report
```

## Tests

Run the unit tests with PowerShell:

```powershell
uv run --with 'openai>=2.0.0' --with 'pydantic>=2.8.0' --with 'pytest>=8.0.0' python -m pytest tests
```

Or, if GNU Make is available:

```bash
make test
```

## Outputs

- Baseline outputs: `data/baseline_outputs.jsonl`
- Eval report: `data/eval_report.json`
- Determinism report: `data/determinism_report.json`

## Configurable Variables

- `MODEL` (default `gpt-4.1-mini`)
- `INPUT` (default `data/patients_sample_50.jsonl`)
- `OUTPUT` (default `data/baseline_outputs.jsonl`)
- `REPORT` (default `data/eval_report.json`)
- `DETERMINISM_REPORT` (default `data/determinism_report.json`)
