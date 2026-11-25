# Shakespeare Harness terminal CLI

The CLI should let a researcher check their environment, see what an experiment will cost, run it, and interpret a saved result. The same commands should work interactively and in a shell script.

## Command design

Use Python's `argparse` with `python -m harness` as the entry point. Subcommands give each workflow its own help and inputs. A single training command would be smaller but would leave environment checks and report reading to separate scripts. An interactive terminal UI would add input and state management to a workflow that needs repeatable commands. Keep PyTorch as the training dependency.

| Command | Purpose | Output |
| --- | --- | --- |
| `harness doctor [--device DEVICE]` | Check Python, PyTorch, and the requested device. Default to CPU. | Python and PyTorch versions, CUDA availability, and whether the requested setup can run. |
| `harness plan CORPUS [experiment options]` | Validate the inputs and preview the experiment without training. | Corpus SHA-256 and split sizes, model widths, seed count, trial counts, steps, compute cap, estimated compute per strategy, and unused compute. |
| `harness run CORPUS [experiment options] [--output FILE] [--overwrite]` | Train the proxy and target models, then compare strategies. | Progress by seed, final paired summary, and a complete JSON report. |
| `harness report REPORT` | Inspect an existing JSON report without loading PyTorch. | Per-seed validation and held-out losses, winning recipes, estimated compute, and overall win, tie, and loss counts for screening. |

In the table, `harness` is shorthand for `python -m harness`. The root command and every subcommand should have `--help` with a runnable example.

```powershell
python -m harness doctor
python -m harness plan data/tinyshakespeare.txt --target-width 32 --seeds 7 19 41
python -m harness run data/tinyshakespeare.txt --target-width 32 --seeds 7 19 41 --output runs/width32.json
python -m harness report runs/width32.json
```

Example terminal output for `plan` with the published Tiny Shakespeare settings:

```text
Corpus        tinyshakespeare.txt
Device        CPU
Model widths  8 proxy / 16 target
Seeds         7, 19, 41
Trials        15 per seed (45 total)

Strategy     Steps per trial               Est. FLOPs / seed
Screened     126 proxy + 400 per target        1,698,416,640
Direct       400 per target                    1,699,430,400

Cap per seed       1,699,430,400 estimated FLOPs per strategy
Unused per seed    1,013,760 screened / 0 direct
```

## Experiment options

| Option | Default | Validation |
| --- | --- | --- |
| `CORPUS` | Required | A readable local file. Every split must hold a full context and its next byte. |
| `--proxy-width` | 8 | Positive integer smaller than the target width. |
| `--target-width` | 16 | Positive integer larger than the proxy width. |
| `--target-steps` | 400 | Positive integer that leaves at least one proxy training step per candidate under the cap. |
| `--seeds` | `7 19 41` | One or more distinct integers from 0 through 2⁶³ − 1. |
| `--device` | `cpu` | `cpu` or `cuda`. The run fails before training if the requested CUDA device is unavailable. |
| `--output` | `runs/results.json` | Run only. Must be a different file from the corpus. Replace an existing report only with `--overwrite`. |

Keep the existing 80/10/10 corpus split, eight recipes, and 15 trials per seed. Proxy screening trains eight proxy trials and promotes three recipes to the target model. Direct search trains four target trials. Both strategies share the cap of four full target trials. Charge screening to that budget, round down to whole proxy steps, and show the resulting difference in actual estimated compute. A plan previews these values from the same settings and calculation the run uses. It can produce a CUDA plan on a CPU-only machine.

## Runtime and reporting requirements

- **Compatibility.** Preserve `python run.py CORPUS ...` with its existing defaults, options, and saved report format. Keep the published Tiny Shakespeare JSON readable by the new report command.
- **Reproducibility.** Record the corpus hash and split sizes, all seeds, the model widths, training steps, device, and Python and PyTorch versions. Preserve the matched initialization and batch sampling within each seed and model size. Choose both winners by validation loss before scoring either on the test split.
- **Progress.** Print the trial count and estimated compute before training starts. Print each completed seed with the total seed count and held-out losses so a multi-seed run shows progress. Print the report path, mean paired delta, and win/tie/loss counts on success. Use plain text that remains readable when redirected.
- **Saved reports.** Keep every trial, its recipe, validation loss, wall-clock time, estimated FLOPs, and CUDA peak allocation when available. Write the final JSON only after the run completes and publish it atomically. A failed or interrupted run must preserve an existing report. A run without `--overwrite` must leave a destination created by another run alone.
- **Result interpretation.** Report loss in bits per byte and define the paired delta as screened minus direct. Compute win/tie/loss from the saved scores before rounding for display. Label FLOPs as estimates. Report wall-clock time separately.
- **Failure behavior.** Use exit status 0 for success, 2 for usage errors or invalid flags, 1 for environment, input or runtime failures, and 130 for an interrupted run. Send errors to stderr with the command, input, or setting to correct. A failed command must avoid a partial report at the final output path.
- **Dependencies.** Help, planning, and report reading must work before PyTorch is installed. `doctor` must explain an unavailable PyTorch installation. Training uses the repository's pinned PyTorch dependency. The CLI adds no dependencies.

## Acceptance checks

| Check | Expected result |
| --- | --- |
| Run root and subcommand help in a Python environment without PyTorch. | Help lists the commands, flags, defaults, and example usage. |
| Run `doctor --device cpu` with and without PyTorch. | The installed environment passes and the missing dependency gets an actionable error. |
| Plan, then run a one-seed CPU experiment with non-default widths. | The plan and report agree on widths, steps, seeds, corpus hash, and per-strategy compute. Both strategies complete and the run reports all 15 trials. |
| Ask for duplicate or out-of-range seeds, reversed widths, an undersized corpus, or too few steps. | Validation fails before the trainer starts. |
| Run on CPU with `--device cuda` and a CPU-only PyTorch build. | The command exits with a clear device error before training. Verify actual CUDA training on a CUDA-capable setup. |
| Run against an existing output, then retry with `--overwrite`. | The first run preserves the output and exits early. The second replaces it only after successful training. Check separately that an interrupted or failed replacement preserves the old result. |
| Read the published JSON, a fresh report, and malformed JSON in a Python environment without PyTorch. | The first two summarize correctly. The malformed report exits with a readable error. |
| Run the existing `run.py` integration test and the full suite. | The original command still produces its expected report and the CLI tests pass. |

## Implementation order

1. Reuse the existing budget and training code through a shared entry point. Verify the legacy CLI and the default benchmark before adding subcommands.
2. Add `doctor`, `plan`, and `run`. Verify dependency-free planning, input validation, progress, output protection, and plan-to-run agreement.
3. Add `report` and update the README. Verify both report versions, run the complete test suite, and run the documented commands against a real corpus.
