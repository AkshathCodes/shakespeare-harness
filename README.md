# Shakespeare Harness

**Can cheap experiments on a tiny transformer help pick better training settings for a larger transformer under the same compute budget?**

This command-line harness trains two causal byte-level transformers from scratch in PyTorch and compares proxy screening with direct random search. It runs on CPU and accepts CUDA as an option.

## Run the experiment

The reference run uses Python 3.12. From this directory in PowerShell, create a virtual environment, install PyTorch, download Tiny Shakespeare, and run the benchmark.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt
New-Item -ItemType Directory -Force data | Out-Null
curl.exe -fL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/tinyshakespeare.txt
.venv/Scripts/python.exe -m harness doctor
.venv/Scripts/python.exe -m harness plan data/tinyshakespeare.txt
.venv/Scripts/python.exe -m harness run data/tinyshakespeare.txt --output runs/tinyshakespeare.json
.venv/Scripts/python.exe -m harness report runs/tinyshakespeare.json
```

The default run uses 400 training steps per target trial and seeds 7, 19, and 41. The corpus SHA-256 for the reference run is `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. [char-rnn Tiny Shakespeare corpus](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)

Use `.venv/bin/python` for a virtual environment on Linux or macOS and `mkdir -p data` to create the dataset directory. On macOS, install with the default pip index. To try CUDA, install a compatible PyTorch build and pass `--device cuda`. [PyTorch installation guide](https://pytorch.org/get-started/locally/)

For a quick check using one seed, or to run the tests:

```powershell
.venv/Scripts/python.exe -m harness run data/tinyshakespeare.txt --target-steps 4 --seeds 7 --output runs/smoke.json
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The harness accepts any local text file. It writes the file fingerprint, model and budget settings, per-trial validation loss and runtime, the selected recipes, and paired held-out losses to JSON. CUDA runs also record peak PyTorch tensor allocations above the pre-trial baseline in MiB.

## Terminal commands

Run from the repository directory. Each command has its own help, for example `python -m harness run --help`.

| Command | Use it to |
| --- | --- |
| `python -m harness doctor` | Check that PyTorch and the requested device can start. |
| `python -m harness plan CORPUS` | Preview the corpus split, trial counts, and estimated compute per seed. |
| `python -m harness run CORPUS` | Run training and write the full report. |
| `python -m harness report REPORT` | Read saved results, including the published reference JSON. |

`plan` and `run` share `--proxy-width`, `--target-width`, `--target-steps`, `--seeds`, and `--device`. The new run command writes to `runs/results.json` by default. Use `--output FILE` to choose a destination and `--overwrite` to replace an existing report after successful training. Atomic publication without overwrite requires a filesystem that supports hard links, such as NTFS.

Help, `plan`, and `report` work without PyTorch. The original `python run.py CORPUS ...` entry point is still available. [CLI requirements](docs/cli-requirements.md)

## How the comparison works

Both models use one attention head, one transformer block, a 12-byte context, and a batch size of two. At the default widths, the proxy has width 8 and 4,960 parameters, and the target has width 16 and 11,456 parameters. The search space is four learning rates crossed with constant and cosine schedules.

| Strategy | Where the budget goes |
| --- | --- |
| Direct search | Train four randomly sampled recipes on the target model. |
| Proxy screening | Train all eight recipes on the cheaper proxy. Promote the three lowest validation losses and train those recipes from scratch on the target. |

Each strategy has a cap equal to four full target trials. Screening uses the compute allowance of the fourth target trial, rounded down to whole proxy steps. The FLOP estimate counts the forward and backward matrix multiplications in the causal attention, linear projections, and output layer. It excludes optimizer operations, nonlinearities, and evaluation. The report also records elapsed time for training and validation.

The data is split into contiguous 80% training, 10% validation, and 10% test segments. Within each seed, trials at a given size start from the same initialization and see the same training batches. An overlapping target recipe is trained independently in each strategy and charged to each budget. The winners are chosen by target validation loss before scoring them on the test segment. All losses are reported in bits per byte, where lower is better.

The JSON report also shows rank agreement for the four recipes sampled by direct search. Each pair of recipes counts as an agreement if the proxy and target rank them in the same order. This subset provides a view of transfer before filtering to only the promoted recipes.

## Tiny Shakespeare benchmark

CPU run with Python 3.12 and PyTorch 2.15.0. Each screened trial ran for 126 steps on the proxy and each target trial ran for 400 steps. Screening used 1,698,416,640 estimated FLOPs per seed and direct search used 1,699,430,400. Lower loss is better.

| Seed | Screened validation | Direct validation | Screened held-out | Direct held-out |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 3.8323 | 3.8323 | 3.6876 | 3.6876 |
| 19 | 3.8100 | 4.2259 | 3.5939 | 4.1026 |
| 41 | 3.7705 | 3.7705 | 3.5982 | 3.5982 |

Screening tied on two seeds and improved held-out loss by 0.5087 bits per byte on seed 19, for a mean difference of −0.1696 bits per byte. The rankings agreed on 17 of 18 paired comparisons in the direct-search subsets. These measurements cover one corpus, two small models, and three seeds. A wider study would be needed to estimate transfer at larger scales. Trial timings are descriptive because the screened strategy always runs first. [Full benchmark report](results/tinyshakespeare.json)

## Related work

- RE-Bench includes a task that uses inexpensive training experiments to predict better compute allocation at a larger budget. [METR RE-Bench](https://github.com/METR/RE-Bench/blob/main/README.md#predict-optimal-hyperparameters-through-scaling-law)
- Karpathy's `autoresearch` uses an agent to propose and test changes to small language-model training. [autoresearch](https://github.com/karpathy/autoresearch)
