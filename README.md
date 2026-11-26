# CAP5610-Keystroke-Analysis

Generative models for keystroke-dynamics verification on the CMU benchmark
(Killourhy & Maxion, 2009): Gaussian variants (diagonal, full, Ledoit-Wolf, tied,
Bayesian/MAP, Student-t, GMM, Laplace) and a small RealNVP normalizing flow, compared
with the published distance detectors under the official protocol, plus open-set,
enrollment-size, template-adaptation and speed-normalization experiments.

## 1. Get the code

```bash
git clone https://github.com/AymanosLexilogos/CAP5610-Keystroke-Analysis.git
cd CAP5610-Keystroke-Analysis
```

## 2. Get the dataset (not committed)

Download `DSL-StrongPasswordData.csv` from the CMU keystroke benchmark page
(<https://www.cs.cmu.edu/~keystroke/>, direct link
<https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv>) and put it in `data/`:

```
data/DSL-StrongPasswordData.csv      MD5 470235f96568f28f9ea0da62234ec857
```

`datasets/` also works. Both folders are gitignored. The loader checks the MD5 and stops
with these instructions if the file is missing. To check the file yourself:
`certutil -hashfile data\DSL-StrongPasswordData.csv MD5` (Windows) or `md5sum` (Linux/macOS).

## 3. Environment (Miniconda)

Install Miniconda with the official installer:
<https://www.anaconda.com/docs/getting-started/miniconda/install>. Then, from the repo root:

```bash
conda env create -f environment.yml
conda activate keystroke-gmm
python -m ipykernel install --user --name keystroke-gmm --display-name "Python (keystroke-gmm)"
```

Everything (notebooks, scripts, tests) runs inside this env. Don't `pip install` globally.
If a package is missing, add it to `environment.yml` and run:

```bash
conda env update -f environment.yml --prune
```

To remove the env: `conda deactivate`, then `conda env remove -n keystroke-gmm`.

## 4. Run

```bash
conda activate keystroke-gmm
jupyter lab                     # open notebooks/, select the kernel "Python (keystroke-gmm)"
pytest                          # reproduction, leak, likelihood, flow and saving checks (~15 s)
```

Three notebooks, all with the same layout (setup → saving helpers → data → models → tuning →
training & test → figures → evaluation → save):

* `notebooks/gmm_benchmark.ipynb` (**main deliverable**, ~1 min): trains a GMM under the exact CMU
  benchmark protocol and places it in the published 14-detector table (31 raw features). It then compares
  it with the distance detectors and with a single Gaussian on equal footing (raw21, log21), using paired
  tests, and checks what the extra mixture components capture.
* `notebooks/experiments.ipynb` (**full suite**, ~1 h; the baseline for extensions): every model family
  (Gaussian variants, GMM, Laplace, Student-t, Bayes/MAP, RealNVP flow), tuning, the official and open-set
  tables, enrollment size, template adaptation, speed normalization and GMM diagnostics. For a few-minute
  smoke run, start Jupyter with `KS_QUICK=1` set (output goes to `outputs/_quick/`).
* `notebooks/baseline.ipynb`: reproduces the four published distance detectors and writes no files.

Headless execution:
`jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/<name>.ipynb`

### Google Colab alternative

Either open the notebook from GitHub (File → Open notebook → GitHub → this repo), or
put the repo in Google Drive and mount it (`from google.colab import drive; drive.mount('/content/drive')`,
then `%cd /content/drive/MyDrive/CAP5610-Keystroke-Analysis`). If you opened it from GitHub,
clone the repo in the first cell (`!git clone ... && %cd CAP5610-Keystroke-Analysis`). Upload
the CSV to `data/`. torch, numpy, pandas, scipy, scikit-learn and matplotlib are preinstalled,
so no environment setup is needed. Outputs written in a Colab session are lost when the session
ends unless the repo lives in Drive.

## 5. Protocol (summary)

* 51 users × 8 sessions × 50 typings of `.tie5Roanl`. One typing = one data point.
* Features: 21 = 11 hold (H) + 10 keydown-keydown (DD), log-transformed. UD = DD − H
  exactly, so the 31 published features have rank 21. They are used only to reproduce
  published results. Any standardization is fitted on training data only.
* Official protocol: train on typings 1-200 (sessions 1-4). Genuine test = 201-400.
  Impostor test = the first 5 typings of each of the other 50 users. EER is computed per
  user, then the mean, SD and bootstrap 95% CI over users are reported. Score: higher =
  more genuine. Models are compared with paired tests (bootstrap CI of the per-user
  difference + Wilcoxon) against **Scaled Manhattan | 21 log**.
* Tuning on validation only: fit on typings 1-160, validate on 161-200, impostors = other
  users' typings 6-10. Then refit on 1-200 and evaluate on test once.
* Population models (tied, Bayes/MAP, pretrained flow) never see impostor test typings
  and are also evaluated open-set: 3 folds of 17 users, with the held-out fold used only
  as impostors. `src/utils.check_no_leak` checks every fit.

## 6. Repo layout

```
environment.yml          conda env (keystroke-gmm, conda-forge)
src/utils.py             loading (MD5), feature sets, splits, leak check, EER, bootstrap, paired tests
src/models.py            distance detectors + Gaussian family (fit / score / sample)
src/flow.py              RealNVP flow (PyTorch, CPU), pretraining + fine-tuning
src/evaluation.py        protocol runner (official / validation / open-set), tuning
src/experiments.py       model registry + grids, enrollment, adaptation, GMM diagnostics
src/plots.py             figure builders (no saving)
scripts/flush_outputs.py delete old run folders (dry run unless --yes)
src/saving.py            run folders, config/env before training, save_model / save_figure / save_run
notebooks/gmm_benchmark.ipynb  GMM vs the published benchmark (main deliverable)
notebooks/experiments.ipynb    full experiment suite (baseline for extensions)
notebooks/baseline.ipynb       published reproduction
                         each notebook has ONE saving cell (section 2) and a final "save everything" cell
tests/                   pytest: reproduction, no-leak, sane log-likelihoods, flow
data/                    put the CSV here (gitignored)
outputs/                 run folders (gitignored except runs_index.csv and report_figures/)
```

## 7. Outputs

Each run writes a new folder `outputs/<NAME><NNN>_<timestamp>/`. Folders are never overwritten.

| file | content |
|---|---|
| `config.json` | written **before** training: model, hyperparameter grid, features, protocol, split, n_train, sampling, seed |
| `metrics.json` | mean EER, SD, 95% CI, test log-likelihood summary |
| `per_user.csv` | EER per user (used for paired tests) |
| `scores.npz` | raw genuine / impostor scores |
| `log.txt` | printed output and warnings |
| `env.json` | git commit, Python and library versions |
| `best_model/` | `best_config.json` (chosen hyperparameters + grid results), per-user parameters (`params_per_user.npz`), GMMs (`gmm_per_user.joblib`), flows (`flow_per_user.pt`, `flow_pretrained.pt`, including standardization statistics) |
| `visualizations/` | figures for the run |

Each run also appends one row to `outputs/runs_index.csv` (committed). The figures used in
the report are copied to `outputs/report_figures/` (committed). The dataset, checkpoints and
raw scores are never committed.

To clear old runs, use `scripts/flush_outputs.py`. It is a dry run unless you pass `--yes`, and it only
deletes run folders named `<NAME><NNN>_<timestamp>`:

```bash
python scripts/flush_outputs.py                       # list what a full flush would delete
python scripts/flush_outputs.py --yes                 # delete all runs, runs_index.csv and report figures
python scripts/flush_outputs.py --only bench_ --yes   # only runs whose name starts with bench_ (index rows pruned)
python scripts/flush_outputs.py --before 20261126-000000 --yes   # only runs created before that time
python scripts/flush_outputs.py --quick --yes         # only outputs/_quick (smoke runs)
```

## 8. Results

GMM vs the CMU benchmark (from `notebooks/gmm_benchmark.ipynb`, which also holds the full tables and conclusions):

| features | GMM | single Gaussian | Mahalanobis | Scaled Manhattan |
|---|---|---|---|---|
| raw31 (benchmark) | 10.98% (rank 5 of 16) | 11.06% | 11.01% | 9.62% |
| raw21 (same information) | 10.45% | 11.10% | 11.01% | 9.05% |
| log21 | 9.78% | 9.74% | 9.74% | 8.55% |

Mean EER over 51 subjects, official protocol. The GMM beats a single Gaussian only on raw21, where the
extra component models the slow tail of skewed timings. It does not beat Scaled Manhattan on any feature set.
Results of the full suite are in `notebooks/experiments.ipynb`, section 11.
