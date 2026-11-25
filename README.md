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
pytest                          # reproduction, leak and likelihood checks (~15 s)
```

* `notebooks/baseline.ipynb`: reproduces the published detectors and doesn't write any files.
* `notebooks/experiments.ipynb`: all experiments (tuning, official and open-set tables,
  enrollment size, template adaptation, speed normalization, GMM diagnostics). On a
  laptop CPU the full run takes about an hour. Headless:
  `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/experiments.ipynb`

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
notebooks/baseline.ipynb     published reproduction
notebooks/experiments.ipynb  all experiments; ONE saving cell does all writing
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

## 8. Results

See [RESULTS.md](RESULTS.md) (generated from the last full run of `experiments.ipynb`).
