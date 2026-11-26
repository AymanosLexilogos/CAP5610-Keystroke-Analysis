# CLAUDE.md

- Repo: CAP5610-Keystroke-Analysis
- Use the keystroke-gmm conda env (`conda activate keystroke-gmm`); never pip-install globally.
- All saving goes through the notebook's saving cell into outputs/<run>/best_model and visualizations/.
- Notebooks: gmm_benchmark.ipynb (GMM vs published benchmark), experiments.ipynb (full suite), baseline.ipynb.
  Saving logic lives in src/saving.py and is set up only in each notebook's saving cell; write config.json before training.
