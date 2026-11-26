"""The saving layer: config before training, never overwrite, full run folder at the end."""
import json

import numpy as np
import pandas as pd

from src import models as M
from src.evaluation import ModelSpec, evaluate
from src.saving import Saver


def test_run_folder_lifecycle(tmp_path, X21):
    saver = Saver(tmp_path, tmp_path / "outputs")
    run = saver.start_run("unit GMM", {"model": "GMM", "features": "log21", "n_train": 200})
    # config.json and env.json exist before anything is trained
    assert (run.path / "config.json").exists() and (run.path / "env.json").exists()
    assert json.loads((run.path / "config.json").read_text())["model"] == "GMM"
    assert not (run.path / "metrics.json").exists()

    with run.capture():
        print("hello log")
        res = evaluate(X21, ModelSpec("GMM", M.GMM, {"reg_covar": 1e-2}), "test", users=[0, 1], keep_models=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    run.add_figure(fig, "line", report=True)
    run.set_results(per_user=res.per_user, metrics=res.summary(), scores=res.scores, models=res.models,
                    best_config={"chosen": {"reg_covar": 1e-2}}, grid_table=pd.DataFrame({"reg_covar": [1e-2]}))
    out = saver.save_all()
    assert len(out) == 1

    files = {p.relative_to(run.path).as_posix() for p in run.path.rglob("*") if p.is_file()}
    for f in ("config.json", "env.json", "metrics.json", "per_user.csv", "scores.npz", "log.txt",
              "best_model/best_config.json", "best_model/grid_results.csv", "best_model/gmm_per_user.joblib",
              "best_model/params_per_user.npz", "visualizations/line.png"):
        assert f in files, f
    assert "hello log" in (run.path / "log.txt").read_text()
    assert (saver.report_dir / "line.png").exists()
    assert set(np.load(run.path / "scores.npz").files) >= {"genuine_0", "impostor_0"}
    idx = pd.read_csv(saver.index_path)
    assert len(idx) == 1 and abs(idx.mean_eer[0] - res.eers.mean()) < 1e-12

    # a second run with the same name gets NNN = 002 and a new folder; save_all skips saved runs
    run2 = saver.start_run("unit GMM", {"model": "GMM"})
    assert run2.path != run.path and run2.path.name.startswith("unit_GMM002_")
    saver.save_all()
    assert len(pd.read_csv(saver.index_path)) == 2
