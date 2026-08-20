"""DeepSurv (pycox CoxPH): small MLP, dropout, weight decay, early stopping on a held-out 20% split."""
from __future__ import annotations

import os

import numpy as np
import torch
import torchtuples as tt
from pycox.models import CoxPH
from sklearn.model_selection import train_test_split

from core.models import SurvivalModel, step_survival
from core.models.neural import early_stopping

torch.set_num_threads(1)


class DeepSurvModel(SurvivalModel):
    name = "deepsurv"
    label = "DeepSurv"
    complexity = 3
    param_grid = [{"width": w, "layers": n, "dropout": d, "lr": lr}
                  for w in (8, 16, 32) for n in (1, 2) for d in (0.1, 0.3) for lr in (1e-3, 1e-2)]

    def __init__(self, width: int = 16, layers: int = 1, dropout: float = 0.1, lr: float = 1e-3,
                 weight_decay: float = 1e-4, epochs: int = 300, patience: int = 15, batch_size: int = 32,
                 seed: int = 0):
        self.width, self.layers, self.dropout, self.lr = width, layers, dropout, lr
        self.weight_decay, self.epochs, self.patience, self.batch_size, self.seed = (
            weight_decay, epochs, patience, batch_size, seed)

    def _build(self, n_features: int) -> CoxPH:
        torch.manual_seed(self.seed)
        net = tt.practical.MLPVanilla(n_features, [self.width] * self.layers, 1, batch_norm=False,
                                      dropout=self.dropout, output_bias=False)
        return CoxPH(net, tt.optim.Adam(lr=self.lr, weight_decay=self.weight_decay))

    def fit(self, X, time, event):
        x = np.asarray(X, "float32")
        t, e = np.asarray(time, "float32"), np.asarray(event, "float32")
        np.random.seed(self.seed)
        tr, va = train_test_split(np.arange(len(x)), test_size=0.2, stratify=e, random_state=self.seed)
        self.n_features_ = x.shape[1]
        self.model_ = self._build(self.n_features_)
        es = early_stopping(self.patience)          # checkpoint in a temp file, not the working directory
        try:
            self.model_.fit(x[tr], (t[tr], e[tr]), batch_size=self.batch_size, epochs=self.epochs, callbacks=[es],
                            val_data=(x[va], (t[va], e[va])), verbose=False)
        finally:
            if os.path.exists(es.file_path):
                os.remove(es.file_path)
        self.model_.compute_baseline_hazards(input=x, target=(t, e))
        return self

    def predict_risk(self, X) -> np.ndarray:
        self.model_.net.eval()
        return self.model_.predict(np.asarray(X, "float32")).ravel().astype(float)

    def predict_survival(self, X, times) -> np.ndarray:
        self.model_.net.eval()
        surv = self.model_.predict_surv_df(np.asarray(X, "float32"))
        return step_survival(surv.index.to_numpy(float), surv.to_numpy().T, np.asarray(times, float))

    # Pickle the weights + baseline hazards, not the torch optimiser.
    def __getstate__(self):
        state = {k: v for k, v in self.__dict__.items() if k != "model_"}
        if "model_" in self.__dict__:
            state["_weights"] = {k: v.cpu() for k, v in self.model_.net.state_dict().items()}
            state["_baseline"] = (self.model_.baseline_hazards_, self.model_.baseline_cumulative_hazards_)
        return state

    def __setstate__(self, state):
        weights, baseline = state.pop("_weights", None), state.pop("_baseline", None)
        self.__dict__.update(state)
        if weights is not None:
            self.model_ = self._build(self.n_features_)
            self.model_.net.load_state_dict(weights)
            self.model_.baseline_hazards_, self.model_.baseline_cumulative_hazards_ = baseline
