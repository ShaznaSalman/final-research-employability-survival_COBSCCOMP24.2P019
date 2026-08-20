"""Established neural survival models from pycox (Kvamme et al.), with the shared SurvivalModel interface:

  - DeepHit  (Lee et al., 2018)      discrete-time; no proportional-hazards assumption
  - Cox-Time (Kvamme et al., 2019)   Cox model whose effects may change over time (non-proportional)
  - N-MTLR   (Fotso, 2018)           neural multi-task logistic regression, discrete-time

All: small MLP, dropout, weight decay, early stopping on a stratified 20% validation split. Risk score =
minus the area under the predicted survival curve to 24 months (shorter expected search = higher risk).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
import torchtuples as tt
from pycox.models import MTLR, CoxTime, DeepHitSingle
from pycox.models.cox_time import MLPVanillaCoxTime
from sklearn.model_selection import train_test_split

from core.models import CURVE_TIMES, SurvivalModel, step_survival

torch.set_num_threads(1)


def early_stopping(patience: int) -> tt.callbacks.EarlyStopping:
    """Early stopping whose weight checkpoint goes to a temp file, not the working directory."""
    fd, path = tempfile.mkstemp(suffix=".pt", prefix="ep_ckpt_")
    os.close(fd)
    return tt.callbacks.EarlyStopping(patience=patience, file_path=path)


class _NeuralBase(SurvivalModel):
    complexity = 3
    param_grid = [{"width": w, "dropout": d, "lr": lr} for w in (16, 32) for d in (0.1, 0.3) for lr in (1e-3, 1e-2)]

    def __init__(self, width: int = 16, layers: int = 1, dropout: float = 0.1, lr: float = 1e-3,
                 weight_decay: float = 1e-4, epochs: int = 300, patience: int = 15, batch_size: int = 32,
                 num_durations: int = 12, seed: int = 0):
        self.width, self.layers, self.dropout, self.lr = width, layers, dropout, lr
        self.weight_decay, self.epochs, self.patience, self.batch_size = weight_decay, epochs, patience, batch_size
        self.num_durations, self.seed = num_durations, seed

    def _split(self, x, t, e):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        return train_test_split(np.arange(len(x)), test_size=0.2, stratify=e, random_state=self.seed)

    def _train(self, x_tr, y_tr, x_va, y_va):
        es = early_stopping(self.patience)
        try:
            self.model_.fit(x_tr, y_tr, batch_size=self.batch_size, epochs=self.epochs, callbacks=[es],
                            val_data=(x_va, y_va), verbose=False)
        finally:
            if os.path.exists(es.file_path):
                os.remove(es.file_path)

    def _surv_df(self, X):
        self.model_.net.eval()
        return self.model_.predict_surv_df(np.asarray(X, "float32"))

    def predict_survival(self, X, times) -> np.ndarray:
        s = self._surv_df(X)
        return step_survival(s.index.to_numpy(float), s.to_numpy().T, np.asarray(times, float))

    def predict_risk(self, X) -> np.ndarray:
        surv = self.predict_survival(X, CURVE_TIMES)
        return -np.trapezoid(surv, CURVE_TIMES, axis=1)


class _DiscreteBase(_NeuralBase):
    model_cls = None
    loss_kwargs: dict = {}

    def fit(self, X, time, event):
        x = np.asarray(X, "float32")
        t, e = np.asarray(time, "float32"), np.asarray(event, "float32")
        tr, va = self._split(x, t, e)
        self.labtrans_ = self.model_cls.label_transform(self.num_durations)
        y_tr = self.labtrans_.fit_transform(t[tr], e[tr])
        y_va = self.labtrans_.transform(t[va], e[va])
        net = tt.practical.MLPVanilla(x.shape[1], [self.width] * self.layers, self.labtrans_.out_features,
                                      batch_norm=False, dropout=self.dropout)
        self.model_ = self.model_cls(net, tt.optim.Adam(lr=self.lr, weight_decay=self.weight_decay),
                                     duration_index=self.labtrans_.cuts, **self.loss_kwargs)
        self._train(x[tr], y_tr, x[va], y_va)
        return self


class DeepHitModel(_DiscreteBase):
    name, label = "deephit", "DeepHit"
    model_cls = DeepHitSingle
    loss_kwargs = {"alpha": 0.2, "sigma": 0.1}


class NMTLRModel(_DiscreteBase):
    name, label = "nmtlr", "N-MTLR"
    model_cls = MTLR


class CoxTimeModel(_NeuralBase):
    name, label = "coxtime", "Cox-Time"

    def fit(self, X, time, event):
        x = np.asarray(X, "float32")
        t, e = np.asarray(time, "float32"), np.asarray(event, "float32")
        tr, va = self._split(x, t, e)
        self.labtrans_ = CoxTime.label_transform()
        y_tr = self.labtrans_.fit_transform(t[tr], e[tr])
        y_va = self.labtrans_.transform(t[va], e[va])
        net = MLPVanillaCoxTime(x.shape[1], [self.width] * self.layers, batch_norm=False, dropout=self.dropout)
        self.model_ = CoxTime(net, tt.optim.Adam(lr=self.lr, weight_decay=self.weight_decay), labtrans=self.labtrans_)
        self._train(x[tr], y_tr, x[va], y_va)
        order = np.argsort(t, kind="stable")                 # pycox needs the baseline data sorted by time
        ts, es_ = self.labtrans_.transform(t[order], e[order])
        self.model_.compute_baseline_hazards(input=x[order], target=(ts, es_))
        return self      # predict_surv_df maps the time axis back to months itself
