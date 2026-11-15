"""
Preprocessing fitted on training samples only (so cross-validation estimates are honest).

Input: samples x proteins log2 matrix (normalised, replicates collapsed), NaN = missing.
Steps, all learned from the training samples and then applied to any new samples:
  1. keep proteins observed in at least min_valid (fraction) of the training samples;
  2. impute missing values:
       model     selection-model conditional mean E[y | not detected], with the protein's
                 mean / SD and the missingness curve fitted on the training samples by an
                 intercept-only model (no outcome or group information is used);
       min_prob  per-sample low quantile + noise (MinProb); the SD is learned on training
       mean      training mean of each protein
       zero      0 (log2 scale)
  3. optionally keep the max_features proteins with the largest training variance;
  4. standardise each protein to training mean 0, SD 1.
"""
import numpy as np
import pandas as pd
import torch

IMPUTERS = ("model", "min_prob", "mean", "zero")


class Preprocessor:
    def __init__(self, min_valid: float = 0.5, impute: str = "model",
                 max_features: int | None = None, scale: bool = True, seed: int = 0):
        if impute not in IMPUTERS:
            raise ValueError(f"impute must be one of {IMPUTERS}")
        self.min_valid, self.impute, self.max_features = min_valid, impute, max_features
        self.scale, self.seed = scale, seed

    def fit(self, Y: pd.DataFrame):
        V = Y.to_numpy(float)
        keep = (~np.isnan(V)).mean(0) >= self.min_valid
        if keep.sum() == 0:
            raise ValueError("no protein passes the min_valid filter")
        self.features_ = Y.columns[keep]
        V = V[:, keep]
        if self.impute == "model":
            from ..missingness import MissingParametersNR
            from ..model import DifferentialAbundanceNR
            Yt = torch.tensor(V)
            ones = torch.ones(V.shape[0], 1, dtype=torch.float64)
            b = torch.zeros(V.shape[0], dtype=torch.long)
            miss = MissingParametersNR(1).double()
            miss.fit(Yt, ones, b)
            self.w_ = float(miss.w.detach().ravel()[0])
            self.r_ = float(miss.r.detach().ravel()[0])
            m = DifferentialAbundanceNR(torch.tensor([self.w_]), torch.tensor([self.r_]))
            m.fit(Yt, ones, b, min_obs=2)
            mu = m.beta.numpy()[:, 0]
            sd = np.sqrt(m.s2.numpy())
            # proteins the model could not fit: fall back to observed mean / pooled SD
            obs_mu = np.nanmean(np.where(np.isnan(V), np.nan, V), 0)
            self.mu_ = np.where(np.isfinite(mu), mu, obs_mu)
            self.sd_imp_ = np.where(np.isfinite(sd), sd, np.nanmedian(sd))
        elif self.impute in ("mean", "min_prob"):
            self.mu_ = np.nanmean(V, 0)
            self.sd_imp_ = np.nanstd(V, 0, ddof=1)
            self.sd_pool_ = float(np.nanmedian(self.sd_imp_))
        Z = self._impute(V)
        if self.max_features is not None and self.max_features < Z.shape[1]:
            top = np.argsort(-Z.var(0))[: self.max_features]
            self._sel = np.sort(top)
        else:
            self._sel = np.arange(Z.shape[1])
        Z = Z[:, self._sel]
        self.center_ = Z.mean(0) if self.scale else np.zeros(Z.shape[1])
        sd = Z.std(0) if self.scale else np.ones(Z.shape[1])
        self.scale_ = np.where(sd > 0, sd, 1.0)
        self.feature_names_ = list(self.features_[self._sel])
        return self

    def _impute(self, V):
        V = V.copy()
        miss = np.isnan(V)
        if not miss.any():
            return V
        if self.impute == "zero":
            V[miss] = 0.0
        elif self.impute == "mean":
            V = np.where(miss, self.mu_[None, :], V)
        elif self.impute == "min_prob":
            rng = np.random.default_rng(self.seed)
            q = np.nanquantile(V, 0.01, axis=1)
            draws = rng.normal(q[:, None], self.sd_pool_, V.shape)
            V = np.where(miss, draws, V)
        else:
            from ..export import conditional_missing_mean
            mu = np.broadcast_to(self.mu_, V.shape)[miss]
            sd = np.broadcast_to(self.sd_imp_, V.shape)[miss]
            V[miss] = conditional_missing_mean(mu, sd, np.full(mu.shape, self.w_),
                                               np.full(mu.shape, self.r_))
        return V

    def transform(self, Y: pd.DataFrame) -> np.ndarray:
        V = Y.reindex(columns=self.features_).to_numpy(float)
        Z = self._impute(V)[:, self._sel]
        return (Z - self.center_) / self.scale_

    def fit_transform(self, Y):
        return self.fit(Y).transform(Y)
