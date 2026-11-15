"""
proteomix.ml: supervised learning on proteomics data with honest (nested) cross-validation.

    lasso       coordinate-descent elastic net: gaussian, binomial, multinomial, Cox
    knn         k-nearest neighbours (regression, classification)
    mlp         PyTorch multilayer perceptron (regression, classification, Cox survival)
    preprocess  fold-safe filtering, imputation (incl. selection-model), standardisation
    models      learners with inner-CV hyperparameter tuning
    cv          nested cross-validation, final models (save / load / predict)
    metrics     RMSE, R2, AUC, balanced accuracy, C-index, log-rank ...
    plots       model comparison, ROC, confusion matrix, Kaplan-Meier, lasso CV ...
    cli         `proteomix-ml train ...` / `proteomix-ml predict ...`
"""
from .cv import CVResult, TrainedModel, cross_validate, make_folds, train_final
from .knn import KNN
from .lasso import ElasticNet, cv_elastic_net
from .metrics import compute_metrics, concordance_index
from .mlp import MLP
from .models import LEARNERS, make_learner
from .preprocess import Preprocessor

__all__ = ["ElasticNet", "cv_elastic_net", "KNN", "MLP", "Preprocessor", "LEARNERS",
           "make_learner", "cross_validate", "train_final", "TrainedModel", "CVResult",
           "make_folds", "compute_metrics", "concordance_index"]
