import json
import os
import pickle
import time
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier


#consider the dl85 with warm labels introduced in https://github.com/ubc-systopia/pydl8.5-lbguess

from dl85 import DL85Classifier

from comp_utils import *

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)

def make_param_tag(
    max_depth,
    min_samples_leaf_frac,
    min_sup,
    time_limit,
    gbdt_n_est,
    gbdt_max_depth,
):
    return (
        f"depth_{max_depth}"
        f"_leafFrac_{min_samples_leaf_frac:g}"
        f"_minSup_{min_sup}"
        f"_tl_{time_limit}"
        f"_gbdtN_{gbdt_n_est}"
        f"_gbdtD_{gbdt_max_depth}"
    ).replace(".", "p")


def fit_dl85_with_optional_warm(clf, X, y, warm):
    try:
        clf.fit(X, y, warm=warm)
    except TypeError:
        print("WARNING: This dl85 version did not accept warm=. Fitting without warm.")
        clf.fit(X, y)

    return clf


if __name__ == "__main__":

    config_path = "../config/dl85_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    dataset_name = config["dataset_name"]
    precision = config["precision"]
    random_state = config.get("random_state", 42)

    data_dir = config["data_dir"]
    gbdt_hyperparam_dir = config["gbdt_hyperparam_dir"]
    models_root = config["models_dir"]
    results_root = config["results_dir"]

    max_depths = config["max_depths"]
    min_samples_leaf_fracs = config["min_samples_leaf_fracs"]
    time_limit = config["time_limit"]

    default_gbdt_n_est = config.get("default_gbdt_n_est", 100)
    default_gbdt_max_depth = config.get("default_gbdt_max_depth", 3)

    bootstrap_n_iter = config.get("bootstrap_n_iter", 200)
    bootstrap_seed = config.get("bootstrap_seed", 12345)

    data_path = os.path.join(data_dir, f"{dataset_name}.csv")
    data = pd.read_csv(data_path)

    train = data[data["split"] == "train"].drop("split", axis=1)
    val = data[data["split"] == "val"].drop("split", axis=1)
    test = data[data["split"] == "test"].drop("split", axis=1)

    X_train = train.drop("target", axis=1).round(precision)
    y_train = train["target"]

    X_val = val.drop("target", axis=1).round(precision)
    y_val = val["target"]

    X_test = test.drop("target", axis=1).round(precision)
    y_test = test["target"]

    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    n_train = len(X_train)
    n_train_val = len(X_train_val)

    gbdt_hyperparam_path = os.path.join(
        gbdt_hyperparam_dir,
        f"res_{dataset_name}.csv",
    )

    if os.path.exists(gbdt_hyperparam_path):
        gbdt_hyperparam = pd.read_csv(gbdt_hyperparam_path).iloc[0]
        gbdt_n_est = int(gbdt_hyperparam["GBDT_N_EST"])
        gbdt_max_depth = int(gbdt_hyperparam["GBDT_MAX_DEPTH"])
    else:
        gbdt_n_est = default_gbdt_n_est
        gbdt_max_depth = default_gbdt_max_depth

    enc = GradientBoostingClassifier(
        n_estimators=gbdt_n_est,
        max_depth=gbdt_max_depth,
        random_state=random_state,
    )

    enc.fit(X_train, y_train)
    gbdt_pred_train = enc.predict(X_train)
    warm_train = (gbdt_pred_train == y_train.to_numpy()).astype(int)

    enc_train_val = GradientBoostingClassifier(
        n_estimators=gbdt_n_est,
        max_depth=gbdt_max_depth,
        random_state=random_state,
    )

    enc_train_val.fit(X_train_val, y_train_val)
    gbdt_pred_train_val = enc_train_val.predict(X_train_val)
    warm_train_val = (gbdt_pred_train_val == y_train_val.to_numpy()).astype(int)

    warm_train_accuracy = float(np.mean(warm_train))
    warm_train_val_accuracy = float(np.mean(warm_train_val))

    model_dir = os.path.join(models_root, dataset_name)
    results_dir = os.path.join(results_root, dataset_name)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    metrics_csv_path = os.path.join(
        results_dir,
        "metrics_all_hyperparameters_bootstrap.csv",
    )
    metrics_pkl_path = os.path.join(
        results_dir,
        "metrics_all_hyperparameters_bootstrap.pkl",
    )

    results = []

    hyperparameter_grid = list(product(
        max_depths,
        min_samples_leaf_fracs,
    ))

    for max_depth, min_leaf_frac in tqdm(
        hyperparameter_grid,
        total=len(hyperparameter_grid),
        desc=f"DL8.5 {dataset_name}",
    ):
        min_sup_train = max(1, int(np.ceil(min_leaf_frac * n_train)))
        min_sup_train_val = max(1, int(np.ceil(min_leaf_frac * n_train_val)))

        param_tag = make_param_tag(
            max_depth=max_depth,
            min_samples_leaf_frac=min_leaf_frac,
            min_sup=min_sup_train,
            time_limit=time_limit,
            gbdt_n_est=gbdt_n_est,
            gbdt_max_depth=gbdt_max_depth,
        )

        val_clf_path = os.path.join(
            model_dir,
            f"dl85_val_clf_{param_tag}.pkl",
        )
        test_clf_path = os.path.join(
            model_dir,
            f"dl85_test_clf_{param_tag}.pkl",
        )

        clf = DL85Classifier(
            max_depth=max_depth,
            min_sup=min_sup_train,
            time_limit=time_limit,
        )

        val_start_time = time.time()

        clf = fit_dl85_with_optional_warm(
            clf=clf,
            X=X_train,
            y=y_train,
            warm=warm_train,
        )

        val_training_time = time.time() - val_start_time

        clf_test = DL85Classifier(
            max_depth=max_depth,
            min_sup=min_sup_train_val,
            time_limit=time_limit,
        )

        test_start_time = time.time()

        clf_test = fit_dl85_with_optional_warm(
            clf=clf_test,
            X=X_train_val,
            y=y_train_val,
            warm=warm_train_val,
        )

        test_training_time = time.time() - test_start_time

        save_pickle(clf, val_clf_path)
        save_pickle(clf_test, test_clf_path)

        val_metrics = compute_metrics(clf, X_val, y_val)
        test_metrics = compute_metrics(clf_test, X_test, y_test)

        val_tree_stats = compute_dl85_tree_stats(clf)
        test_tree_stats = compute_dl85_tree_stats(clf_test)

        val_bootstrap_stats = compute_bootstrap_metrics_ci(
            clf,
            X_val,
            y_val,
            n_iter=bootstrap_n_iter,
            seed=bootstrap_seed,
        )
        test_bootstrap_stats = compute_bootstrap_metrics_ci(
            clf_test,
            X_test,
            y_test,
            n_iter=bootstrap_n_iter,
            seed=bootstrap_seed,
        )

        for split_name, metrics, stats, bootstrap_stats, min_sup, n_fit, training_time, model_path, warm_accuracy in [
            (
                "val",
                val_metrics,
                val_tree_stats,
                val_bootstrap_stats,
                min_sup_train,
                n_train,
                val_training_time,
                val_clf_path,
                warm_train_accuracy,
            ),
            (
                "test",
                test_metrics,
                test_tree_stats,
                test_bootstrap_stats,
                min_sup_train_val,
                n_train_val,
                test_training_time,
                test_clf_path,
                warm_train_val_accuracy,
            ),
        ]:
            row = {
                "dataset": dataset_name,
                "split": split_name,
                "model_file_tag": param_tag,
                "max_depth": max_depth,
                "min_samples_leaf_frac": min_leaf_frac,
                "min_sup": min_sup,
                "time_limit": time_limit,
                "gbdt_n_est": gbdt_n_est,
                "gbdt_max_depth": gbdt_max_depth,
                "n_fit": n_fit,
                "gbdt_warm_accuracy_on_fit_set": warm_accuracy,
                "training_time_seconds": training_time,
                "model_path": model_path,
                "uses_warm_labels": True,
            }

            row.update(metrics)
            row.update(stats)
            row.update(bootstrap_stats)

            results.append(row)

        print(
            f"Finished {dataset_name} | {param_tag} | "
            f"val warm acc: {warm_train_accuracy:.4f} | "
            f"test warm acc: {warm_train_val_accuracy:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(metrics_csv_path, index=False)
    save_pickle(results_df, metrics_pkl_path)

    print(f"Saved DL8.5 models to: {model_dir}")
    print(f"Saved DL8.5 results to: {results_dir}")
