import json
import os
import pickle
import time
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

#this should be pip installed or obtained from https://github.com/aia-uclouvain/pydl8.5

from pydl85 import DL85Classifier

from comp_utils import  * 


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def make_param_tag(
    max_depth,
    min_samples_leaf_frac,
    min_sup,
    time_limit,
):
    return (
        f"depth_{max_depth}"
        f"_leafFrac_{min_samples_leaf_frac:g}"
        f"_minSup_{min_sup}"
        f"_tl_{time_limit}"
    ).replace(".", "p")


if __name__ == "__main__":

    #config file is the same, be sure to change the experiments/original_dl8.5 folder to ensure no model overwriting 
  
    config_path = "../config/dl85_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    dataset_name = config["dataset_name"]
    precision = config["precision"]

    data_dir = config["data_dir"]
    models_root = config["models_dir"]
    results_root = config["results_dir"]

    max_depths = config["max_depths"]
    min_samples_leaf_fracs = config["min_samples_leaf_fracs"]
    time_limit = config["time_limit"]

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
        desc=f"Original DL8.5 {dataset_name}",
    ):
        min_sup_train = max(1, int(np.ceil(min_leaf_frac * n_train)))
        min_sup_train_val = max(1, int(np.ceil(min_leaf_frac * n_train_val)))

        param_tag = make_param_tag(
            max_depth=max_depth,
            min_samples_leaf_frac=min_leaf_frac,
            min_sup=min_sup_train,
            time_limit=time_limit,
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
        clf.fit(X_train, y_train)
        val_training_time = time.time() - val_start_time

        clf_test = DL85Classifier(
            max_depth=max_depth,
            min_sup=min_sup_train_val,
            time_limit=time_limit,
        )

        test_start_time = time.time()
        clf_test.fit(X_train_val, y_train_val)
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

        for split_name, metrics, stats, bootstrap_stats, min_sup, n_fit, training_time, model_path in [
            (
                "val",
                val_metrics,
                val_tree_stats,
                val_bootstrap_stats,
                min_sup_train,
                n_train,
                val_training_time,
                val_clf_path,
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
                "n_fit": n_fit,
                "training_time_seconds": training_time,
                "model_path": model_path,
            }

            row.update(metrics)
            row.update(stats)
            row.update(bootstrap_stats)

            results.append(row)

        print(
            f"Finished {dataset_name} | {param_tag} | "
            f"val training: {val_training_time:.6f}s | "
            f"test training: {test_training_time:.6f}s"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(metrics_csv_path, index=False)
    save_pickle(results_df, metrics_pkl_path)

    print(f"Saved original DL8.5 models to: {model_dir}")
    print(f"Saved original DL8.5 results to: {results_dir}")
