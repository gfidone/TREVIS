import json
import os
import pickle
import time
from itertools import product
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier

from comp_utils import *

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)

def make_param_tag(
    max_depth,
    min_samples_leaf_frac,
    min_samples_leaf,
    min_samples_split,
    ccp_alpha,
    max_leaf_nodes=None,
):
    leaf_frac_tag = (
        "fixed"
        if min_samples_leaf_frac is None
        else f"{min_samples_leaf_frac:g}"
    )

    tag = (
        f"depth_{max_depth}"
        f"_leafFrac_{leaf_frac_tag}"
        f"_leaf_{min_samples_leaf}"
        f"_split_{min_samples_split}"
        f"_ccp_{ccp_alpha:g}"
    )

    if max_leaf_nodes is not None:
        tag += f"_maxLeaves_{max_leaf_nodes}"

    return tag.replace(".", "p")


if __name__ == "__main__":

    config_path = "../config/cart_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    dataset_name = config["dataset_name"]
    precision = config["precision"]
    random_state = config.get("random_state", 42)

    data_dir = config.get("data_dir", "../data/data_splitted")
    model_root = config["model_dir"]
    results_root = config["results_dir"]

    max_depths = config["max_depths"]
    min_samples_leaf_fracs = config["min_samples_leaf_fracs"]
    ccp_alphas = config["ccp_alphas"]
    min_samples_split = config.get("min_samples_split", 2)
    max_leaf_nodes = config.get("max_leaf_nodes", None)

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

    model_dir = os.path.join(model_root, dataset_name)
    results_dir = os.path.join(results_root, dataset_name)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    results = []

    hyperparameter_grid = list(product(
        max_depths,
        min_samples_leaf_fracs,
        ccp_alphas,
    ))

    for max_depth, min_leaf_frac, ccp_alpha in tqdm(
        hyperparameter_grid,
        total=len(hyperparameter_grid),
        desc=f"CART grid {dataset_name}",
    ):
        n_train = len(X_train)
        n_train_val = len(X_train_val)

        min_samples_leaf_train = max(1, int(np.ceil(min_leaf_frac * n_train)))
        min_samples_leaf_train_val = max(1, int(np.ceil(min_leaf_frac * n_train_val)))

        param_tag = make_param_tag(
            max_depth=max_depth,
            min_samples_leaf_frac=min_leaf_frac,
            min_samples_leaf=min_samples_leaf_train,
            min_samples_split=min_samples_split,
            ccp_alpha=ccp_alpha,
            max_leaf_nodes=max_leaf_nodes,
        )

        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf_train,
            min_samples_split=min_samples_split,
            ccp_alpha=ccp_alpha,
            random_state=random_state,
        )

        train_start_time = time.perf_counter()
        clf.fit(X_train, y_train)
        clf = prune_duplicate_leaves(clf)
        train_fit_time_sec = time.perf_counter() - train_start_time

        clf_test = DecisionTreeClassifier(
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf_train_val,
            min_samples_split=min_samples_split,
            ccp_alpha=ccp_alpha,
            random_state=random_state,
        )

        train_val_start_time = time.perf_counter()
        clf_test.fit(X_train_val, y_train_val)
        clf_test = prune_duplicate_leaves(clf_test)
        train_val_fit_time_sec = time.perf_counter() - train_val_start_time

        save_pickle(
            clf,
            os.path.join(model_dir, f"sklearn_tree_val_clf_{param_tag}.pkl"),
        )
        save_pickle(
            clf_test,
            os.path.join(model_dir, f"sklearn_tree_test_clf_{param_tag}.pkl"),
        )

        val_metrics = compute_metrics(clf, X_val, y_val)
        test_metrics = compute_metrics(clf_test, X_test, y_test)

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

        val_tree_stats = compute_tree_stats_after_pruning(
            clf,
            feature_names=X_train.columns,
        )
        test_tree_stats = compute_tree_stats_after_pruning(
            clf_test,
            feature_names=X_train_val.columns,
        )

        split_rows = [
            (
                "val",
                val_metrics,
                val_tree_stats,
                val_bootstrap_stats,
                min_samples_leaf_train,
                n_train,
                train_fit_time_sec,
            ),
            (
                "test",
                test_metrics,
                test_tree_stats,
                test_bootstrap_stats,
                min_samples_leaf_train_val,
                n_train_val,
                train_val_fit_time_sec,
            ),
        ]

        for split_name, metrics, tree_stats, bootstrap_stats, min_samples_leaf, n_fit, fit_time_sec in split_rows:
            row = {
                "dataset": dataset_name,
                "split": split_name,
                "model_file_tag": param_tag,
                "max_depth": max_depth,
                "max_leaf_nodes": max_leaf_nodes,
                "min_samples_leaf_frac": min_leaf_frac,
                "min_samples_leaf": min_samples_leaf,
                "min_samples_split": min_samples_split,
                "ccp_alpha": ccp_alpha,
                "n_fit": n_fit,
                "fit_time_sec": fit_time_sec,
                "train_fit_time_sec": train_fit_time_sec,
                "train_val_fit_time_sec": train_val_fit_time_sec,
            }

            row.update(metrics)
            row.update(tree_stats)
            row.update(bootstrap_stats)

            results.append(row)

        print(
            f"Finished {dataset_name} | {param_tag} | "
            f"val leaves: {val_tree_stats['tree_n_leaves_after_pruning']} | "
            f"test leaves: {test_tree_stats['tree_n_leaves_after_pruning']}"
        )

    results_df = pd.DataFrame(results)

    csv_path = os.path.join(results_dir, "metrics_all_hyperparameters.csv")
    pkl_path = os.path.join(results_dir, "metrics_all_hyperparameters.pkl")

    results_df.to_csv(csv_path, index=False)
    save_pickle(results_df, pkl_path)

    print(f"Saved models to: {model_dir}")
    print(f"Saved results to: {results_dir}")
