import json
import os
import pickle
import time

import pandas as pd
from tqdm import tqdm

from odtlearn.flow_oct import FlowOCT

from comp_utils import * 

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)

def make_param_tag(
    regularization,
    depth,
    time_limit,
):
    return (
        f"lambda_{regularization:g}"
        f"_depth_{depth}"
        f"_tl_{time_limit}"
    ).replace(".", "p")


if __name__ == "__main__":

    config_path = "../config/flow_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    dataset_name = config["dataset_name"]
    precision = config["precision"]

    data_dir = config["data_dir"]
    models_root = config["models_dir"]
    results_root = config["results_dir"]

    regularizations = config["regularizations"]
    depths = config["depths"]
    time_limit = config["time_limit"]
    num_threads = config["num_threads"]

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

    model_dir = os.path.join(models_root, dataset_name)
    results_dir = os.path.join(results_root, dataset_name)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    metrics_csv_path = os.path.join(
        results_dir,
        "metrics_all_lambdas_depths_bootstrap.csv",
    )
    metrics_pkl_path = os.path.join(
        results_dir,
        "metrics_all_lambdas_depths_bootstrap.pkl",
    )

    results = []

    for regularization in tqdm(
        regularizations,
        desc=f"FlowOCT regularizations {dataset_name}",
    ):
        for depth in depths:

            param_tag = make_param_tag(
                regularization=regularization,
                depth=depth,
                time_limit=time_limit,
            )

            val_artifact_pickle = f"flowoct_val_artifact_{param_tag}.pkl"
            test_artifact_pickle = f"flowoct_test_artifact_{param_tag}.pkl"

            val_artifact_path = os.path.join(model_dir, val_artifact_pickle)
            test_artifact_path = os.path.join(model_dir, test_artifact_pickle)

            # ==================================================
            # Validation model: train on train, evaluate on val
            # ==================================================
            flow_clf = FlowOCT(
                solver="gurobi",
                depth=depth,
                _lambda=regularization,
                time_limit=time_limit,
                num_threads=num_threads,
            )

            val_start_time = time.perf_counter()
            flow_clf.fit(X_train, y_train)
            val_training_time = time.perf_counter() - val_start_time

            val_artifact = extract_flowoct_artifact(
                flow_clf,
                params={
                    "solver": "gurobi",
                    "depth": depth,
                    "lambda": regularization,
                    "time_limit": time_limit,
                    "num_threads": num_threads,
                    "split": "val",
                    "trained_on": "train",
                    "evaluated_on": "val",
                },
            )

            save_pickle(val_artifact, val_artifact_path)

            y_pred_val = predict_from_flowoct_artifact(
                val_artifact,
                X_val,
            )

            val_metrics = compute_metrics_from_predictions(
                y_val,
                y_pred_val,
            )

            val_bootstrap_stats = compute_bootstrap_metrics_from_predictions(
                y_val,
                y_pred_val,
                n_iter=bootstrap_n_iter,
                seed=bootstrap_seed,
            )

            # ==================================================
            # Test model: train on train+val, evaluate on test
            # ==================================================
            flow_clf_test = FlowOCT(
                solver="gurobi",
                depth=depth,
                _lambda=regularization,
                time_limit=time_limit,
                num_threads=num_threads,
            )

            test_start_time = time.perf_counter()
            flow_clf_test.fit(X_train_val, y_train_val)
            test_training_time = time.perf_counter() - test_start_time

            test_artifact = extract_flowoct_artifact(
                flow_clf_test,
                params={
                    "solver": "gurobi",
                    "depth": depth,
                    "lambda": regularization,
                    "time_limit": time_limit,
                    "num_threads": num_threads,
                    "split": "test",
                    "trained_on": "train+val",
                    "evaluated_on": "test",
                },
            )

            save_pickle(test_artifact, test_artifact_path)

            y_pred_test = predict_from_flowoct_artifact(
                test_artifact,
                X_test,
            )

            test_metrics = compute_metrics_from_predictions(
                y_test,
                y_pred_test,
            )

            test_bootstrap_stats = compute_bootstrap_metrics_from_predictions(
                y_test,
                y_pred_test,
                n_iter=bootstrap_n_iter,
                seed=bootstrap_seed,
            )

            for split_name, artifact, artifact_pickle, artifact_path, metrics, bootstrap_stats, training_time in [
                (
                    "val",
                    val_artifact,
                    val_artifact_pickle,
                    val_artifact_path,
                    val_metrics,
                    val_bootstrap_stats,
                    val_training_time,
                ),
                (
                    "test",
                    test_artifact,
                    test_artifact_pickle,
                    test_artifact_path,
                    test_metrics,
                    test_bootstrap_stats,
                    test_training_time,
                ),
            ]:
                tree_stats = artifact["tree_stats"]

                row = {
                    "dataset": dataset_name,
                    "split": split_name,
                    "model_file_tag": param_tag,
                    "lambda": regularization,
                    "depth": depth,
                    "time_limit": time_limit,
                    "num_threads": num_threads,
                    "training_time_seconds": training_time,
                    "actual_depth": tree_stats["actual_depth"],
                    "actual_n_nodes": tree_stats["actual_n_nodes"],
                    "actual_n_leaves": tree_stats["actual_n_leaves"],
                    "actual_n_internal_nodes": tree_stats["actual_n_internal_nodes"],
                    "actual_nodes": tree_stats["actual_nodes"],
                    "actual_leaf_nodes": tree_stats["actual_leaf_nodes"],
                    "actual_internal_nodes": tree_stats["actual_internal_nodes"],
                    "artifact_pickle": artifact_pickle,
                    "artifact_path": artifact_path,
                }

                row.update(metrics)
                row.update(bootstrap_stats)

                results.append(row)

            print(f"Finished {dataset_name} | {param_tag}")

    results_df = pd.DataFrame(results)

    results_df.to_csv(metrics_csv_path, index=False)
    save_pickle(results_df, metrics_pkl_path)

    print(f"Saved FlowOCT artifacts to: {model_dir}")
    print(f"Saved FlowOCT results to: {results_dir}")
