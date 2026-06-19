import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier

from gosdt import GOSDTClassifier
from tree_encoding import TreeEncoder
from comp_utils import *


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def make_param_tag(
    regularization,
    depth_budget,
    similar_support,
    time_limit,
    gbdt_n_est,
    gbdt_max_depth,
):
    return (
        f"reg_{regularization:g}"
        f"_depth_{depth_budget}"
        f"_sim_{similar_support}"
        f"_tl_{time_limit}"
        f"_gbdtN_{gbdt_n_est}"
        f"_gbdtD_{gbdt_max_depth}"
    ).replace(".", "p")


def make_gbdt_tag(gbdt_n_est, gbdt_max_depth):
    return f"gbdtN_{gbdt_n_est}_gbdtD_{gbdt_max_depth}"


if __name__ == "__main__":

    config_path = "../config/gosdt_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    dataset_name = config["dataset_name"]
    precision = config["precision"]
    random_state = config.get("random_state", 42)

    data_dir = config["data_dir"]
    gbdt_hyperparam_dir = config["gbdt_hyperparam_dir"]

    models_root = config["models_dir"]
    results_root = config["results_dir"]

    regularizations = config["regularizations"]
    similar_support = config["similar_support"]
    depth_budget = config["depth_budget"]
    time_limit = config["time_limit"]
    verbose = config.get("verbose", True)

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
        "metrics_all_regularizations_bootstrap.csv",
    )
    metrics_pkl_path = os.path.join(
        results_dir,
        "metrics_all_regularizations_bootstrap.pkl",
    )

    gbdt_hyperparam_path = os.path.join(
        gbdt_hyperparam_dir,
        f"res_{dataset_name}.csv",
    )
    gbdt_hyperparam = pd.read_csv(gbdt_hyperparam_path).iloc[0]

    #consider gradient boosting estimation for warm labels for gosdt lower-bound

    gbdt_n_est = int(gbdt_hyperparam["GBDT_N_EST"])
    gbdt_max_depth = int(gbdt_hyperparam["GBDT_MAX_DEPTH"])

    enc = GradientBoostingClassifier(
        n_estimators=gbdt_n_est,
        max_depth=gbdt_max_depth,
        random_state=random_state,
    )

    gbdt_val_start_time = time.time()
    enc.fit(X_train, y_train)
    gbdt_val_training_time = time.time() - gbdt_val_start_time
    warm_labels = enc.predict(X_train)

    enc_train_val = GradientBoostingClassifier(
        n_estimators=gbdt_n_est,
        max_depth=gbdt_max_depth,
        random_state=random_state,
    )

    gbdt_test_start_time = time.time()
    enc_train_val.fit(X_train_val, y_train_val)
    gbdt_test_training_time = time.time() - gbdt_test_start_time
    warm_labels_train_val = enc_train_val.predict(X_train_val)

    results = []

    gbdt_tag = make_gbdt_tag(
        gbdt_n_est=gbdt_n_est,
        gbdt_max_depth=gbdt_max_depth,
    )

    for split_name, model, X_eval, y_eval, training_time in [
        ("val", enc, X_val, y_val, gbdt_val_training_time),
        ("test", enc_train_val, X_test, y_test, gbdt_test_training_time),
    ]:
        row = {
            "dataset": dataset_name,
            "split": split_name,
            "model": "gbdt_warmup",
            "model_file_tag": gbdt_tag,
            "regularization": np.nan,
            "similar_support": np.nan,
            "depth_budget": np.nan,
            "time_limit": np.nan,
            "gbdt_n_est": gbdt_n_est,
            "gbdt_max_depth": gbdt_max_depth,
            "training_time_seconds": training_time,
            "decoded_depth": np.nan,
            "decoded_n_leaves": np.nan,
            "decoded_n_nodes": np.nan,
        }

        row.update(compute_metrics(model, X_eval, y_eval))
        row.update(
            compute_bootstrap_metrics_ci(
                model,
                X_eval,
                y_eval,
                n_iter=bootstrap_n_iter,
                seed=bootstrap_seed,
            )
        )

        results.append(row)

    X_train_swap = X_train.replace({0: 1, 1: 0})

    te = TreeEncoder(
        X=X_train_swap,
        y=y_train,
        tokenization="threshold",
        precision=precision,
    )


    for regularization in tqdm(
        regularizations,
        desc=f"GOSDT {dataset_name}",
    ):
        param_tag = make_param_tag(
            regularization=regularization,
            depth_budget=depth_budget,
            similar_support=similar_support,
            time_limit=time_limit,
            gbdt_n_est=gbdt_n_est,
            gbdt_max_depth=gbdt_max_depth,
        )

        clf = GOSDTClassifier(
            regularization=regularization,
            similar_support=similar_support,
            time_limit=time_limit,
            depth_budget=depth_budget,
            verbose=verbose,
        )

        gosdt_val_start_time = time.time()

        try:
            clf.fit(X_train, y_train, y_ref=warm_labels)
        except Exception:
            clf.fit(X_train, y_train)

        gosdt_val_training_time = time.time() - gosdt_val_start_time

        clf_test = GOSDTClassifier(
            regularization=regularization,
            similar_support=similar_support,
            time_limit=time_limit,
            depth_budget=depth_budget,
            verbose=verbose,
        )

        gosdt_test_start_time = time.time()

        try:
            clf_test.fit(X_train_val, y_train_val, y_ref=warm_labels_train_val)
        except Exception:
            clf_test.fit(X_train_val, y_train_val)

        gosdt_test_training_time = time.time() - gosdt_test_start_time

        save_pickle(
            clf,
            os.path.join(model_dir, f"gosdt_val_clf_{param_tag}.pkl"),
        )
        save_pickle(
            clf_test,
            os.path.join(model_dir, f"gosdt_test_clf_{param_tag}.pkl"),
        )

        gosdt_decoded = decode_gosdt_tree(clf, te)
        gosdt_decoded_test = decode_gosdt_tree(clf_test, te)

        save_pickle(
            gosdt_decoded,
            os.path.join(model_dir, f"gosdt_val_decoded_{param_tag}.pkl"),
        )
        save_pickle(
            gosdt_decoded_test,
            os.path.join(model_dir, f"gosdt_test_decoded_{param_tag}.pkl"),
        )

        for split_name, model, decoded_model, X_eval, y_eval, training_time in [
            ("val", clf, gosdt_decoded, X_val, y_val, gosdt_val_training_time),
            (
                "test",
                clf_test,
                gosdt_decoded_test,
                X_test,
                y_test,
                gosdt_test_training_time,
            ),
        ]:
            row = {
                "dataset": dataset_name,
                "split": split_name,
                "model": "gosdt",
                "model_file_tag": param_tag,
                "regularization": regularization,
                "similar_support": similar_support,
                "depth_budget": depth_budget,
                "time_limit": time_limit,
                "gbdt_n_est": gbdt_n_est,
                "gbdt_max_depth": gbdt_max_depth,
                "training_time_seconds": training_time,
                "decoded_depth": decoded_model.get_depth(),
                "decoded_n_leaves": decoded_model.get_n_leaves(),
                "decoded_n_nodes": decoded_model.get_n_nodes(),
            }

            row.update(compute_metrics(model, X_eval, y_eval))
            row.update(
                compute_bootstrap_metrics_ci(
                    model,
                    X_eval,
                    y_eval,
                    n_iter=bootstrap_n_iter,
                    seed=bootstrap_seed,
                )
            )

            results.append(row)

        print(f"Finished {dataset_name} | {param_tag}")

    results_df = pd.DataFrame(results)

    results_df.to_csv(metrics_csv_path, index=False)
    save_pickle(results_df, metrics_pkl_path)

    print(f"Saved models to: {model_dir}")
    print(f"Saved results to: {results_dir}")
