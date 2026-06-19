import json
import os
import pickle
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.tree import export_text
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
)

from tree_encoding import TreeEncoder

def _safe_metric(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return np.nan

def classification_metrics(y_true, y_pred, y_score=None) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out = {
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
        "f1_macro": _safe_metric(
            f1_score,
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_micro": _safe_metric(
            f1_score,
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "f1_weighted": _safe_metric(
            f1_score,
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "precision_macro": _safe_metric(
            precision_score,
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "precision_micro": _safe_metric(
            precision_score,
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "precision_weighted": _safe_metric(
            precision_score,
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall_macro": _safe_metric(
            recall_score,
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_micro": _safe_metric(
            recall_score,
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "recall_weighted": _safe_metric(
            recall_score,
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    cm = _safe_metric(confusion_matrix, y_true, y_pred)
    out["confusion_matrix"] = cm.tolist() if isinstance(cm, np.ndarray) else np.nan

    if y_score is not None:
        y_score = np.asarray(y_score)
        out["log_loss"] = _safe_metric(log_loss, y_true, y_score)

        unique_classes = np.unique(y_true)
        is_binary = len(unique_classes) == 2

        if is_binary:
            if y_score.ndim == 2 and y_score.shape[1] >= 2:
                pos_score = y_score[:, 1]
            else:
                pos_score = y_score.ravel()

            out["roc_auc"] = _safe_metric(roc_auc_score, y_true, pos_score)
            out["average_precision"] = _safe_metric(
                average_precision_score,
                y_true,
                pos_score,
            )
            out["brier_score"] = _safe_metric(
                brier_score_loss,
                y_true,
                pos_score,
            )
        else:
            out["roc_auc_ovr_macro"] = _safe_metric(
                roc_auc_score,
                y_true,
                y_score,
                multi_class="ovr",
                average="macro",
            )
            out["roc_auc_ovr_weighted"] = _safe_metric(
                roc_auc_score,
                y_true,
                y_score,
                multi_class="ovr",
                average="weighted",
            )
            out["roc_auc_ovo_macro"] = _safe_metric(
                roc_auc_score,
                y_true,
                y_score,
                multi_class="ovo",
                average="macro",
            )
            out["roc_auc_ovo_weighted"] = _safe_metric(
                roc_auc_score,
                y_true,
                y_score,
                multi_class="ovo",
                average="weighted",
            )

    return out


def is_decoded_decision_tree(tree):
    return "DecodedDecisionTree" in str(type(tree))


def infer_tree_stats_from_export_text(tree_text: str) -> Dict[str, Any]:
    lines = [line for line in tree_text.splitlines() if line.strip()]

    n_nodes = len(lines)
    n_leaves = 0
    n_internal_nodes = 0
    max_depth = 0

    for line in lines:
        depth = line.count("|   ")
        max_depth = max(max_depth, depth)

        if "class:" in line:
            n_leaves += 1
        else:
            n_internal_nodes += 1

    return {
        "n_nodes": n_nodes,
        "n_leaves": n_leaves,
        "n_splits": n_internal_nodes,
        "n_internal_nodes": n_internal_nodes,
        "max_depth": max_depth,
    }


def evaluate_one_tree(
    tree_id,
    tree,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
    row = {"tree_id": tree_id}

    if is_decoded_decision_tree(tree):
        n_nodes = _safe_metric(tree.get_n_nodes)
        n_leaves = _safe_metric(tree.get_n_leaves)

        row.update({
            "n_nodes": n_nodes,
            "n_leaves": n_leaves,
            "n_splits": (
                n_nodes - n_leaves
                if not np.isnan(n_nodes) and not np.isnan(n_leaves)
                else np.nan
            ),
            "n_internal_nodes": (
                n_nodes - n_leaves
                if not np.isnan(n_nodes) and not np.isnan(n_leaves)
                else np.nan
            ),
            "max_depth": _safe_metric(tree.get_depth),
        })

    elif hasattr(tree, "tree_"):
        tree_text = export_text(tree)
        stats = infer_tree_stats_from_export_text(tree_text)
        row.update(stats)

        row.update({
            "n_features": int(getattr(tree, "n_features_in_", np.nan)),
            "n_outputs": int(tree.n_outputs_) if hasattr(tree, "n_outputs_") else np.nan,
            "n_classes": (
                int(tree.n_classes_)
                if np.isscalar(tree.n_classes_)
                else list(tree.n_classes_)
            ),
            "export_text": tree_text,
        })

    else:
        raise TypeError(f"Unsupported tree type: {type(tree)}")

    for split_name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        y_pred = tree.predict(X)

        y_score = None
        if hasattr(tree, "predict_proba"):
            y_score = tree.predict_proba(X)
        elif hasattr(tree, "decision_function"):
            y_score = tree.decision_function(X)

        metrics = classification_metrics(y, y_pred, y_score)

        for metric_name, metric_value in metrics.items():
            row[f"{split_name}_{metric_name}"] = metric_value

    return row


def evaluate_trees(
    trees,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
    rows = []

    for tree_id, tree in tqdm(
        enumerate(trees),
        total=len(trees),
        desc="Evaluating trees",
    ):
        row = evaluate_one_tree(
            tree_id=tree_id,
            tree=tree,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
        rows.append(row)

    df = pd.DataFrame(rows).set_index("tree_id")
    df = df.sort_index()

    return df


if __name__ == "__main__":

    config_path = "../config/tokenize_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    dataset_name = config["dataset_name"]
    sample_size = config["sample_size"]
    max_depth = config["max_depth"]
    precision = config["precision"]

    base_dir_tree = config["base_dir_tree"]
    target_dir = config["target_dir"]

    data_dir = config.get("data_dir", "../data/data_splitted")

    os.makedirs(target_dir, exist_ok=True)

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

    trees_path = os.path.join(base_dir_tree, f"{dataset_name}.joblib")
    trees = joblib.load(trees_path)

    print(f"Loaded trees: {len(trees)}")

    df_tree_results = evaluate_trees(
        trees=trees,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    te = TreeEncoder(
        X=X_train,
        y=y_train,
        tokenization="threshold",
        precision=precision,
    )

    seqs_with_performance = []

    for tree_id, tree in tqdm(
        enumerate(trees),
        total=len(trees),
        desc=f"Linearizing {dataset_name}",
    ):
        tree_seq = te.encode_tree(tree)
        tree_perf = df_tree_results.loc[tree_id].to_dict()

        seqs_with_performance.append((tree_seq,tree_perf,))

    assert len(seqs_with_performance) == len(df_tree_results), (
        f"len(seqs_with_performance)={len(seqs_with_performance)} but "
        f"len(df_tree_results)={len(df_tree_results)}"
    )

    output_path = os.path.join(target_dir, f"{dataset_name}.joblib")

    joblib.dump(
        seqs_with_performance,
        output_path,
    )

    print(f"Saved sequences with performance to: {output_path}")
