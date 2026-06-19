import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import shutil  # used to copy existing artifacts into bootstrap folder
import re
import json
import time
import copy
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)
from sklearn.tree import DecisionTreeClassifier, export_text
from tree_encoding import TreeEncoder
from trainer import TTVAETrainer
from model import TTVAE
from surrogate import MLPPredictor


#split datasets sequences
def split_seqs(seqs, train_size):
    train_seqs, test_seqs = train_test_split(
        seqs,
        test_size=20000,
        random_state=42,
    )
    train_seqs, val_seqs = train_test_split(
        train_seqs,
        test_size=10000,
        random_state=42,
    )
    train_seqs, es_seqs = train_test_split(
        train_seqs,
        test_size=10000,
        random_state=42,
    )
    train_seqs = train_seqs[:train_size]
    return train_seqs, test_seqs, val_seqs, es_seqs



#extract latent representation from batch
def extract_latent_batches(model, trainer, dataset, batch_size=128):
    model.eval()
    all_mu = []

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_data = dataset[i:i + batch_size]

            batch = trainer.collate_fn(batch_data)
            batch = {
                k: v.to(model.device) if not isinstance(v, list) else v
                for k, v in batch.items()
            }

            src = batch["src"]
            src_abs_encs = batch["src_abs_encs"]
            src_rel_encs = batch["src_rel_encs"]

            z, mu, logvar, h = model.encode(src, src_abs_encs, src_rel_encs)
            all_mu.append(mu.cpu().detach().numpy())

    return np.concatenate(all_mu, axis=0)


def evaluate_latent(trainer, z):
    model = trainer.model
    model.eval()

    if not torch.is_tensor(z):
        z = torch.tensor(z, dtype=torch.float32, device=model.device)
    else:
        z = z.detach().float().to(model.device)

    if z.ndim == 1:
        z = z.unsqueeze(0)

    with torch.no_grad():
        output = model.generate(
            z=z,
            use_cache=False,
            do_sample=False,
        )

    return output


#objective and classification metrics
def compute_reference_metric(y_true, y_pred, metric="accuracy"):
    metrics = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "f1_macro": lambda y, p: f1_score(y, p, average="macro", zero_division=0),
        "f1_weighted": lambda y, p: f1_score(y, p, average="weighted", zero_division=0),
    }

    try:
        return metrics[metric](y_true, y_pred)
    except KeyError:
        raise ValueError(f"Unknown reference metric: {metric}")


def get_tree_n_leaves(clf):
    value = clf.get_n_leaves() if hasattr(clf, "get_n_leaves") else getattr(clf, "n_leaves", None)

    if value is None:
        raise AttributeError(
            "Could not find number of leaves. Expected get_n_leaves() or n_leaves."
        )

    return int(value)

def get_tree_depth(clf):
    value = clf.get_depth() if hasattr(clf, "get_depth") else getattr(clf, "max_depth", None)
    return None if value is None else int(value)


def get_tree_n_nodes(clf):
    value = clf.get_n_nodes() if hasattr(clf, "get_n_nodes") else getattr(
        getattr(clf, "tree_", None),
        "node_count",
        None,
    )
    return None if value is None else int(value)
    

def get_effective_regularizer(requested_lambda_reg, sample_size):
    """
        effective_lambda = min(requested_lambda_reg, 1 / sample_size)
    """
    lambda_cap = 1.0 / float(sample_size)
    effective_lambda_reg = min(float(requested_lambda_reg), lambda_cap)
    adjusted = bool(effective_lambda_reg < float(requested_lambda_reg))

    return effective_lambda_reg, lambda_cap, adjusted


def regularized_objective_from_metric(metric_value, n_leaves, lambda_reg):
    return float(metric_value) - float(lambda_reg) * int(n_leaves)

def bo_surrogate_target(metric_value, n_leaves, lambda_reg):
    return regularized_objective_from_metric(
        metric_value=metric_value,
        n_leaves=n_leaves,
        lambda_reg=lambda_reg,
    )

def classification_metrics_dict(y_true, y_pred, prefix):
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))

    out = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}_f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        f"{prefix}_f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        f"{prefix}_precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        f"{prefix}_recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_mcc": matthews_corrcoef(y_true, y_pred),
        f"{prefix}_cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out[f"{prefix}_labels"] = labels.tolist()
    out[f"{prefix}_confusion_matrix"] = cm.tolist()
    out[f"{prefix}_classification_report"] = classification_report(
        y_true,
        y_pred,
        zero_division=0,
        output_dict=True,
    )

    return out

if __name__ == "__main__":

    config_path = "../config/gradient_config.json"
    config_model_path = "../config/train_ttvae_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)
    with open(config_model_path, "r") as f:
        config_model = json.load(f)

    dataset_name = config["dataset_name"]
    sample_size = config["sample_size"]
    max_depth = config["max_depth"]
    precision = config["precision"]

    #initialize dataset, splits, tokenized sequences

    data_dir = config.get("data_dir", "../data/data_splitted")
    
    tokenized_trees_dir = config["tokenized_trees_dir"]

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

    te = TreeEncoder(
        X=X_train,
        y=y_train,
        tokenization="threshold",
        precision=precision,
    )

    feature_names = list(X_train.columns)
    regularizer_sample_size = len(X_train)
    regularizer_cap_1_over_sample_size = 1.0 / float(regularizer_sample_size)

    trees_seqs_perf_path = os.path.join(
        tokenized_trees_dir,
        f"{dataset_name}.joblib",
    )

    trees_seqs_perf = joblib.load(trees_seqs_perf_path)

    trees_seq = [tree_seq for tree_seq, tree_perf in trees_seqs_perf]


    size_of_trees = config['size_of_trees']

    train_seqs, test_seqs, val_seqs, es_seqs = split_seqs(
        trees_seq,
        size_of_trees,
    )

    train_seqs_perf, test_seqs_perf, val_seqs_perf, es_seqs_perf = split_seqs(
        trees_seqs_perf,
        size_of_trees,
    )

    #initialize model

    model = TTVAE(
        tree_encoder=te,
        max_depth=config["max_depth"],
        d_model_encoder=config_model["d_model_encoder"],
        d_model_decoder=config_model["d_model_decoder"],
        num_heads_encoder=config_model["num_heads_encoder"],
        num_heads_decoder=config_model["num_heads_decoder"],
        num_layers_encoder=config_model["num_layers_encoder"],
        num_layers_decoder=config_model["num_layers_decoder"],
        d_ff_encoder=config_model["d_ff_encoder"],
        d_ff_decoder=config_model["d_ff_decoder"],
        dropout_encoder=config_model["dropout_encoder"],
        dropout_decoder=config_model["dropout_decoder"],
        abs_pos_enc=config_model["abs_pos_enc"],
        rel_pos_enc=config_model["rel_pos_enc"],
        latent_dim=config_model["latent_dim"],
        max_len=config_model["max_len"],
        num_scales=config_model["num_scales"],
        weight_tying=config_model.get("weight_tying", False),
        device=config_model.get("device", "cuda"),
    )

    trainer = TTVAETrainer(
        model=model,
        train_trees=train_seqs,
        val_trees=val_seqs,
        es_trees=es_seqs,
        batch_size=config_model["batch_size"],
        pad_token_id=te.token_to_id["<PAD>"],
        cls_token_id=te.token_to_id["<CLS>"],
        bos_token_id=te.token_to_id["<BOS>"],
        eos_token_id=te.token_to_id["<EOS>"],
        unk_token_id=te.token_to_id["<UNK>"],
        shuffle=config_model.get("shuffle", True),
        seed=config_model.get("seed", 42),
    )

    #"saved_model_path" : "../experiments/trained_ttvae_models"
    saved_model_dir = config["saved_model_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = os.path.join(saved_model_dir, f"{dataset_name}.pt")
    
    ckp = torch.load(model_path, map_location=device)
    
    trainer.model.load_state_dict(ckp["model_state_dict"])
    trainer.model.to(device)
    trainer.model.eval()

    #check validation results for the loaded model
    val_res = trainer.evaluate(split="val")
    print("TTVAE validation result:")
    print(val_res)


        
    #latent features extraction
    #Z_train = extract_latent_batches(trainer.model, trainer, [x[0] for x in train_seqs_perf])
     
    Z_train = extract_latent_batches(
        trainer.model,
        trainer,
        [tree_seq for tree_seq, tree_perf in train_seqs_perf],
    )

    X_sgp = np.vstack(Z_train).astype(np.float64)

   
    
    all_results = []

    regularizations = config["regularizations"]
    reference_metric = config["reference_metric"]
    bo_iterations = config["bo_iterations"]
    candidate_batch_size = config["candidate_batch_size"]
    n_grid = config["n_grid"]
    sample_dist = config["sample_dist"]
    grad_ascent = config["grad_ascent"]
    ga_lr = config["ga_lr"]
    ga_steps = config["ga_steps"]

    random_seed_split = config.get("random_seed_split", 42)
    
    for requested_lambda_reg in regularizations:
    
        (
            effective_lambda_reg,
            lambda_cap_1_over_sample_size,
            lambda_reg_adjusted_due_to_sample_size,
        ) = get_effective_regularizer(
            requested_lambda_reg=requested_lambda_reg,
            sample_size=regularizer_sample_size,
        )
    
        lambda_reg = effective_lambda_reg
    
        y_sgp_values = []
    
        for tree_seq, perf_dict in train_seqs_perf:
            metric_value = get_train_metric_from_perf(
                perf_dict=perf_dict,
                reference_metric=reference_metric,
            )
    
            n_leaves = int(perf_dict["n_leaves"])
    
            target_value = bo_surrogate_target(
                metric_value=metric_value,
                n_leaves=n_leaves,
                lambda_reg=lambda_reg,
            )
    
            y_sgp_values.append(target_value)
    
        y_sgp = np.asarray(y_sgp_values, dtype=np.float64)
    
        X_train_sgp_base, X_test_sgp_base, y_train_sgp_base, y_test_sgp_base = train_test_split(
            X_sgp,
            y_sgp,
            test_size=0.2,
            random_state=42,
        )
    
        for rand_idx in range(config["bo_rand_start"], config["bo_rand_end"] + 1):
    
            torch.manual_seed(rand_idx)
            np.random.seed(rand_idx)
    
            if torch.cuda.is_available():
                torch.cuda.manual_seed(rand_idx)
                torch.cuda.manual_seed_all(rand_idx)
    
            X_train_sgp = X_train_sgp_base.copy()
            X_test_sgp = X_test_sgp_base.copy()
            y_train_sgp = y_train_sgp_base.copy()
            y_test_sgp = y_test_sgp_base.copy()
    
            mean_y_train_sgp = np.mean(y_train_sgp)
            std_y_train_sgp = np.std(y_train_sgp)
    
            if std_y_train_sgp == 0:
                raise ValueError("std_y_train_sgp is zero. Cannot normalize y_train_sgp.")
    
            y_train_sgp = (y_train_sgp - mean_y_train_sgp) / std_y_train_sgp
            y_test_sgp = (y_test_sgp - mean_y_train_sgp) / std_y_train_sgp
    
            mlp_predictor = MLPPredictor(
                input_dim=X_train_sgp.shape[1],
                hidden_dim=config["mlp_hidden_dim"],
                lr=config["mlp_lr"],
                weight_decay=config["mlp_weight_decay"],
                device=device,
            )
    
            mlp_predictor.fit(
                X_train=X_train_sgp,
                y_train=y_train_sgp,
                X_test=X_test_sgp,
                y_test=y_test_sgp,
                epochs=config["mlp_epochs"],
                batch_size=config["mlp_batch_size"],
                verbose=True,
            )
    
            best_score = -1e15
            best_clf = None
            best_seq = None
            best_feature = None
            best_iteration = 0
            best_candidate_info = {}
    
            for iteration in range(bo_iterations):
    
                mlp_predictor.eval()
    
                X_train_t = torch.tensor(
                    X_train_sgp,
                    dtype=torch.float32,
                    device=mlp_predictor.device,
                )
                X_test_t = torch.tensor(
                    X_test_sgp,
                    dtype=torch.float32,
                    device=mlp_predictor.device,
                )
                y_train_t = torch.tensor(
                    y_train_sgp,
                    dtype=torch.float32,
                    device=mlp_predictor.device,
                )
                y_test_t = torch.tensor(
                    y_test_sgp,
                    dtype=torch.float32,
                    device=mlp_predictor.device,
                )
    
                if y_train_t.ndim == 1:
                    y_train_t = y_train_t.unsqueeze(1)
    
                if y_test_t.ndim == 1:
                    y_test_t = y_test_t.unsqueeze(1)
    
                train_metrics = mlp_predictor.metrics(X_train_t, y_train_t)
                test_metrics = mlp_predictor.metrics(X_test_t, y_test_t)
    
                print(
                    f"lambda={requested_lambda_reg} | "
                    f"rand={rand_idx} | "
                    f"iter={iteration} | "
                    f"train pearson={train_metrics['pearson']:.4f} | "
                    f"test pearson={test_metrics['pearson']:.4f}"
                )
    
                if sample_dist == "standard_normal":
                    grid = np.random.randn(n_grid, X_train_sgp.shape[1])
    
                elif sample_dist == "normal":
                    grid = (
                        X_train_sgp.mean(axis=0)
                        + np.random.randn(n_grid, X_train_sgp.shape[1])
                        * X_train_sgp.std(axis=0)
                    )
    
                elif sample_dist == "uniform":
                    grid = (
                        X_train_sgp.min(axis=0)
                        + np.random.rand(n_grid, X_train_sgp.shape[1])
                        * (X_train_sgp.max(axis=0) - X_train_sgp.min(axis=0))
                    )
    
                else:
                    raise ValueError(f"Unknown sample_dist: {sample_dist}")
    
                grid_t = torch.tensor(
                    grid,
                    dtype=torch.float32,
                    device=mlp_predictor.device,
                )
    
                if grad_ascent:
                    grid_t = grid_t.detach().requires_grad_(True)
    
                    for _ in range(ga_steps):
                        pred = mlp_predictor(grid_t)
                        grads = torch.autograd.grad(pred.sum(), grid_t)[0]
                        grid_t = grid_t + ga_lr * grads
                        grid_t = grid_t.detach().requires_grad_(True)
    
                    with torch.no_grad():
                        pred = mlp_predictor(grid_t).detach().cpu().numpy()
    
                    selected_idxs = np.argsort(-pred[:, 0])[:candidate_batch_size]
                    next_inputs = grid_t[selected_idxs].detach().cpu().numpy()
    
                else:
                    with torch.no_grad():
                        pred = mlp_predictor(grid_t).detach().cpu().numpy()
    
                    selected_idxs = np.argsort(-pred[:, 0])[:candidate_batch_size]
                    next_inputs = grid[selected_idxs]
    
                tensor_next_inputs = torch.tensor(
                    next_inputs,
                    dtype=torch.float32,
                    device=trainer.model.device,
                )
    
                new_features = []
                new_scores = []
    
                for candidate_idx in range(candidate_batch_size):
    
                    try:
                        seq = evaluate_latent(
                            trainer,
                            tensor_next_inputs[candidate_idx],
                        ).cpu().detach().numpy()
    
                        clf = trainer.model.decoder.tree_encoder.decode_tree(
                            seq.tolist()[0]
                        )
    
                        y_pred_train_candidate = clf.predict(X_train)
    
                        metric_raw = compute_reference_metric(
                            y_train,
                            y_pred_train_candidate,
                            metric=reference_metric,
                        )
    
                        n_leaves = get_tree_n_leaves(clf)
                        n_nodes = get_tree_n_nodes(clf)
                        depth = get_tree_depth(clf)
    
                        score_raw = bo_surrogate_target(
                            metric_value=metric_raw,
                            n_leaves=n_leaves,
                            lambda_reg=lambda_reg,
                        )
    
                        score = (score_raw - mean_y_train_sgp) / std_y_train_sgp
    
                    except Exception as e:
                        print(f"[{candidate_idx}] Candidate failed: {e}")
                        continue
    
                    new_features.append(next_inputs[candidate_idx])
                    new_scores.append(score)
    
                    candidate_info = {
                        "dataset_name": dataset_name,
                        "size_of_trees": size_of_trees,
                        "reference_metric": reference_metric,
                        "requested_lambda_reg": requested_lambda_reg,
                        "effective_lambda_reg": effective_lambda_reg,
                        "rand_idx": rand_idx,
                        "iteration": iteration,
                        "candidate_idx": candidate_idx,
                        "metric_raw": metric_raw,
                        "score_raw": score_raw,
                        "score_normalized": score,
                        "n_leaves": n_leaves,
                        "n_nodes": n_nodes,
                        "depth": depth,
                    }
    
                    if score > best_score:
                        best_score = score
                        best_clf = clf
                        best_seq = seq
                        best_feature = next_inputs[candidate_idx].copy()
                        best_iteration = iteration
                        best_candidate_info = candidate_info
    
                        print(f"New best candidate found: score={best_score:.6f}")
    
                if len(new_features) > 0:
                    X_train_sgp = np.concatenate(
                        [X_train_sgp, np.vstack(new_features)],
                        axis=0,
                    )
    
                    y_train_sgp = np.concatenate(
                        [y_train_sgp, np.asarray(new_scores).reshape(-1, 1)],
                        axis=0,
                    )
    
                print(f"Ended iteration {iteration}")
                print(f"Current X_train_sgp size: {len(X_train_sgp)}")
    
            if best_clf is None:
                print(
                    f"No valid classifier found for lambda={requested_lambda_reg}, "
                    f"rand_idx={rand_idx}."
                )
                continue
    
            pruned_best_clf = best_clf.prune()
    
            y_pred_train_bayesian = pruned_best_clf.predict(X_train)
            y_pred_val_bayesian = pruned_best_clf.predict(X_val)
            y_pred_test_bayesian = pruned_best_clf.predict(X_test)
    
            train_ref_metric = compute_reference_metric(
                y_train,
                y_pred_train_bayesian,
                metric=reference_metric,
            )
            val_ref_metric = compute_reference_metric(
                y_val,
                y_pred_val_bayesian,
                metric=reference_metric,
            )
            test_ref_metric = compute_reference_metric(
                y_test,
                y_pred_test_bayesian,
                metric=reference_metric,
            )
    
            pruned_n_leaves = get_tree_n_leaves(pruned_best_clf)
            pruned_n_nodes = get_tree_n_nodes(pruned_best_clf)
            pruned_depth = get_tree_depth(pruned_best_clf)
    
            train_score = bo_surrogate_target(
                train_ref_metric,
                pruned_n_leaves,
                lambda_reg,
            )
            val_score = bo_surrogate_target(
                val_ref_metric,
                pruned_n_leaves,
                lambda_reg,
            )
            test_score = bo_surrogate_target(
                test_ref_metric,
                pruned_n_leaves,
                lambda_reg,
            )
    
            result_row = {
                "dataset_name": dataset_name,
                "size_of_trees": size_of_trees,
                "reference_metric": reference_metric,
                "requested_lambda_reg": requested_lambda_reg,
                "effective_lambda_reg": effective_lambda_reg,
                "rand_idx": rand_idx,
                "best_iteration": best_iteration,
                "best_score_normalized": best_score,
                "train_reference_metric": train_ref_metric,
                "val_reference_metric": val_ref_metric,
                "test_reference_metric": test_ref_metric,
                "train_regularized_score": train_score,
                "val_regularized_score": val_score,
                "test_regularized_score": test_score,
                "pruned_depth": pruned_depth,
                "pruned_n_leaves": pruned_n_leaves,
                "pruned_n_nodes": pruned_n_nodes,
                **{f"best_candidate_{k}": v for k, v in best_candidate_info.items()},
            }
    
            all_results.append(result_row)
    
            print("Best pruned classifier results:")
            print(result_row)
