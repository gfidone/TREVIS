import os, re, json, time, pickle, joblib, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from trainer_v2_fidone import TTVAETrainer
from model_v2 import TTVAE

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# ============================================================
# Arguments
# ============================================================
def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ("yes", "true", "t", "1", "y"): return True
    if v.lower() in ("no", "false", "f", "0", "n"): return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--config_path", type=str, default="ttvae_config.json")

parser.add_argument("--size_of_trees", type=int, default=20000)
parser.add_argument("--patience", type=int, default=1)
parser.add_argument("--min_delta", type=float, default=0.1)
parser.add_argument("--free_bits", type=float, default=0.0)
parser.add_argument("--abs_pos_enc", type=str, default="tree")

parser.add_argument("--reference_metric", type=str, default="accuracy",
                    choices=["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"])
parser.add_argument("--regularizations", type=float, nargs="+",
                    default=[0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01])

parser.add_argument("--bo_rand_start", type=int, default=1)
parser.add_argument("--bo_rand_end", type=int, default=1)
parser.add_argument("--bo_iterations", type=int, default=10)
parser.add_argument("--candidate_batch_size", type=int, default=50)
parser.add_argument("--n_grid", type=int, default=5000)
parser.add_argument("--sample_dist", type=str, default="normal", choices=["normal", "uniform"])
parser.add_argument("--grad_ascent", type=str2bool, default=True)
parser.add_argument("--ga_lr", type=float, default=0.001)
parser.add_argument("--ga_steps", type=int, default=10)

parser.add_argument("--mlp_epochs", type=int, default=100)
parser.add_argument("--mlp_batch_size", type=int, default=64)
parser.add_argument("--mlp_hidden_dim", type=int, default=128)
parser.add_argument("--mlp_lr", type=float, default=1e-3)
parser.add_argument("--mlp_weight_decay", type=float, default=1e-5)

parser.add_argument("--bootstrap_n_iter", type=int, default=200)
parser.add_argument("--bootstrap_seed", type=int, default=12345)
parser.add_argument("--results_root", type=str, default="trevis_paper_results")
parser.add_argument("--save_models", type=str2bool, default=True)
args = parser.parse_args()


PRECISION = 4
TREE_DIR = "linearized_trees_binarized"
TREE_PERF_DIR = "linearized_trees_binarized_performance"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Checkpoint
# ============================================================
def build_checkpoint_path(dataset, train_size):
    return (
        f"binarized_FULL_models_checkpoints_dataset_size20k_upd/{dataset}/threshold_tree_False/"
        "config_d_model_encoder_120_num_layers_encoder_2_num_heads_encoder_2_"
        "early_stopping_True_epochs_50_beta_start_0.0_beta_epoch_10_beta_eval_1.0_"
        f"patience_1_min_delta_0.1_train_size_{train_size}.pt"
    )


def parse_checkpoint_name(path):
    filename = os.path.basename(path).replace(".pt", "")
    patterns = {
        "d_model_encoder": r"d_model_encoder_(\d+)",
        "d_model_decoder": r"d_model_decoder_(\d+)",
        "num_layers_encoder": r"num_layers_encoder_(\d+)",
        "num_layers_decoder": r"num_layers_decoder_(\d+)",
        "num_heads_encoder": r"num_heads_encoder_(\d+)",
        "num_heads_decoder": r"num_heads_decoder_(\d+)",
        "epochs": r"epochs_(\d+)",
        "beta_start": r"beta_start_([\d.]+)",
        "beta_epoch": r"beta_epoch_(\d+)",
        "beta_eval": r"beta_eval_([\d.]+)",
        "patience": r"patience_(\d+)",
        "min_delta": r"min_delta_([\d.]+)",
        "train_size": r"train_size_(\d+)",
    }

    config = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, filename)
        if match:
            value = match.group(1)
            config[key] = float(value) if "." in value else int(value)

    for enc, dec in [
        ("d_model_encoder", "d_model_decoder"),
        ("num_layers_encoder", "num_layers_decoder"),
        ("num_heads_encoder", "num_heads_decoder"),
    ]:
        if enc in config and dec not in config: config[dec] = config[enc]

    return config


def get_ttvae_training_time(checkpoint):
    for key in ["training_time_seconds", "train_time_seconds", "training_time",
                "train_time", "total_training_time", "total_train_time"]:
        if key in checkpoint:
            try: return float(checkpoint[key])
            except (TypeError, ValueError): pass
    return np.nan


# ============================================================
# Data / latent helpers
# ============================================================
def split_tree_sequences(sequences, train_size):
    train, test = train_test_split(sequences, test_size=20000, random_state=42)
    train, val = train_test_split(train, test_size=10000, random_state=42)
    train, early_stop = train_test_split(train, test_size=10000, random_state=42)
    return train[:train_size], test, val, early_stop


def extract_latents(model, trainer, dataset, batch_size=128):
    model.eval()
    latents = []

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = trainer.collate_fn(dataset[start:start + batch_size])
            batch = {k: v.to(model.device) if not isinstance(v, list) else v
                     for k, v in batch.items()}

            _, mu, _, _ = model.encode(batch["src"], batch["src_abs_encs"], batch["src_rel_encs"])
            latents.append(mu.detach().cpu().numpy())

    return np.concatenate(latents, axis=0)


def decode_latent(trainer, z):
    model = trainer.model
    model.eval()

    z = torch.as_tensor(z, dtype=torch.float32, device=model.device).detach()
    if z.ndim == 1: z = z.unsqueeze(0)

    with torch.no_grad():
        return model.generate(z=z, use_cache=False, do_sample=False)


# ============================================================
# Surrogate g
# ============================================================
class MLPPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, lr=1e-3, weight_decay=1e-5, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def fit(self, X, y, epochs, batch_size):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            self.train()
            for xb, yb in loader:
                self.optimizer.zero_grad()
                loss = self.loss_fn(self(xb), yb)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        self.eval()
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self(X).detach().cpu().numpy().reshape(-1)

    def metrics(self, X, y):
        pred, y = self.predict(X), np.asarray(y).reshape(-1)
        mse = np.mean((pred - y) ** 2)
        pearson = np.nan if np.std(pred) == 0 or np.std(y) == 0 else np.corrcoef(pred, y)[0, 1]
        return {"mse": float(mse), "rmse": float(np.sqrt(mse)), "pearson": float(pearson)}


# ============================================================
# Metrics / objective
# ============================================================
def reference_metric(y_true, y_pred, metric):
    if metric == "accuracy": return accuracy_score(y_true, y_pred)
    if metric == "balanced_accuracy": return balanced_accuracy_score(y_true, y_pred)
    if metric == "f1_macro": return f1_score(y_true, y_pred, average="macro", zero_division=0)
    if metric == "f1_weighted": return f1_score(y_true, y_pred, average="weighted", zero_division=0)
    raise ValueError(metric)


def tree_leaves(clf):
    if hasattr(clf, "get_n_leaves"): return int(clf.get_n_leaves())
    if hasattr(clf, "n_leaves"): return int(clf.n_leaves)
    raise AttributeError("Cannot determine number of leaves.")


def tree_depth(clf):
    if hasattr(clf, "get_depth"): return int(clf.get_depth())
    if hasattr(clf, "max_depth"): return int(clf.max_depth)
    return np.nan


def regularized_objective(metric_value, n_leaves, lambda_reg):
    return 1.0 - float(metric_value) + float(lambda_reg) * int(n_leaves)


def effective_lambda(requested_lambda, train_sample_size):
    return min(float(requested_lambda), 1.0 / float(train_sample_size))


def training_metric_from_perf(perf, metric):
    if metric == "accuracy": return perf["train_accuracy"]
    if metric == "balanced_accuracy":
        return perf.get("train_balanced_accuracy", perf["train_accuracy"])
    if metric == "f1_macro": return perf["train_f1_macro"]
    if metric == "f1_weighted":
        return perf.get("train_f1_weighted", perf["train_f1_macro"])
    raise ValueError(metric)


def bootstrap_weighted_f1(y_true, y_pred, n_iter=200, seed=12345):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng, n = np.random.RandomState(seed), len(y_true)
    values = np.empty(n_iter)

    for i in range(n_iter):
        idx = rng.randint(0, n, size=n)
        values[i] = f1_score(y_true[idx], y_pred[idx], average="weighted", zero_division=0)

    return {
        "test_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "test_f1_weighted_bootstrap_mean": float(values.mean()),
        "test_f1_weighted_bootstrap_std": float(values.std()),
        "test_f1_weighted_bootstrap_ci_low": float(np.percentile(values, 2.5)),
        "test_f1_weighted_bootstrap_ci_high": float(np.percentile(values, 97.5)),
    }


# ============================================================
# CART
# ============================================================
def tune_cart(X_train, y_train, X_val, y_val, X_test, y_test, lambda_grid):
    start = time.time()
    best_clf, best_val_f1, best_params = None, -np.inf, None
    n_train = len(X_train)

    for lam in lambda_grid:
        min_leaf = max(1, int(np.ceil(lam * n_train)))

        for alpha in [0.0, 0.01, 0.1]:
            for depth in [2, 3, 4, 5]:
                clf = DecisionTreeClassifier(
                    max_depth=depth, ccp_alpha=alpha,
                    min_samples_leaf=min_leaf, random_state=42
                )
                clf.fit(X_train, y_train)
                val_f1 = f1_score(y_val, clf.predict(X_val), average="weighted", zero_division=0)

                if val_f1 > best_val_f1:
                    best_clf, best_val_f1 = clf, val_f1
                    best_params = {
                        "cart_lambda_support": lam,
                        "cart_min_samples_leaf": min_leaf,
                        "cart_ccp_alpha": alpha,
                        "cart_max_depth": depth,
                    }

    elapsed = time.time() - start
    metrics = bootstrap_weighted_f1(
        y_test, best_clf.predict(X_test),
        args.bootstrap_n_iter, args.bootstrap_seed
    )

    result = {
        "cart_val_f1_weighted": best_val_f1,
        "cart_test_f1_weighted": metrics["test_f1_weighted"],
        "cart_test_f1_weighted_bootstrap_mean": metrics["test_f1_weighted_bootstrap_mean"],
        "cart_test_f1_weighted_bootstrap_std": metrics["test_f1_weighted_bootstrap_std"],
        "cart_test_f1_weighted_bootstrap_ci_low": metrics["test_f1_weighted_bootstrap_ci_low"],
        "cart_test_f1_weighted_bootstrap_ci_high": metrics["test_f1_weighted_bootstrap_ci_high"],
        "cart_num_leaves": tree_leaves(best_clf),
        "cart_depth": tree_depth(best_clf),
        "cart_training_time_seconds": elapsed,
        **best_params,
    }
    return best_clf, result


def save_result_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = pd.DataFrame([row])

    if os.path.exists(path):
        old = pd.read_csv(path)

        required = ["dataset", "requested_lambda", "effective_lambda",
                    "rand_idx", "size_of_trees", "n_grid"]

        if all(c in old.columns for c in required):
            same = (
                (old["dataset"] == row["dataset"])
                & np.isclose(old["requested_lambda"], row["requested_lambda"])
                & np.isclose(old["effective_lambda"], row["effective_lambda"])
                & (old["rand_idx"] == row["rand_idx"])
                & (old["size_of_trees"] == row["size_of_trees"])
                & (old["n_grid"] == row["n_grid"])
            )
            old = old.loc[~same]

        new = pd.concat([old, new], ignore_index=True)

    new.to_csv(path, index=False)


# ============================================================
# Load data
# ============================================================
dataset = args.dataset_name
data = pd.read_csv(f"TITAN/data/data_splitted_binarized/{dataset}.csv")

train = data[data.split == "train"].drop(columns="split")
val = data[data.split == "val"].drop(columns="split")
test = data[data.split == "test"].drop(columns="split")

X_train, y_train = train.drop(columns="target").round(PRECISION), train["target"]
X_val, y_val = val.drop(columns="target").round(PRECISION), val["target"]
X_test, y_test = test.drop(columns="target").round(PRECISION), test["target"]
n_train_samples = len(X_train)

trees_seq = joblib.load(f"{TREE_DIR}/{dataset}.joblib")
trees_seq_perf = joblib.load(f"{TREE_PERF_DIR}/{dataset}.joblib")

train_seqs, test_seqs, val_seqs, es_seqs = split_tree_sequences(
    trees_seq, args.size_of_trees
)
train_seqs_perf, _, _, _ = split_tree_sequences(
    trees_seq_perf, args.size_of_trees
)

with open(f"{TREE_DIR}/te_encoders_binarized/{dataset}.pkl", "rb") as f:
    te = pickle.load(f)


# ============================================================
# TTVAE
# ============================================================
checkpoint_path = build_checkpoint_path(dataset, args.size_of_trees)
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found:\n{checkpoint_path}")

with open(args.config_path) as f:
    config = json.load(f)

config.update(parse_checkpoint_name(checkpoint_path))
config.update({
    "dataset": dataset,
    "patience": args.patience,
    "min_delta": args.min_delta,
    "free_bits": args.free_bits,
    "abs_pos_enc": args.abs_pos_enc,
    "num_scales": 10,
    "start_early_stop": 10,
})

model = TTVAE(
    tree_encoder=te,
    max_depth=config["max_depth"],
    d_model_encoder=config["d_model_encoder"],
    d_model_decoder=config["d_model_decoder"],
    num_heads_encoder=config["num_heads_encoder"],
    num_heads_decoder=config["num_heads_decoder"],
    num_layers_encoder=config["num_layers_encoder"],
    num_layers_decoder=config["num_layers_decoder"],
    d_ff_encoder=config["d_ff_encoder"],
    d_ff_decoder=config["d_ff_decoder"],
    dropout_encoder=config["dropout_encoder"],
    dropout_decoder=config["dropout_decoder"],
    abs_pos_enc=config["abs_pos_enc"],
    rel_pos_enc=config["rel_pos_enc"],
    latent_dim=config["latent_dim"],
    max_len=config["max_len"],
    num_scales=config["num_scales"],
    weight_tying=False,
    device="cuda",
)

trainer = TTVAETrainer(
    model=model,
    train_trees=train_seqs,
    val_trees=val_seqs,
    es_trees=es_seqs,
    batch_size=config["batch_size"],
    pad_token_id=te.token_to_id["<PAD>"],
    cls_token_id=te.token_to_id["<CLS>"],
    bos_token_id=te.token_to_id["<BOS>"],
    eos_token_id=te.token_to_id["<EOS>"],
    unk_token_id=te.token_to_id["<UNK>"],
    shuffle=True,
    seed=42,
)

checkpoint = torch.load(checkpoint_path, map_location=device)
trainer.model.load_state_dict(checkpoint["model_state_dict"])
trainer.model.to(device).eval()

ttvae_training_time = get_ttvae_training_time(checkpoint)
if np.isnan(ttvae_training_time):
    print("WARNING: TTVAE training time not stored in checkpoint.")


# ============================================================
# Output + latent representations
# ============================================================
results_dir = os.path.join(
    args.results_root, dataset,
    f"size_{args.size_of_trees}",
    f"grid_{args.n_grid}"
)
models_dir = os.path.join(results_dir, "models")
results_csv = os.path.join(results_dir, "results.csv")

os.makedirs(results_dir, exist_ok=True)
if args.save_models: os.makedirs(models_dir, exist_ok=True)

print("Extracting latent representations...")
X_surrogate = np.asarray(
    extract_latents(trainer.model, trainer, [x[0] for x in train_seqs_perf]),
    dtype=np.float64
)
print("Latent dataset:", X_surrogate.shape)


# ============================================================
# CART
# ============================================================
print("\nTuning CART...")
cart_clf, cart_results = tune_cart(
    X_train, y_train, X_val, y_val, X_test, y_test, args.regularizations
)

print("CART test bootstrap weighted F1:",
      cart_results["cart_test_f1_weighted_bootstrap_mean"])
print("CART leaves:", cart_results["cart_num_leaves"])

if args.save_models:
    with open(os.path.join(models_dir, "cart.pkl"), "wb") as f:
        pickle.dump(cart_clf, f)


# ============================================================
# TREVIS
# ============================================================
for requested_lambda in args.regularizations:
    lambda_reg = effective_lambda(requested_lambda, n_train_samples)

    print("\n" + "=" * 70)
    print(f"Dataset: {dataset} | requested λ: {requested_lambda} | effective λ: {lambda_reg}")
    print("=" * 70)

    # Surrogate targets
    y_surrogate = np.asarray([
        -regularized_objective(
            training_metric_from_perf(perf, args.reference_metric),
            int(perf["n_leaves"]),
            lambda_reg
        )
        for _, perf in train_seqs_perf
    ], dtype=np.float64)

    X_g_train, X_g_test, y_g_train, y_g_test = train_test_split(
        X_surrogate, y_surrogate, test_size=0.10, random_state=42
    )

    # Train surrogate
    torch.manual_seed(42)
    np.random.seed(42)

    surrogate = MLPPredictor(
        X_surrogate.shape[1],
        args.mlp_hidden_dim,
        args.mlp_lr,
        args.mlp_weight_decay,
        trainer.model.device,
    )

    start = time.time()
    surrogate.fit(X_g_train, y_g_train, args.mlp_epochs, args.mlp_batch_size)
    g_training_time = time.time() - start
    g_metrics = surrogate.metrics(X_g_test, y_g_test)

    print(f"g Pearson={g_metrics['pearson']:.4f} | RMSE={g_metrics['rmse']:.4f}")

    # Random restarts
    for rand_idx in range(args.bo_rand_start, args.bo_rand_end + 1):
        torch.manual_seed(rand_idx)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(rand_idx)
        np.random.seed(rand_idx)

        X_search_population = X_g_train.copy()
        best_tree, best_score, best_iteration = None, -np.inf, None
        valid_candidates_total = 0
        search_start = time.time()

        for iteration in range(args.bo_iterations):
            # Sample grid
            if args.sample_dist == "normal":
                grid = (
                    X_search_population.mean(0)
                    + np.random.randn(args.n_grid, X_search_population.shape[1])
                    * X_search_population.std(0)
                )
            else:
                low, high = X_search_population.min(0), X_search_population.max(0)
                grid = low + np.random.rand(args.n_grid, X_search_population.shape[1]) * (high - low)

            grid_t = torch.tensor(grid, dtype=torch.float32, device=surrogate.device)

            # Gradient ascent
            if args.grad_ascent:
                grid_t = grid_t.detach().requires_grad_(True)
                for _ in range(args.ga_steps):
                    gradients = torch.autograd.grad(surrogate(grid_t).sum(), grid_t)[0]
                    grid_t = (grid_t + args.ga_lr * gradients).detach().requires_grad_(True)

            # Best surrogate candidates
            with torch.no_grad():
                predicted_scores = surrogate(grid_t).detach().cpu().numpy().reshape(-1)

            selected = np.argsort(-predicted_scores)[:args.candidate_batch_size]
            next_latents = grid_t[selected].detach().cpu().numpy()
            valid_features = []

            # Decode and evaluate
            for latent in next_latents:
                try:
                    seq = decode_latent(trainer, latent).cpu().detach().numpy()
                    clf = trainer.model.decoder.tree_encoder.decode_tree(seq.tolist()[0])

                    train_pred = clf.predict(X_train)
                    metric = reference_metric(y_train, train_pred, args.reference_metric)
                    score = -regularized_objective(metric, tree_leaves(clf), lambda_reg)
                except Exception:
                    continue

                valid_candidates_total += 1
                valid_features.append(latent)

                if score > best_score:
                    best_tree, best_score, best_iteration = clf, score, iteration

            if valid_features:
                X_search_population = np.concatenate(
                    [X_search_population, np.asarray(valid_features)], axis=0
                )

            print(
                f"Iteration {iteration + 1:02d}/{args.bo_iterations} | "
                f"valid={len(valid_features)} | best={best_score:.6f}"
            )

        search_time = time.time() - search_start

        if best_tree is None:
            print("No valid tree found; skipping run.")
            continue

        # Final pruned model
        pruned_tree = best_tree.prune()
        val_pred, test_pred = pruned_tree.predict(X_val), pruned_tree.predict(X_test)

        val_f1 = f1_score(y_val, val_pred, average="weighted", zero_division=0)
        boot = bootstrap_weighted_f1(
            y_test, test_pred, args.bootstrap_n_iter, args.bootstrap_seed + rand_idx
        )

        n_leaves, depth = tree_leaves(pruned_tree), tree_depth(pruned_tree)
        val_ref = reference_metric(y_val, val_pred, args.reference_metric)
        test_ref = reference_metric(y_test, test_pred, args.reference_metric)

        val_obj = regularized_objective(val_ref, n_leaves, lambda_reg)
        test_obj = regularized_objective(test_ref, n_leaves, lambda_reg)

        total_time = g_training_time + search_time
        if not np.isnan(ttvae_training_time):
            total_time += ttvae_training_time

        result = {
            "dataset": dataset,
            "size_of_trees": args.size_of_trees,
            "n_grid": args.n_grid,
            "reference_metric": args.reference_metric,
            "requested_lambda": requested_lambda,
            "effective_lambda": lambda_reg,
            "rand_idx": rand_idx,
            "best_iteration": best_iteration,
            "valid_candidates_total": valid_candidates_total,

            "trevis_val_f1_weighted": float(val_f1),
            "trevis_test_f1_weighted": boot["test_f1_weighted"],
            "trevis_test_f1_weighted_bootstrap_mean": boot["test_f1_weighted_bootstrap_mean"],
            "trevis_test_f1_weighted_bootstrap_std": boot["test_f1_weighted_bootstrap_std"],
            "trevis_test_f1_weighted_bootstrap_ci_low": boot["test_f1_weighted_bootstrap_ci_low"],
            "trevis_test_f1_weighted_bootstrap_ci_high": boot["test_f1_weighted_bootstrap_ci_high"],

            "trevis_num_leaves": n_leaves,
            "trevis_depth": depth,
            "trevis_val_reference_metric": float(val_ref),
            "trevis_test_reference_metric": float(test_ref),
            "trevis_val_objective": float(val_obj),
            "trevis_test_objective": float(test_obj),
            "best_train_negative_objective": float(best_score),

            "surrogate_test_pearson": g_metrics["pearson"],
            "surrogate_test_rmse": g_metrics["rmse"],

            "ttvae_training_time_seconds": ttvae_training_time,
            "g_training_time_seconds": g_training_time,
            "gradient_search_time_seconds": search_time,
            "trevis_total_time_seconds": total_time,

            **cart_results,
        }

        if args.save_models:
            model_name = (
                f"trevis_lambda_{requested_lambda}_"
                f"effective_{lambda_reg}_rand_{rand_idx}.pkl"
            )
            with open(os.path.join(models_dir, model_name), "wb") as f:
                pickle.dump(pruned_tree, f)

        save_result_row(results_csv, result)

        print(
            f"\nRESULT | F1={boot['test_f1_weighted_bootstrap_mean']:.4f} | "
            f"leaves={n_leaves} | Pearson={g_metrics['pearson']:.4f} | "
            f"RMSE={g_metrics['rmse']:.4f} | g={g_training_time:.2f}s | "
            f"search={search_time:.2f}s"
        )


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETED")
print("Results:", results_csv)

if os.path.exists(results_csv):
    results = pd.read_csv(results_csv)

    columns = [
        "dataset", "requested_lambda", "effective_lambda", "rand_idx",
        "trevis_test_f1_weighted_bootstrap_mean", "trevis_num_leaves",
        "surrogate_test_pearson", "surrogate_test_rmse",
        "ttvae_training_time_seconds", "g_training_time_seconds",
        "gradient_search_time_seconds", "trevis_total_time_seconds",
        "cart_test_f1_weighted_bootstrap_mean", "cart_num_leaves",
        "cart_training_time_seconds",
    ]

    print(results[[c for c in columns if c in results.columns]].to_string(index=False))