import os, re, json, time, pickle, shutil, joblib, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, export_text
from trainer_v2_fidone import TTVAETrainer
from model_v2 import TTVAE

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
pd.set_option("display.max_columns", None)


# ============================================================
# ARGUMENTS
# ============================================================
def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ("yes", "true", "t", "1", "y"): return True
    if v.lower() in ("no", "false", "f", "0", "n"): return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

parser = argparse.ArgumentParser()

for name, default, dtype in [
    ("config_path", "ttvae_config.json", str), ("patience", 1, int),
    ("abs_pos_enc", "tree", str), ("size_of_trees", 20000, int),
    ("beta_epoch", 10, int), ("start_early_stop", 0, int),
    ("dropout", 0.0, float), ("learning_rate", 0.001, float),
    ("early_stopping", True, str2bool)
]:
    parser.add_argument(f"--{name}", type=dtype, default=default)

parser.add_argument("--dataset_name", required=True)
parser.add_argument("--min_delta", type=float, default=0.1)
parser.add_argument("--free_bits", type=float, default=0.0)
parser.add_argument("--reference_metric", default="f1_weighted", choices=["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"])
parser.add_argument("--regularizations", type=float, nargs="+", default=[0.0, 0.001, 0.005, 0.0001, 0.0005, 0.01])

parser.add_argument("--bo_rand_start", type=int, default=1)
parser.add_argument("--bo_rand_end", type=int, default=1)
parser.add_argument("--bo_iterations", type=int, default=10)
parser.add_argument("--candidate_batch_size", type=int, default=50)
parser.add_argument("--n_grid", type=int, default=5000)
parser.add_argument("--sample_dist", default="standard_normal", choices=["standard_normal", "normal", "uniform"])
parser.add_argument("--grad_ascent", type=str2bool, default=True)
parser.add_argument("--ga_lr", type=float, default=0.001)
parser.add_argument("--ga_steps", type=int, default=10)

parser.add_argument("--mlp_epochs", type=int, default=100)
parser.add_argument("--mlp_batch_size", type=int, default=64)
parser.add_argument("--mlp_hidden_dim", type=int, default=128)
parser.add_argument("--mlp_lr", type=float, default=1e-3)
parser.add_argument("--mlp_weight_decay", type=float, default=1e-5)

parser.add_argument("--results_root", default="trevis_paper_results")
parser.add_argument("--save_predictions", type=str2bool, default=True)
parser.add_argument("--save_tree_text", type=str2bool, default=True)
parser.add_argument("--bootstrap_n_iter", type=int, default=200)
parser.add_argument("--bootstrap_seed", type=int, default=12345)
parser.add_argument("--bootstrap_results_root", default=None)
args = parser.parse_args()

DATASET, SIZE, N_GRID, METRIC = args.dataset_name, int(args.size_of_trees), int(args.n_grid), args.reference_metric
PRECISION, TREE_DIR, PERF_DIR = 4, "linearized_trees_binarized", "linearized_trees_binarized_performance"
SIZE_TAG, GRID_TAG = f"size_of_trees_{SIZE}", f"n_grid_{N_GRID}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# HELPERS
# ============================================================
def lambda_tag(x): return str(x).replace("-", "m").replace(".", "p")

def run_tag(req, eff, rand):
    return f"{SIZE_TAG}_{GRID_TAG}_lambda_requested_{lambda_tag(req)}_effective_{lambda_tag(eff)}_rand_idx_{rand}"

def checkpoint_path():
    return (
        f"binarized_FULL_models_checkpoints_dataset_size20k_upd/{DATASET}/threshold_tree_False/"
        "config_d_model_encoder_120_num_layers_encoder_2_num_heads_encoder_2_"
        "early_stopping_True_epochs_50_beta_start_0.0_beta_epoch_10_beta_eval_1.0_"
        f"patience_1_min_delta_0.1_train_size_{SIZE}.pt"
    )

def metadata(path): return {"size_of_trees": SIZE, "checkpoint_train_size": SIZE, "n_grid": N_GRID, "checkpoint_path": path}

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2, default=str)

def flatten_csv(d):
    return {k: json.dumps(v, default=str) if isinstance(v, (dict, list, tuple, np.ndarray)) else v.item() if isinstance(v, (np.integer, np.floating)) else v for k, v in d.items()}

def load_pickle(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f: return pickle.load(f)

def copy_if_exists(src, dst):
    if not os.path.exists(src): return False
    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(src, dst); return True

def safe_tree_text(clf, feature_names=None):
    try: return export_text(clf, feature_names=feature_names) if hasattr(clf, "tree_") else str(clf)
    except Exception as e: return f"Could not export tree text: {e}\n\n{clf}"


# ============================================================
# CHECKPOINT
# ============================================================
def parse_checkpoint(path):
    name = os.path.basename(path).replace(".pt", "")
    patterns = {
        "d_model_encoder": r"d_model_encoder_(\d+)", "d_model_decoder": r"d_model_decoder_(\d+)",
        "num_layers_encoder": r"num_layers_encoder_(\d+)", "num_layers_decoder": r"num_layers_decoder_(\d+)",
        "num_heads_encoder": r"num_heads_encoder_(\d+)", "num_heads_decoder": r"num_heads_decoder_(\d+)",
        "early_stopping": r"early_stopping_(True|False)", "epochs": r"epochs_(\d+)",
        "beta_start": r"beta_start_([\d.]+)", "beta_epoch": r"beta_epoch_(\d+)",
        "beta_eval": r"beta_eval_([\d.]+)", "patience": r"patience_(\d+)",
        "min_delta": r"min_delta_([\d.]+)", "train_size": r"train_size_(\d+)"
    }

    cfg = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, name)
        if not m: continue
        v = m.group(1)
        cfg[key] = v == "True" if v in ("True", "False") else float(v) if "." in v else int(v)

    for enc, dec in [("d_model_encoder", "d_model_decoder"), ("num_layers_encoder", "num_layers_decoder"), ("num_heads_encoder", "num_heads_decoder")]:
        cfg.setdefault(dec, cfg.get(enc))
    return cfg


# ============================================================
# DATA / LATENTS
# ============================================================
def split_seqs(seqs):
    train, test = train_test_split(seqs, test_size=20000, random_state=42)
    train, val = train_test_split(train, test_size=10000, random_state=42)
    train, es = train_test_split(train, test_size=10000, random_state=42)
    return train[:SIZE], test, val, es

def extract_latents(model, trainer, dataset, batch_size=128):
    model.eval(); mus = []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = trainer.collate_fn(dataset[i:i + batch_size])
            batch = {k: v.to(model.device) if not isinstance(v, list) else v for k, v in batch.items()}
            _, mu, _, _ = model.encode(batch["src"], batch["src_abs_encs"], batch["src_rel_encs"])
            mus.append(mu.cpu().numpy())
    return np.concatenate(mus)

def decode_latent(trainer, z):
    z = torch.as_tensor(z, dtype=torch.float32, device=trainer.model.device)
    if z.ndim == 1: z = z.unsqueeze(0)
    with torch.no_grad(): return trainer.model.generate(z=z, use_cache=False, do_sample=False)


# ============================================================
# MLP
# ============================================================
class MLPPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.device = trainer.model.device
        self.net = nn.Sequential(nn.Linear(input_dim, args.mlp_hidden_dim), nn.Tanh(), nn.Linear(args.mlp_hidden_dim, 1))
        self.loss_fn = nn.MSELoss(reduction="sum")
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)

    def forward(self, x): return self.net(x)

    def fit(self, X, y, X_test=None, y_test=None):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)
        loader = DataLoader(TensorDataset(X, y), batch_size=args.mlp_batch_size, shuffle=True)

        for epoch in range(args.mlp_epochs):
            self.train()
            for xb, yb in loader:
                self.optimizer.zero_grad()
                loss = self.loss_fn(self(xb), yb) / xb.shape[0]
                loss.backward(); self.optimizer.step()

            if epoch % 50 == 0 or epoch == args.mlp_epochs - 1:
                tr = self.metrics(X, y)
                msg = f"Epoch {epoch:04d} | train MSE={tr['mse']:.4f} RMSE={tr['rmse']:.4f} Pearson={tr['pearson']:.4f}"
                if X_test is not None:
                    te = self.metrics(X_test, y_test)
                    msg += f" | test MSE={te['mse']:.4f} RMSE={te['rmse']:.4f} Pearson={te['pearson']:.4f}"
                print(msg)

    def predict(self, X):
        self.eval(); X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad(): return self(X).cpu().numpy()

    def metrics(self, X, y):
        self.eval()
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)

        with torch.no_grad():
            pred = self(X); mse = torch.mean((pred - y) ** 2)
            p, t = pred.flatten(), y.flatten()
            p, t = p - p.mean(), t - t.mean()
            pearson = torch.sum(p * t) / (torch.sqrt(torch.sum(p ** 2)) * torch.sqrt(torch.sum(t ** 2)) + 1e-8)

        return {"mse": mse.item(), "rmse": torch.sqrt(mse).item(), "pearson": pearson.item()}


# ============================================================
# METRICS / OBJECTIVE
# ============================================================
def reference_metric(y_true, y_pred):
    if METRIC == "accuracy": return accuracy_score(y_true, y_pred)
    if METRIC == "balanced_accuracy": return balanced_accuracy_score(y_true, y_pred)
    if METRIC == "f1_macro": return f1_score(y_true, y_pred, average="macro", zero_division=0)
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)

def tree_leaves(clf):
    if hasattr(clf, "get_n_leaves"): return int(clf.get_n_leaves())
    if hasattr(clf, "n_leaves"): return int(clf.n_leaves)
    raise AttributeError("Cannot determine number of leaves.")

def tree_depth(clf):
    if hasattr(clf, "get_depth"): return int(clf.get_depth())
    if hasattr(clf, "max_depth"): return int(clf.max_depth)
    return None

def tree_nodes(clf):
    if hasattr(clf, "get_n_nodes"): return int(clf.get_n_nodes())
    if hasattr(clf, "tree_"): return int(clf.tree_.node_count)
    return None

def objective(metric_value, leaves, lam): return 1.0 - float(metric_value) + float(lam) * int(leaves)

def effective_lambda(requested):
    cap = 1.0 / len(X_train); effective = min(float(requested), cap)
    return effective, cap, effective < requested

def metric_from_perf(perf):
    if METRIC == "accuracy": return perf["train_accuracy"]
    if METRIC == "balanced_accuracy": return perf.get("train_balanced_accuracy", perf["train_accuracy"])
    if METRIC == "f1_macro": return perf["train_f1_macro"]
    return perf.get("train_f1_weighted", perf["train_f1_macro"])

def classification_metrics(y_true, y_pred, prefix):
    labels = np.unique(np.r_[y_true, y_pred])
    out = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}_mcc": matthews_corrcoef(y_true, y_pred),
        f"{prefix}_cohen_kappa": cohen_kappa_score(y_true, y_pred),
        f"{prefix}_labels": labels.tolist(),
        f"{prefix}_confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        f"{prefix}_classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    }

    for avg in ("macro", "micro", "weighted"):
        out[f"{prefix}_f1_{avg}"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
        out[f"{prefix}_precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        out[f"{prefix}_recall_{avg}"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
    return out


# ============================================================
# BOOTSTRAP
# ============================================================
def bootstrap_metrics(y_true, y_pred, prefix):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.RandomState(args.bootstrap_seed)
    acc, f1w = [], []

    for _ in range(args.bootstrap_n_iter):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        acc.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1w.append(f1_score(y_true[idx], y_pred[idx], average="weighted", zero_division=0))

    acc, f1w = np.asarray(acc), np.asarray(f1w)
    out = {}

    for name, values, original in [
        ("accuracy", acc, accuracy_score(y_true, y_pred)),
        ("f1_weighted", f1w, f1_score(y_true, y_pred, average="weighted", zero_division=0))
    ]:
        out.update({
            f"{prefix}_bootstrap_{name}_original": float(original),
            f"{prefix}_bootstrap_{name}_mean": float(values.mean()),
            f"{prefix}_bootstrap_{name}_ci_lower": float(np.percentile(values, 2.5)),
            f"{prefix}_bootstrap_{name}_ci_upper": float(np.percentile(values, 97.5))
        })

    out.update({f"{prefix}_bootstrap_n_iter": args.bootstrap_n_iter, f"{prefix}_bootstrap_seed": args.bootstrap_seed})
    return out

def add_bootstrap(row, trevis, baseline):
    for prefix, clf in [("bayesian_pruned", trevis), ("baseline", baseline)]:
        for split, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
            row.update(bootstrap_metrics(y, clf.predict(X), f"{prefix}_{split}"))

    row.update({"bootstrap_status": "ok", "bootstrap_n_iter": args.bootstrap_n_iter, "bootstrap_seed": args.bootstrap_seed})
    return row


# ============================================================
# EVALUATION
# ============================================================
DATASETS = {}

def evaluate_classifier(clf, prefix, lam):
    pred = {s: clf.predict(X) for s, (X, _) in DATASETS.items()}
    leaves = tree_leaves(clf)
    refs = {s: reference_metric(DATASETS[s][1], pred[s]) for s in DATASETS}
    metrics = {}

    for s in DATASETS: metrics.update(classification_metrics(DATASETS[s][1], pred[s], f"{prefix}_{s}"))

    return {
        "pred": pred,
        "ref": refs,
        "obj": {s: objective(refs[s], leaves, lam) for s in DATASETS},
        "depth": tree_depth(clf),
        "leaves": leaves,
        "nodes": tree_nodes(clf),
        "metrics": metrics
    }

def evaluation_to_row(trevis_eval, baseline_eval):
    row = {}

    for prefix, ev in [("bayesian_pruned", trevis_eval), ("baseline", baseline_eval)]:
        for split in ("train", "val", "test"):
            row[f"{prefix}_{split}_reference_metric"] = ev["ref"][split]
            row[f"{prefix}_{split}_regularized_objective"] = ev["obj"][split]

    row.update({
        "pruned_depth": trevis_eval["depth"], "pruned_n_leaves": trevis_eval["leaves"], "pruned_n_nodes": trevis_eval["nodes"],
        "baseline_depth": baseline_eval["depth"], "baseline_n_leaves": baseline_eval["leaves"], "baseline_n_nodes": baseline_eval["nodes"]
    })
    row.update(trevis_eval["metrics"]); row.update(baseline_eval["metrics"])
    return row


# ============================================================
# RESULT MATCHING
# ============================================================
RUN_COLUMNS = {"rand_idx", "requested_lambda_reg", "effective_lambda_reg", "reference_metric", "size_of_trees", "n_grid"}

def run_mask(df, rand, requested, effective):
    return (
        (df["rand_idx"].astype(int) == rand)
        & np.isclose(df["requested_lambda_reg"].astype(float), requested, atol=1e-12)
        & np.isclose(df["effective_lambda_reg"].astype(float), effective, atol=1e-12)
        & (df["reference_metric"] == METRIC)
        & (df["size_of_trees"].astype(int) == SIZE)
        & (df["n_grid"].astype(int) == N_GRID)
    )

def find_existing(df, rand, requested, effective):
    if df.empty or not RUN_COLUMNS.issubset(df.columns): return None
    m = run_mask(df, rand, requested, effective)
    return df.loc[m].iloc[-1].to_dict() if m.any() else None

def save_result(path, row, rand, requested, effective):
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if not old.empty and RUN_COLUMNS.issubset(old.columns): old = old.loc[~run_mask(old, rand, requested, effective)]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


# ============================================================
# DATA
# ============================================================
path = checkpoint_path()
if not os.path.exists(path): raise FileNotFoundError(f"Checkpoint not found: {path}")

data = pd.read_csv(f"TITAN/data/data_splitted_binarized/{DATASET}.csv")
splits = {s: data[data.split == s].drop(columns="split") for s in ("train", "val", "test")}

X_train, y_train = splits["train"].drop(columns="target").round(PRECISION), splits["train"]["target"]
X_val, y_val = splits["val"].drop(columns="target").round(PRECISION), splits["val"]["target"]
X_test, y_test = splits["test"].drop(columns="target").round(PRECISION), splits["test"]["target"]

DATASETS.update({"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)})
feature_names = list(X_train.columns)

trees = joblib.load(f"{TREE_DIR}/{DATASET}.joblib")
trees_perf = joblib.load(f"{PERF_DIR}/{DATASET}.joblib")
train_seqs, test_seqs, val_seqs, es_seqs = split_seqs(trees)
train_seqs_perf, _, _, _ = split_seqs(trees_perf)

with open(f"{TREE_DIR}/te_encoders_binarized/{DATASET}.pkl", "rb") as f: te = pickle.load(f)


# ============================================================
# TTVAE
# ============================================================
cfg = parse_checkpoint(path)

with open(args.config_path) as f: config = json.load(f)

config.update(cfg)
config.update({
    "dataset": DATASET, "patience": args.patience, "min_delta": args.min_delta,
    "free_bits": args.free_bits, "abs_pos_enc": args.abs_pos_enc,
    "num_scales": 10, "start_early_stop": 10,
    "size_of_trees": SIZE, "checkpoint_train_size": SIZE, "n_grid": N_GRID
})

model = TTVAE(
    tree_encoder=te, max_depth=config["max_depth"],
    d_model_encoder=config["d_model_encoder"], d_model_decoder=config["d_model_decoder"],
    num_heads_encoder=config["num_heads_encoder"], num_heads_decoder=config["num_heads_decoder"],
    num_layers_encoder=config["num_layers_encoder"], num_layers_decoder=config["num_layers_decoder"],
    d_ff_encoder=config["d_ff_encoder"], d_ff_decoder=config["d_ff_decoder"],
    dropout_encoder=config["dropout_encoder"], dropout_decoder=config["dropout_decoder"],
    abs_pos_enc=config["abs_pos_enc"], rel_pos_enc=config["rel_pos_enc"],
    latent_dim=config["latent_dim"], max_len=config["max_len"], num_scales=config["num_scales"],
    weight_tying=False, device="cuda"
)

trainer = TTVAETrainer(
    model=model, train_trees=train_seqs, val_trees=val_seqs, es_trees=es_seqs,
    batch_size=config["batch_size"], pad_token_id=te.token_to_id["<PAD>"],
    cls_token_id=te.token_to_id["<CLS>"], bos_token_id=te.token_to_id["<BOS>"],
    eos_token_id=te.token_to_id["<EOS>"], unk_token_id=te.token_to_id["<UNK>"],
    shuffle=True, seed=42
)

ckp = torch.load(path, map_location=DEVICE)
trainer.model.load_state_dict(ckp["model_state_dict"])
trainer.model.to(DEVICE).eval()

val_res = trainer.evaluate(split="val")
print(f"Dataset={DATASET} | metric={METRIC} | size={SIZE} | grid={N_GRID} | dist={args.sample_dist}")
print("TTVAE validation:", val_res)


# ============================================================
# OUTPUT
# ============================================================
relative = os.path.join(DATASET, SIZE_TAG, GRID_TAG, f"metric_{METRIC}")
source_dir = os.path.join(args.results_root, relative)
results_dir = os.path.join(args.bootstrap_results_root or f"{args.results_root}_bootstrap", relative)

dirs = {name: os.path.join(results_dir, name) for name in ("models", "predictions", "trees", "json")}
source_dirs = {name: os.path.join(source_dir, name) for name in ("models", "predictions", "trees", "json")}
for d in dirs.values(): os.makedirs(d, exist_ok=True)

source_csv = os.path.join(source_dir, "results.csv")
results_csv = os.path.join(results_dir, "results.csv")
run_config_path = os.path.join(results_dir, "run_config.json")
source_results = pd.read_csv(source_csv) if os.path.exists(source_csv) else pd.DataFrame()

save_json(run_config_path, {
    "args": vars(args), "checkpoint_path": path, "checkpoint_config": cfg,
    "base_config": config, "ttvae_val_res": val_res,
    "regularizer_sample_size": len(X_train),
    "regularizer_cap_1_over_sample_size": 1.0 / len(X_train),
    "regularizer_rule": "effective_lambda_reg = min(requested_lambda_reg, 1 / regularizer_sample_size)",
    **metadata(path), "sample_dist": args.sample_dist,
    "source_results_dir": source_dir, "bootstrap_results_dir": results_dir,
    "bootstrap_n_iter": args.bootstrap_n_iter, "bootstrap_seed": args.bootstrap_seed
})


# ============================================================
# LATENTS
# ============================================================
X_sgp = extract_latents(trainer.model, trainer, [x[0] for x in train_seqs_perf]).astype(np.float64)


# ============================================================
# EXPERIMENTS
# ============================================================
for requested_lambda in args.regularizations:
    lam, lambda_cap, lambda_adjusted = effective_lambda(requested_lambda)

    print(f"\n{'=' * 70}\nrequested λ={requested_lambda} | effective λ={lam} | cap={lambda_cap} | dist={args.sample_dist}\n{'=' * 70}")

    y_sgp = np.array([
        -objective(metric_from_perf(perf), perf["n_leaves"], lam)
        for _, perf in train_seqs_perf
    ]).reshape(-1, 1)

    source_rows = [
        {
            "idx": i, "dataset_name": DATASET, **metadata(path),
            "sample_dist": args.sample_dist, "reference_metric": METRIC,
            "reference_metric_value": metric_from_perf(perf), "n_leaves": int(perf["n_leaves"]),
            "requested_lambda_reg": requested_lambda, "effective_lambda_reg": lam,
            "lambda_reg_adjusted_due_to_sample_size": lambda_adjusted,
            "lambda_cap_1_over_sample_size": lambda_cap, "regularizer_sample_size": len(X_train),
            "regularized_objective": objective(metric_from_perf(perf), perf["n_leaves"], lam),
            "surrogate_target_negative_objective": -objective(metric_from_perf(perf), perf["n_leaves"], lam)
        }
        for i, (_, perf) in enumerate(train_seqs_perf)
    ]

    source_targets_path = os.path.join(
        dirs["json"],
        f"source_targets_{SIZE_TAG}_{GRID_TAG}_lambda_requested_{lambda_tag(requested_lambda)}_effective_{lambda_tag(lam)}.json"
    )
    save_json(source_targets_path, source_rows)

    np.random.seed(42)
    perm = np.random.choice(len(X_sgp), len(X_sgp), replace=False)
    cut = round(0.9 * len(X_sgp))

    X_base_train, X_base_test = X_sgp[perm][:cut], X_sgp[perm][cut:]
    y_base_train, y_base_test = y_sgp[perm][:cut], y_sgp[perm][cut:]

    mlp = MLPPredictor(X_sgp.shape[1])

    start = time.time()
    mlp.fit(X_base_train, y_base_train, X_base_test, y_base_test)
    mlp_train_time = time.time() - start
    mlp_raw = mlp.metrics(X_base_test, y_base_test)

    mlp_path = os.path.join(
        dirs["models"],
        f"mlp_predictor_{SIZE_TAG}_{GRID_TAG}_lambda_requested_{lambda_tag(requested_lambda)}_effective_{lambda_tag(lam)}.pt"
    )

    torch.save({
        "model_state_dict": mlp.state_dict(), "input_dim": X_sgp.shape[1],
        "hidden_dim": args.mlp_hidden_dim, "dataset_name": DATASET, **metadata(path),
        "sample_dist": args.sample_dist, "requested_lambda_reg": requested_lambda,
        "effective_lambda_reg": lam, "lambda_reg_adjusted_due_to_sample_size": lambda_adjusted,
        "lambda_cap_1_over_sample_size": lambda_cap, "regularizer_sample_size": len(X_train),
        "reference_metric": METRIC, "mlp_metrics_raw": mlp_raw
    }, mlp_path)

    for rand_idx in range(args.bo_rand_start, args.bo_rand_end + 1):
        tag = run_tag(requested_lambda, lam, rand_idx)

        paths = {
            "model": os.path.join(dirs["models"], f"best_clf_pruned_{tag}.pkl"),
            "unpruned": os.path.join(dirs["models"], f"best_clf_unpruned_{tag}.pkl"),
            "baseline": os.path.join(dirs["models"], f"baseline_decision_tree_{tag}.pkl")
        }
        source_paths = {
            "model": os.path.join(source_dirs["models"], f"best_clf_pruned_{tag}.pkl"),
            "unpruned": os.path.join(source_dirs["models"], f"best_clf_unpruned_{tag}.pkl"),
            "baseline": os.path.join(source_dirs["models"], f"baseline_decision_tree_{tag}.pkl")
        }

        old_row = find_existing(source_results, rand_idx, requested_lambda, lam)
        existing = {k: load_pickle(p) for k, p in paths.items()}

        if existing["model"] is None and os.path.exists(source_paths["model"]) and os.path.exists(source_paths["baseline"]):
            for k in paths:
                copy_if_exists(source_paths[k], paths[k]); existing[k] = load_pickle(paths[k])

            for split in ("train", "val", "test"):
                copy_if_exists(
                    os.path.join(source_dirs["predictions"], f"{split}_predictions_{tag}.csv"),
                    os.path.join(dirs["predictions"], f"{split}_predictions_{tag}.csv")
                )

            for name in ("best_clf_pruned", "best_clf_unpruned", "baseline_decision_tree"):
                copy_if_exists(
                    os.path.join(source_dirs["trees"], f"{name}_{tag}.txt"),
                    os.path.join(dirs["trees"], f"{name}_{tag}.txt")
                )

            copy_if_exists(
                os.path.join(source_dirs["json"], f"iteration_summaries_{tag}.json"),
                os.path.join(dirs["json"], f"iteration_summaries_{tag}.json")
            )

        if existing["model"] is not None and existing["baseline"] is not None:
            print("Reusing:", tag)

            if old_row is not None:
                row = old_row.copy()
            else:
                trevis_eval = evaluate_classifier(existing["model"], "bayesian_pruned", lam)
                baseline_eval = evaluate_classifier(existing["baseline"], "baseline", lam)

                row = {
                    "dataset_name": DATASET, "sample_dist": args.sample_dist,
                    "requested_lambda_reg": requested_lambda, "effective_lambda_reg": lam,
                    "lambda_reg_adjusted_due_to_sample_size": lambda_adjusted,
                    "lambda_cap_1_over_sample_size": lambda_cap, "regularizer_sample_size": len(X_train),
                    "lambda_reg": lam, "reference_metric": METRIC, "rand_idx": rand_idx,
                    **evaluation_to_row(trevis_eval, baseline_eval),
                    "unpruned_depth": tree_depth(existing["unpruned"]) if existing["unpruned"] else None,
                    "unpruned_n_leaves": tree_leaves(existing["unpruned"]) if existing["unpruned"] else None,
                    "unpruned_n_nodes": tree_nodes(existing["unpruned"]) if existing["unpruned"] else None
                }

            row.update({
                **metadata(path), "sample_dist": args.sample_dist,
                "model_path": paths["model"], "unpruned_model_path": paths["unpruned"],
                "baseline_model_path": paths["baseline"], "result_source": "reused_existing_model",
                "source_results_dir": source_dir, "bootstrap_results_dir": results_dir
            })

            row = flatten_csv(add_bootstrap(row, existing["model"], existing["baseline"]))
            save_result(results_csv, row, rand_idx, requested_lambda, lam)
            continue

        torch.manual_seed(rand_idx); np.random.seed(rand_idx)
        if torch.cuda.is_available(): torch.cuda.manual_seed(rand_idx); torch.cuda.manual_seed_all(rand_idx)

        X_search, y_search = X_base_train.copy(), y_base_train.copy()
        mean_y, std_y = y_search.mean(), y_search.std()
        if std_y == 0: raise ValueError("std_y_train_sgp is zero.")

        y_search = (y_search - mean_y) / std_y
        y_test_norm = (y_base_test - mean_y) / std_y

        best_clf, best_score, best_iteration, best_candidate = None, -np.inf, 0, {}
        iteration_times, summaries = [], []
        search_start = time.time()

        for iteration in range(args.bo_iterations):
            iter_start = time.time()
            train_metrics = mlp.metrics(X_search, y_search)
            test_metrics = mlp.metrics(X_base_test, y_test_norm)

            if args.sample_dist == "standard_normal":
                grid = np.random.randn(N_GRID, X_search.shape[1])
            elif args.sample_dist == "normal":
                grid = X_search.mean(0) + np.random.randn(N_GRID, X_search.shape[1]) * X_search.std(0)
            else:
                lo, hi = X_search.min(0), X_search.max(0)
                grid = lo + np.random.rand(N_GRID, X_search.shape[1]) * (hi - lo)

            grid_t = torch.tensor(grid, dtype=torch.float32, device=mlp.device)

            if args.grad_ascent:
                grid_t = grid_t.detach().requires_grad_(True)

                for _ in range(args.ga_steps):
                    grad = torch.autograd.grad(mlp(grid_t).sum(), grid_t)[0]
                    grid_t = (grid_t + args.ga_lr * grad).detach().requires_grad_(True)

            with torch.no_grad(): pred = mlp(grid_t).cpu().numpy().reshape(-1)

            idx = np.argsort(-pred)[:args.candidate_batch_size]
            next_inputs = grid_t[idx].detach().cpu().numpy() if args.grad_ascent else grid[idx]

            valid_features, valid_scores, candidates = [], [], []

            for i, latent in enumerate(next_inputs):
                try:
                    seq = decode_latent(trainer, latent).cpu().numpy()
                    clf = trainer.model.decoder.tree_encoder.decode_tree(seq.tolist()[0])
                    train_pred = clf.predict(X_train)
                    raw_metric = reference_metric(y_train, train_pred)
                    leaves = tree_leaves(clf)
                    raw_obj = objective(raw_metric, leaves, lam)
                    raw_score = -raw_obj
                    score = (raw_score - mean_y) / std_y

                    candidate = {
                        "dataset_name": DATASET, **metadata(path), "sample_dist": args.sample_dist,
                        "requested_lambda_reg": requested_lambda, "effective_lambda_reg": lam,
                        "lambda_reg_adjusted_due_to_sample_size": lambda_adjusted,
                        "lambda_cap_1_over_sample_size": lambda_cap, "regularizer_sample_size": len(X_train),
                        "reference_metric": METRIC, "rand_idx": rand_idx, "iteration": iteration, "candidate_idx": i,
                        "reference_metric_raw_train": raw_metric, "objective_raw": raw_obj,
                        "score_raw_negative_objective": raw_score, "score_normalized": score,
                        "n_leaves": leaves, "n_nodes": tree_nodes(clf), "depth": tree_depth(clf),
                        **classification_metrics(y_train, train_pred, "candidate_train")
                    }

                except Exception as e:
                    print(f"[{i}] candidate failed:", e)
                    continue

                valid_features.append(latent); valid_scores.append(score); candidates.append(candidate)

                if score > best_score:
                    best_clf, best_score, best_iteration, best_candidate = clf, score, iteration, candidate.copy()

            if valid_features:
                X_search = np.concatenate([X_search, np.vstack(valid_features)])
                y_search = np.concatenate([y_search, np.asarray(valid_scores).reshape(-1, 1)])

            dt = time.time() - iter_start
            iteration_times.append(dt)

            summaries.append({
                "dataset_name": DATASET, **metadata(path), "sample_dist": args.sample_dist,
                "requested_lambda_reg": requested_lambda, "effective_lambda_reg": lam,
                "lambda_reg_adjusted_due_to_sample_size": lambda_adjusted,
                "lambda_cap_1_over_sample_size": lambda_cap, "regularizer_sample_size": len(X_train),
                "reference_metric": METRIC, "rand_idx": rand_idx, "iteration": iteration,
                "iteration_time_seconds": dt, "num_valid_candidates": len(candidates),
                "mean_valid_score_normalized": float(np.mean(valid_scores)) if valid_scores else None,
                "best_valid_score_normalized": float(np.max(valid_scores)) if valid_scores else None,
                "best_score_so_far_normalized": float(best_score), "current_X_train_sgp_size": len(X_search),
                "train_metrics": train_metrics, "test_metrics": test_metrics, "valid_candidates": candidates
            })

            print(f"Iteration {iteration + 1}/{args.bo_iterations} | valid={len(candidates)} | best={best_score:.6f}")

        if best_clf is None:
            print("No valid classifier found.")
            continue

        search_time = time.time() - search_start
        pruned = best_clf.prune()
        baseline = DecisionTreeClassifier(max_depth=4, random_state=42, ccp_alpha=0.0001).fit(X_train, y_train)

        trevis_eval = evaluate_classifier(pruned, "bayesian_pruned", lam)
        baseline_eval = evaluate_classifier(baseline, "baseline", lam)
        best_score_raw = best_score * std_y + mean_y

        for clf, p in [(pruned, paths["model"]), (best_clf, paths["unpruned"]), (baseline, paths["baseline"])]:
            with open(p, "wb") as f: pickle.dump(clf, f)

        pred_paths = {}
        if args.save_predictions:
            for split, (_, y) in DATASETS.items():
                p = os.path.join(dirs["predictions"], f"{split}_predictions_{tag}.csv")
                pd.DataFrame({
                    "dataset_name": DATASET, "sample_dist": args.sample_dist,
                    "requested_lambda_reg": requested_lambda, "effective_lambda_reg": lam,
                    "lambda_reg_adjusted_due_to_sample_size": lambda_adjusted,
                    "lambda_cap_1_over_sample_size": lambda_cap, "regularizer_sample_size": len(X_train),
                    "reference_metric": METRIC, "rand_idx": rand_idx, **metadata(path),
                    f"y_{split}_true": np.asarray(y),
                    f"y_{split}_pred_bayesian_pruned": np.asarray(trevis_eval["pred"][split]),
                    f"y_{split}_pred_baseline": np.asarray(baseline_eval["pred"][split])
                }).to_csv(p, index=False)
                pred_paths[split] = p

        tree_paths = {}
        if args.save_tree_text:
            for name, clf in {
                "best_clf_pruned": pruned,
                "best_clf_unpruned": best_clf,
                "baseline_decision_tree": baseline
            }.items():
                p = os.path.join(dirs["trees"], f"{name}_{tag}.txt")
                with open(p, "w") as f: f.write(safe_tree_text(clf, feature_names))
                tree_paths[name] = p

        summary_path = os.path.join(dirs["json"], f"iteration_summaries_{tag}.json")
        save_json(summary_path, summaries)

        row = {
            "dataset_name": DATASET, **metadata(path), "sample_dist": args.sample_dist,
            "requested_lambda_reg": requested_lambda, "effective_lambda_reg": lam,
            "lambda_reg_adjusted_due_to_sample_size": lambda_adjusted,
            "lambda_cap_1_over_sample_size": lambda_cap, "regularizer_sample_size": len(X_train),
            "lambda_reg": lam, "reference_metric": METRIC, "rand_idx": rand_idx, "best_iteration": best_iteration,

            "best_score_normalized_train": best_score,
            "best_score_raw_train_negative_objective": best_score_raw,
            "best_objective_raw_train": -best_score_raw,

            **evaluation_to_row(trevis_eval, baseline_eval),

            "unpruned_depth": tree_depth(best_clf),
            "unpruned_n_leaves": tree_leaves(best_clf),
            "unpruned_n_nodes": tree_nodes(best_clf),

            "mlp_train_time_seconds": mlp_train_time,
            "mlp_test_mse_raw": mlp_raw["mse"],
            "mlp_test_rmse_raw": mlp_raw["rmse"],
            "mlp_test_pearson_raw": mlp_raw["pearson"],

            "final_mlp_train_mse_normalized": train_metrics["mse"],
            "final_mlp_train_rmse_normalized": train_metrics["rmse"],
            "final_mlp_train_pearson_normalized": train_metrics["pearson"],
            "final_mlp_test_mse_normalized": test_metrics["mse"],
            "final_mlp_test_rmse_normalized": test_metrics["rmse"],
            "final_mlp_test_pearson_normalized": test_metrics["pearson"],

            "num_bo_iterations": len(iteration_times),
            "iteration_times_seconds": iteration_times,
            "mean_iteration_time_seconds": float(np.mean(iteration_times)),
            "std_iteration_time_seconds": float(np.std(iteration_times)),
            "total_rand_time_seconds": search_time,

            "model_path": paths["model"],
            "unpruned_model_path": paths["unpruned"],
            "baseline_model_path": paths["baseline"],
            "mlp_model_path": mlp_path,

            **{f"{s}_predictions_path": pred_paths.get(s) for s in ("train", "val", "test")},

            "pruned_tree_text_path": tree_paths.get("best_clf_pruned"),
            "unpruned_tree_text_path": tree_paths.get("best_clf_unpruned"),
            "baseline_tree_text_path": tree_paths.get("baseline_decision_tree"),
            "iteration_summary_path": summary_path,
            "source_targets_path": source_targets_path,
            "run_config_path": run_config_path,

            **{f"best_candidate_{k}": v for k, v in best_candidate.items()},

            "result_source": "trained_new",
            "source_results_dir": source_dir,
            "bootstrap_results_dir": results_dir
        }

        row = flatten_csv(add_bootstrap(row, pruned, baseline))
        save_result(results_csv, row, rand_idx, requested_lambda, lam)

        print(
            f"Saved λ={requested_lambda}, rand={rand_idx} | "
            f"test {METRIC}={trevis_eval['ref']['test']:.4f} | leaves={trevis_eval['leaves']}"
        )


print("\nAll done.")
print(f"Results: {results_csv}")
print(f"Results folder: {results_dir}")
print(f"Source folder: {source_dir}")
print(f"size={SIZE} | grid={N_GRID} | dist={args.sample_dist}")
