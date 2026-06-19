def compute_metrics(model, X, y):
    y_pred = model.predict(X)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),

        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),

        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y, y_pred, average="micro", zero_division=0),
        "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),

        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y, y_pred, average="micro", zero_division=0),
        "precision_weighted": precision_score(y, y_pred, average="weighted", zero_division=0),

        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "recall_micro": recall_score(y, y_pred, average="micro", zero_division=0),
        "recall_weighted": recall_score(y, y_pred, average="weighted", zero_division=0),
    }


def compute_bootstrap_metrics_ci(
    model,
    X,
    y,
    n_iter=1000,
    seed=12345,
):
    predictions = model.predict(X)

    y_array = np.asarray(y)
    predictions = np.asarray(predictions)

    accuracy_original = accuracy_score(y_array, predictions)
    f1_weighted_original = f1_score(
        y_array,
        predictions,
        average="weighted",
        zero_division=0,
    )

    rng = np.random.RandomState(seed=seed)
    idx = np.arange(y_array.shape[0])

    bootstrap_accuracies = []
    bootstrap_f1_weighted_scores = []

    for _ in range(n_iter):
        sample_idx = rng.choice(idx, size=idx.shape[0], replace=True)

        y_boot = y_array[sample_idx]
        pred_boot = predictions[sample_idx]

        acc_boot = accuracy_score(y_boot, pred_boot)
        f1_weighted_boot = f1_score(
            y_boot,
            pred_boot,
            average="weighted",
            zero_division=0,
        )

        bootstrap_accuracies.append(acc_boot)
        bootstrap_f1_weighted_scores.append(f1_weighted_boot)

    bootstrap_accuracies = np.asarray(bootstrap_accuracies)
    bootstrap_f1_weighted_scores = np.asarray(bootstrap_f1_weighted_scores)

    return {
        "bootstrap_accuracy_original": accuracy_original,
        "bootstrap_accuracy_mean": np.mean(bootstrap_accuracies),
        "bootstrap_accuracy_ci_lower": np.percentile(bootstrap_accuracies, 2.5),
        "bootstrap_accuracy_ci_upper": np.percentile(bootstrap_accuracies, 97.5),

        "bootstrap_f1_weighted_original": f1_weighted_original,
        "bootstrap_f1_weighted_mean": np.mean(bootstrap_f1_weighted_scores),
        "bootstrap_f1_weighted_ci_lower": np.percentile(bootstrap_f1_weighted_scores, 2.5),
        "bootstrap_f1_weighted_ci_upper": np.percentile(bootstrap_f1_weighted_scores, 97.5),

        "bootstrap_n_iter": n_iter,
        "bootstrap_seed": seed,
    }


def compute_metrics_from_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),

        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),

        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),

        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),

        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }


#
def compute_bootstrap_metrics_from_predictions(
    y_true,
    y_pred,
    n_iter=200,
    seed=12345,
):
    """
    Bootstrap accuracy and weighted F1 confidence intervals.

    The model is NOT refit.
    We bootstrap fixed predictions by resampling evaluation indices.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rng = np.random.RandomState(seed)
    idx = np.arange(y_true.shape[0])

    bootstrap_accuracies = []
    bootstrap_f1_weighted_scores = []

    for _ in range(n_iter):
        sample_idx = rng.choice(idx, size=idx.shape[0], replace=True)

        y_boot = y_true[sample_idx]
        pred_boot = y_pred[sample_idx]

        bootstrap_accuracies.append(accuracy_score(y_boot, pred_boot))
        bootstrap_f1_weighted_scores.append(
            f1_score(
                y_boot,
                pred_boot,
                average="weighted",
                zero_division=0,
            )
        )

    bootstrap_accuracies = np.asarray(bootstrap_accuracies)
    bootstrap_f1_weighted_scores = np.asarray(bootstrap_f1_weighted_scores)

    return {
        "bootstrap_accuracy_original": accuracy_score(y_true, y_pred),
        "bootstrap_accuracy_mean": float(np.mean(bootstrap_accuracies)),
        "bootstrap_accuracy_ci_lower": float(np.percentile(bootstrap_accuracies, 2.5)),
        "bootstrap_accuracy_ci_upper": float(np.percentile(bootstrap_accuracies, 97.5)),

        "bootstrap_f1_weighted_original": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "bootstrap_f1_weighted_mean": float(np.mean(bootstrap_f1_weighted_scores)),
        "bootstrap_f1_weighted_ci_lower": float(np.percentile(bootstrap_f1_weighted_scores, 2.5)),
        "bootstrap_f1_weighted_ci_upper": float(np.percentile(bootstrap_f1_weighted_scores, 97.5)),

        "bootstrap_n_iter": n_iter,
        "bootstrap_seed": seed,
    }





### util functions for sklearn tree evaluation
def is_leaf(inner_tree, index):
    """Return True if node index is currently a leaf."""
    if index == TREE_LEAF:
        return False

    return (
        inner_tree.children_left[index] == TREE_LEAF
        and inner_tree.children_right[index] == TREE_LEAF
    )


def node_prediction(inner_tree, index):
    return int(inner_tree.value[index].argmax())


def subtree_leaf_predictions(inner_tree, index):
    if index == TREE_LEAF:
        return set()

    if is_leaf(inner_tree, index):
        return {node_prediction(inner_tree, index)}

    preds = set()

    left = inner_tree.children_left[index]
    right = inner_tree.children_right[index]

    if left != TREE_LEAF:
        preds |= subtree_leaf_predictions(inner_tree, left)

    if right != TREE_LEAF:
        preds |= subtree_leaf_predictions(inner_tree, right)

    return preds

def prune_homogeneous_subtrees(inner_tree, index=0):
    """
    Prune any subtree whose reachable leaves all predict the same class.
    """
    if index == TREE_LEAF or is_leaf(inner_tree, index):
        return False

    changed = False

    left = inner_tree.children_left[index]
    right = inner_tree.children_right[index]

    if left != TREE_LEAF:
        changed |= prune_homogeneous_subtrees(inner_tree, left)

    if right != TREE_LEAF:
        changed |= prune_homogeneous_subtrees(inner_tree, right)

    preds = subtree_leaf_predictions(inner_tree, index)

    if len(preds) == 1:
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        inner_tree.feature[index] = TREE_UNDEFINED
        inner_tree.threshold[index] = TREE_UNDEFINED
        changed = True

    return changed


def prune_duplicate_leaves(dt, max_iter=100):
    """
    Prune until no homogeneous subtree remains.
    """
    for _ in range(max_iter):
        changed = prune_homogeneous_subtrees(dt.tree_, index=0)

        if not changed:
            break

    return dt


def export_pruned_tree_text(clf, feature_names=None, decimals=4):
    """
    Correct replacement for sklearn.export_text after manual pruning.

    It traverses only the active tree reachable from the root.
    """
    tree = clf.tree_
    lines = []

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(tree.n_features)]
    else:
        feature_names = list(feature_names)

    def class_label(node_id):
        return int(tree.value[node_id].argmax())

    def recurse(node_id, depth):
        if node_id == TREE_LEAF:
            return

        indent = "|   " * depth

        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        if left == TREE_LEAF and right == TREE_LEAF:
            lines.append(f"{indent}|--- class: {class_label(node_id)}")
            return

        feature_id = tree.feature[node_id]
        threshold = tree.threshold[node_id]

        if feature_id == TREE_UNDEFINED:
            lines.append(f"{indent}|--- class: {class_label(node_id)}")
            return

        feature_name = feature_names[feature_id]
        threshold_str = f"{threshold:.{decimals}f}"

        lines.append(f"{indent}|--- {feature_name} <= {threshold_str}")
        recurse(left, depth + 1)

        lines.append(f"{indent}|--- {feature_name} >  {threshold_str}")
        recurse(right, depth + 1)

    recurse(0, 0)

    return "\n".join(lines)


def compute_active_tree_stats_from_text(tree_text):
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
        "tree_n_nodes_after_pruning": n_nodes,
        "tree_n_leaves_after_pruning": n_leaves,
        "tree_n_internal_nodes_after_pruning": n_internal_nodes,
        "tree_n_splits_after_pruning": n_internal_nodes,
        "tree_max_depth_after_pruning": max_depth,
    }


def compute_tree_stats_after_pruning(clf, feature_names=None):
    tree_text = export_pruned_tree_text(
        clf,
        feature_names=feature_names,
        decimals=4,
    )

    stats = compute_active_tree_stats_from_text(tree_text)
    stats["tree_export_text_after_pruning"] = tree_text

    return stats




## util functions for gosdt tree evaluation
class SimpleTree:
    def __init__(self, children_left, children_right, feature, threshold):
        self.children_left = np.array(children_left)
        self.children_right = np.array(children_right)
        self.feature = np.array(feature)
        self.threshold = np.array(threshold)

        self.node_count = len(self.feature)
        self.n_outputs = 1
        self.max_depth = self._max_depth()

    def _max_depth(self):
        def depth(node):
            left = self.children_left[node]
            right = self.children_right[node]

            if left == -1 and right == -1:
                return 0

            return 1 + max(depth(left), depth(right))

        return depth(0)


class ClfGOSDT:
    def __init__(self, children_left, children_right, feature, threshold):
        self.tree_ = SimpleTree(
            children_left,
            children_right,
            feature,
            threshold,
        )

        valid_features = feature[feature >= 0]
        self.n_features_in_ = (
            int(np.max(valid_features)) + 1 if len(valid_features) > 0 else 0
        )


def gosdt_tree_to_dict(s):
    s = re.sub(r'([{\[,]\s*)([A-Za-z ]+)(\s*:)', r'\1"\2"\3', s)
    s = re.sub(r'("feature"\s*:\s*\d+)\s*\[', r'\1,', s)
    s = s.replace("]", "")
    return ast.literal_eval(s)


def gosdt_dict_to_sklearn_arrays(tree_dict, threshold_value=0.5):
    children_left = []
    children_right = []
    feature = []
    threshold = []

    def build(node):
        node_id = len(feature)

        children_left.append(-1)
        children_right.append(-1)
        feature.append(-2)
        threshold.append(-2.0)

        if "prediction" in node:
            return node_id

        feature[node_id] = int(node["feature"])
        threshold[node_id] = float(threshold_value)

        left_id = build(node["left child"])
        right_id = build(node["right child"])

        children_left[node_id] = left_id
        children_right[node_id] = right_id

        return node_id

    build(tree_dict)

    return (
        np.array(children_left, dtype=np.int64),
        np.array(children_right, dtype=np.int64),
        np.array(feature, dtype=np.int64),
        np.array(threshold, dtype=np.float64),
    )


# here below useful function to extract information from trees trained through FLOW
def build_flowoct_node_dict(flow_clf):
    """
    Build node status dictionary from a fitted FlowOCT classifier.

    Each entry has format:
        node -> (pruned, branching, selected_feature, cutoff, leaf, value)
    """

    node_dict = {}

    for node in np.arange(1, flow_clf._tree.total_nodes + 1):
        node_dict[int(node)] = flow_clf._get_node_status(
            flow_clf.b_value,
            flow_clf.w_value,
            flow_clf.p_value,
            node,
            feature_names=flow_clf._X_col_labels,
        )

    return node_dict


def get_actual_tree_stats(node_dict):
    """
    Compute actual depth, actual number of nodes, actual number of leaves,
    and actual number of internal nodes from a FlowOCT node_dict.

    node_dict format:
        node -> (pruned, branching, selected_feature, cutoff, leaf, value)
    """

    actual_nodes = []
    actual_leaf_nodes = []
    actual_internal_nodes = []

    for node, status in node_dict.items():
        pruned, branching, selected_feature, cutoff, leaf, value = status

        if not pruned:
            node = int(node)
            actual_nodes.append(node)

            if leaf:
                actual_leaf_nodes.append(node)

            elif branching:
                actual_internal_nodes.append(node)

    actual_depth = (
        max(int(np.floor(np.log2(node))) for node in actual_nodes)
        if actual_nodes
        else 0
    )

    return {
        "actual_depth": actual_depth,
        "actual_n_nodes": len(actual_nodes),
        "actual_n_leaves": len(actual_leaf_nodes),
        "actual_n_internal_nodes": len(actual_internal_nodes),
        "actual_nodes": sorted(actual_nodes),
        "actual_leaf_nodes": sorted(actual_leaf_nodes),
        "actual_internal_nodes": sorted(actual_internal_nodes),
    }


def extract_flowoct_tree_structure(node_dict):
    """
    Convert FlowOCT node_dict into an explicit, pickle-safe tree structure.

    node_dict format:
        node -> (pruned, branching, selected_feature, cutoff, leaf, value)

    Returned format:
        node -> {
            "node": int,
            "type": "branch" | "leaf" | "active_unknown",
            "feature": selected_feature or None,
            "cutoff": cutoff or None,
            "prediction": value or None,
            "left_child": int or None,
            "right_child": int or None,
        }

    Pruned nodes are excluded.
    """

    tree_structure = {}

    for node, status in node_dict.items():
        pruned, branching, selected_feature, cutoff, leaf, value = status

        if pruned:
            continue

        node = int(node)

        if leaf:
            tree_structure[node] = {
                "node": node,
                "type": "leaf",
                "feature": None,
                "cutoff": None,
                "prediction": value,
                "left_child": None,
                "right_child": None,
            }

        elif branching:
            left_child = 2 * node
            right_child = 2 * node + 1

            tree_structure[node] = {
                "node": node,
                "type": "branch",
                "feature": selected_feature,
                "cutoff": cutoff,
                "prediction": None,
                "left_child": left_child,
                "right_child": right_child,
            }

        else:
            tree_structure[node] = {
                "node": node,
                "type": "active_unknown",
                "feature": None,
                "cutoff": cutoff,
                "prediction": None,
                "left_child": None,
                "right_child": None,
            }

    return tree_structure


def extract_flowoct_nested_tree(tree_structure, root_node=1):
    """
    Convert the flat tree_structure into a nested dictionary.

    This is useful if you want a recursive tree representation:
        root -> left subtree / right subtree

    Pruned children are returned as None.
    """

    if root_node not in tree_structure:
        return None

    node_info = tree_structure[root_node]

    if node_info["type"] == "leaf":
        return {
            "node": node_info["node"],
            "type": "leaf",
            "prediction": node_info["prediction"],
        }

    if node_info["type"] == "branch":
        return {
            "node": node_info["node"],
            "type": "branch",
            "feature": node_info["feature"],
            "cutoff": node_info["cutoff"],
            "left_child": extract_flowoct_nested_tree(
                tree_structure,
                node_info["left_child"],
            ),
            "right_child": extract_flowoct_nested_tree(
                tree_structure,
                node_info["right_child"],
            ),
        }

    return {
        "node": node_info["node"],
        "type": node_info["type"],
    }


def extract_flowoct_artifact(flow_clf, params=None):
    """
    Extract a pickle-safe representation of a fitted FlowOCT model.

    Do NOT pickle the full FlowOCT object, because it can contain Gurobi/CFFI
    backend objects that are not pickleable.

    This artifact stores:
        - node statuses
        - explicit flat tree structure
        - explicit nested tree structure
        - actual tree stats
        - b, w, p solution values
        - feature labels
        - class labels
        - total number of nodes
        - model parameters
    """

    node_dict = build_flowoct_node_dict(flow_clf)
    tree_stats = get_actual_tree_stats(node_dict)
    tree_structure = extract_flowoct_tree_structure(node_dict)
    nested_tree_structure = extract_flowoct_nested_tree(tree_structure, root_node=1)

    artifact = {
        "params": params if params is not None else {},

        "node_dict": node_dict,

        "tree_structure": tree_structure,
        "nested_tree_structure": nested_tree_structure,

        "tree_stats": tree_stats,

        "b_value": dict(flow_clf.b_value),
        "w_value": dict(flow_clf.w_value),
        "p_value": dict(flow_clf.p_value),

        "X_col_labels": list(flow_clf._X_col_labels),
        "labels": list(flow_clf._labels),
        "total_nodes": flow_clf._tree.total_nodes,
    }

    return artifact


def predict_from_flowoct_artifact(artifact, X):
    """
    Predict using a saved FlowOCT artifact.

    This avoids needing to pickle/load the full FlowOCT object.
    It traverses artifact["tree_structure"] from node 1.

    FlowOCT node status stores:
        - branch feature
        - cutoff
        - leaf prediction

    Assumption:
        left branch is taken when X[feature] <= cutoff,
        right branch otherwise.

    This matches the standard binary-tree convention used by ODT-style trees.
    """
    tree_structure = artifact["tree_structure"]

    if not isinstance(X, pd.DataFrame):
        feature_names = artifact.get("X_col_labels", None)
        if feature_names is None:
            raise ValueError("X must be a DataFrame or artifact must contain X_col_labels.")
        X_eval = pd.DataFrame(X, columns=feature_names)
    else:
        X_eval = X

    preds = []

    for _, row in X_eval.iterrows():
        node = 1

        while True:
            if node not in tree_structure:
                raise ValueError(
                    f"Traversal reached missing/pruned node {node}. "
                    "Cannot predict from this artifact."
                )

            node_info = tree_structure[node]

            if node_info["type"] == "leaf":
                preds.append(node_info["prediction"])
                break

            if node_info["type"] != "branch":
                raise ValueError(
                    f"Cannot predict from node {node} with type={node_info['type']}."
                )

            feature = node_info["feature"]
            cutoff = node_info["cutoff"]

            value = row[feature]

            if value <= cutoff:
                node = node_info["left_child"]
            else:
                node = node_info["right_child"]

    return np.asarray(preds)




