import copy
import warnings
import numpy as np
import pandas as pd
import torch
from graphviz import Digraph
import numpy as np

class DecodedDecisionTree:

    def __init__(self, state):
        self.state = state
        self.root_id = state['root_id']
        self.feature = state['feature']
        self.threshold = state['threshold']
        self.children_left = state['children_left']
        self.children_right = state['children_right']
        self.value = state['value']
        self.classes = state['classes']

    def prune(self):
        
        new_state = copy.deepcopy(self.state)
        feature = new_state['feature']
        threshold = new_state['threshold']
        children_left = new_state['children_left']
        children_right = new_state['children_right']
        value = new_state['value']
    
        def is_leaf(node_id):
            return children_left[node_id] == -1 and children_right[node_id] == -1
    
        def predicted_class(node_id):
            v = np.asarray(value[node_id])
            if v.ndim == 2 and v.shape[0] == 1:
                v = v[0]
            return int(np.argmax(v))
    
        def visit(node_id):
            if node_id == -1 or is_leaf(node_id):
                return
    
            left = children_left[node_id]
            right = children_right[node_id]
    
            visit(left)
            visit(right)
    
            if is_leaf(left) and is_leaf(right) and predicted_class(left) == predicted_class(right):
                children_left[node_id] = -1
                children_right[node_id] = -1
                feature[node_id] = -2
                threshold[node_id] = -2.0
                value[node_id] = value[left] + value[right]
    
        visit(new_state['root_id'])
    
        return DecodedDecisionTree(new_state)

    def get_n_nodes(self):
        return len(self.feature)

    def get_depth(self):
        def depth(node_id):
            left = self.children_left[node_id]
            right = self.children_right[node_id]
            if left == -1 and right == -1:
                return 0
            return 1 + max(depth(left), depth(right))
        return depth(self.root_id)
    
    def get_n_leaves(self):
        def count_leaves(node_id):
            left = self.children_left[node_id]
            right = self.children_right[node_id]
            if left == -1 and right == -1:
                return 1
            return count_leaves(left) + count_leaves(right)
        return count_leaves(self.root_id)
    
    def _predict_row(self, row):
        node_id = self.root_id
    
        while self.children_left[node_id] != -1 and self.children_right[node_id] != -1:
            feat_id = self.feature[node_id]
            thr = self.threshold[node_id]

            if float(row.iloc[feat_id]) <= thr:
                node_id = self.children_left[node_id]
            else:
                node_id = self.children_right[node_id]

        leaf_counts = self.value[node_id][0]  
        pred_class = self.classes[np.argmax(leaf_counts)]
        
        return pred_class

    def predict(self, X):
        y_pred = np.array([self._predict_row(X.loc[idx]) for idx in X.index])
        return y_pred

    def __str__(self):
        def recurse(node_id, depth=0):
            indent = "|   " * depth
    
            left = self.children_left[node_id]
            right = self.children_right[node_id]
    
            if left == -1 and right == -1:
                leaf_counts = self.value[node_id][0]
                pred_class = self.classes[np.argmax(leaf_counts)]
                return f"{indent}|--- class: {pred_class}\n"
    
            feat_id = self.feature[node_id]
            thr = self.threshold[node_id]
            feat_name = f"feature_{feat_id}"
    
            s = f"{indent}|--- {feat_name} <= {thr}\n"
            s += recurse(left, depth + 1)
            s += f"{indent}|--- {feat_name} >  {thr}\n"
            s += recurse(right, depth + 1)
    
            return s
    
        return recurse(self.root_id).rstrip()

    def plot_tree(self, feature_names=None, class_names=None, filename="tree", view=False):
    
        dot = Digraph()
        dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")
        dot.attr("edge", fontname="Helvetica")
    
        def recurse(node_id):
            left = self.children_left[node_id]
            right = self.children_right[node_id]
    
            if left == -1 and right == -1:
                counts = self.value[node_id][0]
                pred_idx = np.argmax(counts)
                pred_class = self.classes[pred_idx] if class_names is None else class_names[pred_idx]
                label = f"node {node_id}\nclass = {pred_class}\ncounts = {counts.tolist()}"
                dot.node(str(node_id), label=label)
                return
    
            feat_id = self.feature[node_id]
            thr = self.threshold[node_id]
    
            if feature_names is None:
                feat_name = f"feature_{feat_id}"
            else:
                feat_name = feature_names[feat_id]
    
            label = f"node {node_id}\n{feat_name} <= {thr:.4f}"
            dot.node(str(node_id), label=label)
    
            recurse(left)
            recurse(right)
    
            dot.edge(str(node_id), str(left), label="True")
            dot.edge(str(node_id), str(right), label="False")
    
        recurse(self.root_id)
    
        dot.render(filename, format="png", cleanup=True, view=view)
        return dot

class TreeEncoder:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        token_to_id: dict = None,
        tokenization: str = 'threshold',
        precision: int = 3,
        source: str = 'data',
        trees: list = None,
    ):
        self.precision = precision
        self.X = X
        if source=='data':
            self.X = self.X.round(self.precision)  # pd.DataFrame
        self.y = y
        self.feature_names = X.columns
        self.token_to_id = token_to_id
        self.tokenization = tokenization
        self.source = source
        self.trees = trees
        self.max_depth = 5
        self._build_vocabulary()

    def _build_vocabulary(self):
        """Builds token vocabulary."""
        
        if self.token_to_id is None:
            
            self.token_to_id = {
                '<PAD>': 0,
                '<CLS>': 1,
                '<BOS>': 2,
                '<EOS>': 3,
                '<UNK>': 4,
                '<T>': 5,  # leaf token
            }

            tokens = list()

            if self.source == 'data':
                for i, feature_name in enumerate(self.X.columns):
                    thrs = sorted(float(v) for v in pd.unique(self.X[feature_name]))
                    max_value = max(thrs)

                    if self.tokenization == 'node':
                        tokens.extend([(str(feature_name), thr) for thr in thrs if thr!=max_value])
                        
                    elif self.tokenization == 'threshold':
                        tokens.append(str(feature_name))
                        tokens.extend([thr for thr in thrs if thr!=max_value])
                    else:
                        raise ValueError('Invalid tokenization.')

            elif self.source == 'trees':
                
                if not self.trees:
                    raise ValueError('No trees available.')

                for tree in self.trees:
                    feature = tree.tree_.feature
                    threshold = tree.tree_.threshold

                    for i, feature_id in enumerate(feature):
                        if feature_id != -2:
                            feature_name = self.feature_names[feature_id]
                            thr = float(threshold[i])

                            if self.tokenization == 'node':
                                tokens.append((feature_name, thr))
                            elif self.tokenization == 'threshold':
                                tokens.append(feature_name)
                                tokens.append(thr)
                            else:
                                raise ValueError('Invalid tokenization.')
            else:
                raise ValueError("Invalid source.")

            for token in tokens:
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.token_to_id)

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def _map_token(self, token):
        """Maps token to token_id."""
        
        tid = self.token_to_id.get(token)
        if tid is not None:
            return tid
        return self.token_to_id['<UNK>']

    def get_examples(self, clf, feature_id, node_id=None):
        """Get values of a feature for samples reaching a given node, or all nodes."""

        X = np.asarray(self.X)
        
        n_samples, n_features = X.shape

        if not (0 <= feature_id < n_features):
            raise ValueError(f"feature_id must be in [0, {n_features - 1}].")

        if node_id is not None:
            node_id = int(node_id)

        node_indicator = clf.decision_path(X)

        if node_id is not None:
            result_idx = list()

            for i in range(n_samples):
                start = node_indicator.indptr[i]
                end = node_indicator.indptr[i + 1]
                path_node_ids = node_indicator.indices[start:end]

                if node_id in path_node_ids:
                    result_idx.append(i)

            return X[result_idx, feature_id].tolist()

        node_to_examples = dict()

        for i in range(n_samples):
            start = node_indicator.indptr[i]
            end = node_indicator.indptr[i + 1]
            path_node_ids = node_indicator.indices[start:end]

            for nid in path_node_ids:
                nid = int(nid)
                if nid not in node_to_examples:
                    node_to_examples[nid] = []
                node_to_examples[nid].append(X[i, feature_id])

        return {nid: np.asarray(vals).tolist() for nid, vals in node_to_examples.items()}

    def _map_threshold(self, values, threshold):
        """
        Maps a threshold to the largest observed value <= threshold.
        """
        
        unique = np.unique(np.asarray(values, dtype=np.float64))
        candidates = unique[unique <= float(threshold)]

        if candidates.size == 0:
            raise ValueError(
                f"No observed value <= threshold={threshold}."
            )

        return float(candidates.max())

    def _map_thresholds(self, clf, get_from_nodes=False):
        """Maps tree thresholds to observed values in self.X."""
        
        thresholds = clf.tree_.threshold
        feature_ids = clf.tree_.feature

        new_thresholds = list()

        for i, (feature_id, threshold) in enumerate(zip(feature_ids, thresholds)):
            if feature_id == -2:
                new_thresholds.append(-2.0)
            else:
                if get_from_nodes:
                    values = self.get_examples(clf, feature_id, i)
                else:
                    values = self.X.iloc[:, feature_id]

                new_thresholds.append(self._map_threshold(values, threshold))

        new_thresholds = np.asarray(new_thresholds, dtype=np.float64)
        clf.tree_.threshold[:] = new_thresholds
        return clf

    def _get_dist(self, token_to_node, depth, parent):
        """
        Returns matrix of distances between nodes in the tree.
        If self.tokenization='threshold', each feature-threshold pair is assigned
        the same distance.
        """
        
        T = len(token_to_node)
        dist = torch.zeros((T, T), dtype=torch.long)

        for i in range(T):
            for j in range(i + 1, T):
                u = int(token_to_node[i])
                v = int(token_to_node[j])

                uu, vv = u, v

                while depth[uu] > depth[vv]:
                    uu = parent[uu]
                while depth[vv] > depth[uu]:
                    vv = parent[vv]

                while uu != vv:
                    uu = parent[uu]
                    vv = parent[vv]

                d = depth[u] + depth[v] - 2 * depth[uu]
                dist[i, j] = int(d)
                dist[j, i] = int(d)

        return dist

    def encode_tree(self, clf, abs_encoding=True, rel_encoding=True):
        """
        Maps a DecisionTreeClassifier to a sequence of token ids.
        Optionally returns raw absolute tree embeddings and raw relative distances.
        """
        
        if self.source == 'data':
            clf = copy.deepcopy(clf)
            clf = self._map_thresholds(clf)

        tree_ = clf.tree_

        children_left = tree_.children_left
        children_right = tree_.children_right
        feature = tree_.feature
        threshold = tree_.threshold
        feature_names = np.asarray(self.feature_names, dtype=object)

        token_ids = list()
        abs_encs = list()
        token_to_node = list()

        parent = np.full(tree_.node_count, -1, dtype=np.int32)
        depth = np.zeros(tree_.node_count, dtype=np.int32)

        def add(token_id, code, node_id):
            token_ids.append(token_id)
            if abs_encoding:
                abs_encs.append(code)
            if rel_encoding:
                token_to_node.append(node_id)

        def recurse_dfs(node_id, code):
            left = int(children_left[node_id])
            right = int(children_right[node_id])

            if left == right:  # leaf
                add(self.token_to_id['<T>'], code, node_id)
                return

            feat_idx = int(feature[node_id])
            feature_name = str(feature_names[feat_idx])
            thr = threshold[node_id]

            if self.source == 'data':
                thr = round(float(thr), self.precision)

            if rel_encoding:
                parent[left] = node_id
                parent[right] = node_id
                depth[left] = depth[node_id] + 1
                depth[right] = depth[node_id] + 1

            if self.tokenization == 'node':
                tok = self._map_token((feature_name, thr))
                add(tok, code, node_id)

            elif self.tokenization == 'threshold':
                feat_tok = self._map_token(feature_name)
                thr_tok = self._map_token(thr)
                add(feat_tok, code, node_id)
                add(thr_tok, code, node_id)

            else:
                raise ValueError("Invalid tokenization. Use 'node' or 'threshold'.")

            recurse_dfs(left, [1, 0] + code)
            recurse_dfs(right, [0, 1] + code)

        recurse_dfs(0, [0, 0])

        if self.token_to_id['<UNK>'] in token_ids:
            warnings.warn('<UNK> tokens found in encoded tree.', UserWarning)

        src = torch.tensor([self.token_to_id['<CLS>']] + token_ids, dtype=torch.long)
        tgt = torch.tensor(
            [self.token_to_id['<BOS>']] + token_ids + [self.token_to_id['<EOS>']],
            dtype=torch.long,
        )

        outputs = {'src': src, 'tgt': tgt}

        if abs_encoding:
            max_len = (self.max_depth + 1) * 2
            abs_encs = [code + [0] * (max_len - len(code)) for code in abs_encs]
            zero_row = [0] * max_len
            eos_row = abs_encs[-1] 
        
            outputs['src_abs_encs'] = torch.tensor([zero_row] + abs_encs, dtype=torch.long) 
            outputs['tgt_abs_encs'] = torch.tensor([zero_row] + abs_encs + [eos_row], dtype=torch.long) 
    
        if rel_encoding:
            rel_encs = self._get_dist(token_to_node, depth, parent)
            T = rel_encs.size(0)
            special_rel = 1 
            
            src_rel_encs = torch.empty((T + 1, T + 1), dtype=torch.long)
            src_rel_encs[1:, 1:] = rel_encs
            src_rel_encs[0, 1:] = rel_encs[0, :]
            src_rel_encs[1:, 0] = rel_encs[:, 0]
            src_rel_encs[0, 0] = rel_encs[0, 0]
            outputs['src_rel_encs'] = src_rel_encs

            tgt_rel_encs = torch.full((T + 2, T + 2), special_rel, dtype=torch.long)
            tgt_rel_encs[1:T+1, 1:T+1] = rel_encs
            
            tgt_rel_encs[0, 1:T+1] = rel_encs[0, :]
            tgt_rel_encs[1:T+1, 0] = rel_encs[:, 0]
            tgt_rel_encs[0, 0] = rel_encs[0, 0]
            
            tgt_rel_encs[T+1, 1:T+1] = rel_encs[-1, :] 
            tgt_rel_encs[1:T+1, T+1] = rel_encs[:, -1]
            tgt_rel_encs[T+1, T+1] = rel_encs[-1, -1]
            
            tgt_rel_encs[0, T+1] = rel_encs[0, -1] 
            tgt_rel_encs[T+1, 0] = rel_encs[-1, 0]
            
            outputs['tgt_rel_encs'] = tgt_rel_encs
    
        return outputs

    def _get_node_ids(self, token_ids):
        """
        Maps token ids to structural node-type ids used by ids_to_pos_encoding.
        """
  
        special_tokens = {'<PAD>', '<CLS>', '<BOS>', '<EOS>', '<UNK>'}
        leaf_token = '<T>'

        out = list()

        for tid in token_ids:
            tid = int(tid)
            tok = self.id_to_token.get(tid, '<UNK>')

            if tok in special_tokens:
                out.append(0)
            
            elif tok == leaf_token:
                out.append(1)

            else:
                if self.tokenization == 'node':
                    out.append(2)
                elif self.tokenization == 'threshold':
                    if isinstance(tok, str):
                        out.append(2)   # feature name
                    else:
                        out.append(3)  # threshold 
                else:
                    raise ValueError("Invalid tokenization.")
        return out

    def ids_to_pos_encoding(self, token_ids, abs_encoding=True, rel_encoding=True):
        """
        Rebuild tgt positional encodings at inference time from an autoregressive
        target prefix.
        """
    
        if not abs_encoding and not rel_encoding:
            return dict()
    
        if isinstance(token_ids, torch.Tensor):
            device = token_ids.device
            token_ids = token_ids.detach().to(torch.long).view(-1).tolist()
        else:
            device = None
            token_ids = list(token_ids)
    
        if len(token_ids) == 0:
            raise ValueError("token_ids cannot be empty.")
    
        bos_id = self.token_to_id['<BOS>']
        eos_id = self.token_to_id['<EOS>']
    
        if token_ids[0] != bos_id:
            raise ValueError("token_ids must start with <BOS>.")
    
        has_eos = len(token_ids) > 1 and token_ids[-1] == eos_id
        tree_token_ids = token_ids[1:-1] if has_eos else token_ids[1:]
    
        outputs = dict()
    
        max_len = (self.max_depth + 1) * 2 if abs_encoding else None
        zero_row = [0] * max_len if abs_encoding else None
        special_rel = 1
    
        if len(tree_token_ids) == 0:
            if abs_encoding:
                rows = [zero_row]
                if has_eos:
                    rows.append(zero_row)
                outputs["tgt_abs_encs"] = torch.tensor(rows, dtype=torch.long, device=device)
    
            if rel_encoding:
                size = 2 if has_eos else 1
                outputs["tgt_rel_encs"] = torch.full(
                    (size, size), special_rel, dtype=torch.long, device=device
                )
    
            return outputs
    
        nids = self._get_node_ids(tree_token_ids)
        if isinstance(nids, torch.Tensor):
            nids = nids.tolist()
    
        abs_encs = list()
        token_to_node = list()
    
        parent = list()
        depth = list()
    
        pending_right = list()
        next_code = [0, 0]
        next_parent = -1
        next_depth = 0
    
        current_node_id = None
        current_code = None
    
        def new_node(parent_id, d):
            node_id = len(parent)
            parent.append(parent_id)
            depth.append(d)
            return node_id
    
        def move_to_next_pending():
            nonlocal next_code, next_parent, next_depth
            if pending_right:
                parent_node_id, parent_code, parent_depth = pending_right.pop()
                next_code = [0, 1] + parent_code
                next_parent = parent_node_id
                next_depth = parent_depth + 1
            else:
                next_code = None
                next_parent = -1
                next_depth = 0
    
        for nid in nids:
            nid = int(nid)
    
            if nid == 0:
                raise ValueError("Special tokens are not allowed inside tree_token_ids.")
    
            elif nid == 1: # Leaf <T>
                if next_code is None:
                    raise ValueError("Invalid sequence: leaf token cannot be placed here.")
    
                node_id = new_node(next_parent, next_depth)
                current_node_id = node_id
                current_code = next_code
    
                if abs_encoding:
                    abs_encs.append(current_code)
                if rel_encoding:
                    token_to_node.append(node_id)
    
                move_to_next_pending()
    
            elif nid == 2:
                if next_code is None:
                    raise ValueError("Invalid sequence: internal node cannot be placed here.")
    
                node_id = new_node(next_parent, next_depth)
                current_node_id = node_id
                current_code = next_code
    
                if abs_encoding:
                    abs_encs.append(current_code)
                if rel_encoding:
                    token_to_node.append(node_id)
    
                if self.tokenization == "node":
                    pending_right.append((node_id, current_code, next_depth))
                    next_code = [1, 0] + current_code
                    next_parent = node_id
                    next_depth = next_depth + 1
    
            elif nid == 3: # threshold token
                if self.tokenization != "threshold":
                    raise ValueError("Threshold token found while tokenization != 'threshold'.")
    
                if current_node_id is None or current_code is None:
                    raise ValueError("Invalid sequence: threshold token without preceding feature token.")
    
                if abs_encoding:
                    abs_encs.append(current_code)
                if rel_encoding:
                    token_to_node.append(current_node_id)
    
                pending_right.append((current_node_id, current_code, depth[current_node_id]))
                next_code = [1, 0] + current_code
                next_parent = current_node_id
                next_depth = depth[current_node_id] + 1
    
            else:
                raise ValueError(f"Invalid node type id: {nid}")
    
        if abs_encoding:
            abs_encs = [
                code + [0] * (max_len - len(code))
                for code in abs_encs
            ]
    
            bos_row = abs_encs[0]
            eos_row = zero_row
    
            tgt_abs_encs = [bos_row] + abs_encs
            if has_eos:
                tgt_abs_encs.append(eos_row)
    
            outputs["tgt_abs_encs"] = torch.tensor(
                tgt_abs_encs, dtype=torch.long, device=device
            )
    
        if rel_encoding:
            T = len(token_to_node)
            total_len = T + 1 + (1 if has_eos else 0)
    
            tgt_rel_encs = torch.full(
                (total_len, total_len), special_rel, dtype=torch.long
            )
    
            tree_rel_encs = self._get_dist(token_to_node, depth, parent)
            if not isinstance(tree_rel_encs, torch.Tensor):
                tree_rel_encs = torch.tensor(tree_rel_encs, dtype=torch.long)
            else:
                tree_rel_encs = tree_rel_encs.to(torch.long)
    
            tgt_rel_encs[1:T + 1, 1:T + 1] = tree_rel_encs
    
            tgt_rel_encs[0, 1:T + 1] = tree_rel_encs[0, :] # BOS shares position with first generated token
            tgt_rel_encs[1:T + 1, 0] = tree_rel_encs[:, 0]
            tgt_rel_encs[0, 0] = tree_rel_encs[0, 0]
    
            outputs["tgt_rel_encs"] = (
                tgt_rel_encs.to(device) if device is not None else tgt_rel_encs
            )
    
        return outputs

    def decode_tree(self, ids, prune=True):
        """
        Maps a sequence of token ids to a DecodedTreeClassifier.
        Decisions at terminal nodes are computed from class counts of the
        training examples reaching the node.
        """
        
        special_ids = {
            self.token_to_id['<PAD>'],
            self.token_to_id['<BOS>'],
            self.token_to_id['<EOS>'],
            self.token_to_id['<CLS>'],
        }

        seq = [i for i in ids if i not in special_ids]
        feature_name_to_id = {str(name): i for i, name in enumerate(self.feature_names)}

        feature = list()
        threshold = list()
        value = list()
        children_left = list()
        children_right = list()

        classes = sorted(self.y.unique())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        n_classes = len(classes)

        def new_node():
            node_id = len(feature)
            children_left.append(-1)
            children_right.append(-1)
            feature.append(-2)
            threshold.append(-2.0)
            value.append(None)
            return node_id

        def counts_from_idx(sample_idx):
            counts = np.zeros((1, n_classes), dtype=np.float64)
            if len(sample_idx) > 0:
                for cls in self.y.loc[sample_idx]:
                    counts[0, class_to_idx[cls]] += 1.0
            return counts

        def split(feat_id, sample_idx, thr):

            left_idx = list()
            right_idx = list()

            X_sub = self.X.loc[sample_idx]

            if self.source == 'data':
                values = X_sub.iloc[:, feat_id].round(self.precision).to_dict()
            elif self.source == 'trees':
                values = X_sub.iloc[:, feat_id].to_dict()
                
            if self.source == 'data':
                thr = round(float(thr), self.precision)

            left_idx = [k for k in values if values[k] <= thr]
            right_idx = [k for k in values if values[k] > thr]

            return left_idx, right_idx

        pos = 0

        def parse(sample_idx):
            nonlocal pos

            if pos >= len(seq):
                raise DecodingError('Invalid tree: sequence ended before completion.')

            token_id = seq[pos]
            token = self.id_to_token[token_id]

            node_id = new_node()
            value[node_id] = counts_from_idx(sample_idx)

            if token == '<T>':
                feature[node_id] = -2
                threshold[node_id] = -2.0
                pos += 1
                return node_id

            if self.tokenization == 'node':
                feature_name, thr_tok = token
                thr = float(thr_tok)
                feature[node_id] = feature_name_to_id[feature_name]
                threshold[node_id] = thr
                pos += 1

            elif self.tokenization == 'threshold':
                feature_name = self.id_to_token[seq[pos]]
                pos += 1

                if pos >= len(seq):
                    raise DecodingError('Invalid tree: sequence ended before threshold.')

                thr_tok = self.id_to_token[seq[pos]]
                pos += 1

                try:
                    thr = float(thr_tok)
                except Exception as e:
                    raise DecodingError('Invalid tree: missing or invalid threshold.') from e

                feature[node_id] = feature_name_to_id[str(feature_name)]
                threshold[node_id] = thr

            else:
                raise ValueError('Invalid tokenization.')

            feat_id = feature[node_id]
            left_idx, right_idx = split(feat_id, sample_idx, thr)
            left_id = parse(left_idx)
            right_id = parse(right_idx)

            children_left[node_id] = left_id
            children_right[node_id] = right_id

            return node_id

        root_idx = self.X.index.tolist()
        root_id = parse(root_idx)

        if pos != len(seq):
            raise DecodingError('Invalid tree: trailing tokens after completion.')

        node_count = len(feature)

        state = {
            'root_id': root_id,
            'node_count': node_count,
            'feature': feature,
            'threshold': threshold,
            'children_left': children_left,
            'children_right': children_right,
            'value': value,
            'classes': classes,
        }

        clf = DecodedDecisionTree(state=state)
        return clf