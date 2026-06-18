import os
import pandas as pd
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import numpy as np
import joblib
import json
from sklearn.tree._tree import TREE_LEAF

def is_leaf(inner_tree, index):
    """Return True if node index is a leaf."""
    
    if index == TREE_LEAF:
        return False
    return (
        inner_tree.children_left[index] == TREE_LEAF and
        inner_tree.children_right[index] == TREE_LEAF
    )

def prune_index(inner_tree, decisions, index=0):
    """Recursively prune subtrees whose two leaf children predict the same class."""
    
    left = inner_tree.children_left[index]
    right = inner_tree.children_right[index]

    if left != TREE_LEAF and not is_leaf(inner_tree, left):
        prune_index(inner_tree, decisions, left)

    if right != TREE_LEAF and not is_leaf(inner_tree, right):
        prune_index(inner_tree, decisions, right)

    left = inner_tree.children_left[index]
    right = inner_tree.children_right[index]

    if (
        left != TREE_LEAF and
        right != TREE_LEAF and
        is_leaf(inner_tree, left) and
        is_leaf(inner_tree, right) and
        decisions[left] == decisions[right]
    ):
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        inner_tree.feature[index] = -2
        inner_tree.threshold[index] = -2.0

def prune_duplicate_leaves(dt):
    """Prune duplicate leaves in a fitted DecisionTreeClassifier."""
    
    decisions = dt.tree_.value.argmax(axis=2).flatten()
    prune_index(dt.tree_, decisions)
    return dt

def tree_signature(clf):
    t = clf.tree_
    return (
        tuple(t.feature),
        tuple(np.round(t.threshold, 10)), 
        tuple(t.children_left),
        tuple(t.children_right),
        tuple(t.value.flatten())
    )

def remove_duplicates(trees):
    
    seen = set()
    unique = list()
    
    for clf in trees:
        sig = tree_signature(clf)
        if sig not in seen:
            seen.add(sig)
            unique.append(clf)
    
    return unique

if __name__ == '__main__':

    with open('../config/random_trees_config.json', 'r') as file:
         config = json.load(file)

    target_dir = config['target_dir']
    dataset_name = config['dataset_name']
    sample_size = config['sample_size']
    max_depth = config['max_depth']
    precision = config['precision']

    os.makedirs(config['target_dir'], exist_ok=True)

    base_dir = '../data'
    data = pd.read_csv(os.path.join(base_dir, f'{dataset_name}.csv'))
    train = data[data.split=='train'].drop('split', axis=1)
    X_train, y_train = train.drop('target', axis=1).round(precision), train['target']

    rng = random.Random(42)
    clfs = list()
    
    with tqdm(total=sample_size, desc=dataset_name) as pbar:
        while len(clfs) < sample_size:
            
            random_state = rng.randint(0, 2**31 - 1)
            depth = rng.randint(2, max_depth)
    
            clf = ExtraTreeClassifier(
                max_features=1,
                max_depth=depth,
                splitter="random",
                random_state=random_state
            )
    
            clf.fit(X_train, y_train)
            clf = prune_duplicate_leaves(clf)
    
            if clf.tree_.feature[0] == -2:
                continue
    
            clfs.append(clf)
            pbar.update(1)

    clfs = remove_duplicates(clfs)
    joblib.dump(clfs, os.path.join(target_dir, f'{dataset_name}.joblib'))

    
    
        

    
    
    
