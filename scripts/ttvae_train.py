import torch
from tqdm import tqdm
import pandas as pd
import json
import argparse
from ttvae.tree_encoding import TreeEncoder
from ttvae.trainer import TTVAETrainer
from ttvae.model import TTVAE
import joblib
import os
from sklearn.model_selection import train_test_split
import Levenshtein
from itertools import combinations

def save(dataset_name, result):
    
    results_path = f'trained_ttvae_models/{dataset_name}_metrics.csv'

    columns = [
        'dataset',
        'val_recon',
        'val_kl',
        'val_loss',
        'validity',
        'novelty',
        'diversity'
    ]

    if os.path.exists(results_path):
        results = pd.read_csv(results_path)
    else:
        results = pd.DataFrame(columns=columns)

    new_data = {
        'dataset': dataset_name,
        'val_recon': result['recon'],
        'val_kl': result['kl'],
        'val_loss': result['loss'],
        'validity': result['validity'],
        'novelty': result['novelty'],
        'diversity': result['diversity']
    }

    if dataset_name in results['dataset'].values:
        for key, value in new_data.items():
            results.loc[results['dataset'] == dataset_name, key] = value
    else:
        results = pd.concat(
            [results, pd.DataFrame([new_data])],
            ignore_index=True
        )

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results.to_csv(results_path, index=False)

def split_seqs(seqs, train_size):

    train_seqs, test_seqs = train_test_split(seqs, 
                                             test_size=20000, 
                                             random_state=42)
    
    train_seqs, val_seqs = train_test_split(train_seqs, 
                                            test_size=10000, 
                                            random_state=42)

    train_seqs, es_seqs = train_test_split(train_seqs, 
                                            test_size=10000, 
                                            random_state=42)

    train_seqs = train_seqs[:train_size]

    return train_seqs, test_seqs, val_seqs, es_seqs

if __name__ == '__main__':

    with open('../config/ttvae_train_config.json', 'r') as file:
         config = json.load(file)

    device = config['device']
    train_size = config['train_size']
    dataset_name = config['dataset_name']
    precision = config['precision']
    target_dir = config['target_dir']

    data = pd.read_csv(os.path.join('../data', f'{dataset_name}.csv'))
    train = data[data.split=='train'].drop('split', axis=1)
    X_train, y_train = train.drop('target', axis=1).round(precision), train['target']

    trees = joblib.load(os.path.join('../experiments/tokenized_trees', f'{dataset_name}.joblib'))
    trees = [tree[0] for tree in trees]
    train_trees, test_trees, val_trees, es_trees = split_seqs(trees, train_size=train_size)

    te = TreeEncoder(X=X_train, 
                     y=y_train,
                     tokenization='threshold', # default
                     precision=precision) 
    
    model = TTVAE(tree_encoder=te,
                  max_depth=config['max_depth'],
                  d_model_encoder=config['d_model_encoder'],
                  d_model_decoder=config['d_model_decoder'],
                  num_heads_encoder=config['num_heads_encoder'],
                  num_heads_decoder=config['num_heads_decoder'],
                  num_layers_encoder=config['num_layers_encoder'],
                  num_layers_decoder=config['num_layers_decoder'],
                  d_ff_encoder=config['d_ff_encoder'],
                  d_ff_decoder=config['d_ff_decoder'],
                  dropout_encoder=config['dropout_encoder'], 
                  dropout_decoder=config['dropout_decoder'], 
                  abs_pos_enc=config['abs_pos_enc'],
                  rel_pos_enc=config['rel_pos_enc'],
                  latent_dim=config['latent_dim'], 
                  max_len=config['max_len'],
                  weight_tying=False,
                  device=device
            )
    
    trainer = TTVAETrainer(model=model, 
                          train_trees=train_trees, 
                          val_trees=val_trees,
                          es_trees=es_trees,
                          batch_size=config['batch_size'],
                          pad_token_id=te.token_to_id['<PAD>'],
                          cls_token_id=te.token_to_id['<CLS>'],
                          bos_token_id=te.token_to_id['<BOS>'],
                          eos_token_id=te.token_to_id['<EOS>'],
                          unk_token_id=te.token_to_id['<UNK>'],
                          shuffle=True, 
                          seed=42)

    trainer.fit(lr=config['lr'],
                epochs=config['epochs'],
                early_stopping=config['early_stopping'],
                patience=config['patience'],
                min_delta=config['min_delta'],
                beta_config=config['beta_config'],
                start_early_stop=config['beta_epoch'], 
                free_bits=config['free_bits'],
                beta_epoch=config['beta_epoch'],
                mask_prob=config['mask_prob'],
                max_kl=config['max_kl'],
                min_delta_recon=config['min_delta_recon'],
                patience_recon=config['patience_recon'],
                save_path=f'{target_dir}/{dataset_name}.pt')

    result = dict()
    result['loss'], result['recon'], result['kl'] = trainer.evaluate(split='val')

    n_iter = 1000
    valid_seqs = list()
    invalid_seqs = list()
    clfs = list()
    
    for i in tqdm(range(n_iter)):
        try:
            gen = model.generate(do_sample=False, use_cache=False).tolist()[0]; 
            clf = model.decoder.tree_encoder.decode_tree(gen)
            clfs.append(clf)
            valid_seqs.append(gen)
        except:
            continue

    result['validity'] = len(valid_seqs) / n_iter

    unique_train_seqs = set(tuple(x['tgt'].tolist()) for x in train_trees)
    novel_seqs = [x for x in valid_seqs if tuple(x) not in unique_train_seqs]
    
    if len(valid_seqs) == 0:
        novelty = 0.0
    else:
        novelty = len(novel_seqs) / len(valid_seqs)
    
    result['novelty'] = novelty
    
    if len(valid_seqs) < 2:
        diversity = 0.0
    else:
        distances = list()
        for a, b in combinations(valid_seqs, 2):
            distances.append(Levenshtein.distance(a, b) / max(len(a), len(b))) 
        diversity = sum(distances) / len(distances)
    
    result['diversity'] = diversity

    save(dataset_name, result)
            
