import torch
import torch.nn as nn
from torch.functional import F
import numpy as np
import math
import time
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from collections import deque
from sklearn.metrics import log_loss, f1_score
import os
import copy


class EarlyStopping:
    def __init__(self, 
                 monitor='val_loss', 
                 min_delta=0.0, 
                 patience=0, 
                 save_path='best_model.pt'):
        
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.save_path = save_path
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False
        self.best_model_state = None

    def save(self, epoch, optimizer, history, config):

        last_ckpt = {
                'epoch': self.best_epoch,
                'monitor': self.monitor,
                'best_score': self.best_score,
                'model_state_dict': self.best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'config': config,
                }    
        
        torch.save(last_ckpt, self.save_path)
        
    def __call__(self, 
                 epoch, 
                 current_loss, 
                 current_recon,
                 current_kl,
                 max_kl, 
                 beta, 
                 model, 
                 optimizer, 
                 history, 
                 config):

        if current_kl < max_kl:
            self.early_stop = True
            return

        if self.best_score is None: # first epoch
            self.best_score = current_loss
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.save(epoch, optimizer, history, config)
            self.epochs_no_improve = 0
            
        elif current_loss < (self.best_score - self.min_delta): # improvement
            self.best_score = current_loss
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.save(epoch, optimizer, history, config)
            self.epochs_no_improve = 0
            
        else: # no improvement or max_kl 
            self.epochs_no_improve += 1
            self.save(epoch, optimizer, history, config)
            if self.epochs_no_improve > self.patience:
                self.early_stop = True
                    

    def load_best_model(self, model):
        """Load the best model from disk."""
        ckpt = torch.load(self.save_path)
        model.load_state_dict(ckpt["model_state_dict"])


class TTVAETrainer:
    def __init__(self,
                 model,
                 train_trees, 
                 val_trees, 
                 es_trees,
                 batch_size,
                 pad_token_id,
                 cls_token_id,
                 bos_token_id,
                 eos_token_id,
                 unk_token_id,
                 shuffle=True,
                 seed=42):

        self.model = model
        self.device = model.device

        self.train_trees = train_trees
        self.val_trees = val_trees
        self.es_trees = es_trees
        self.batch_size = batch_size 
    
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id
    
        g = torch.Generator()
        g.manual_seed(seed)
    
        self.train_loader = DataLoader(
                self.train_trees, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                collate_fn=self.collate_fn, 
                generator=g
            )

        self.val_loader = DataLoader(
            self.val_trees, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn
        )
    
        self.es_loader = DataLoader(
                self.es_trees, 
                batch_size=batch_size, 
                shuffle=False, 
                collate_fn=self.collate_fn
            )

    def _pad_batch(self, seqs):
        max_len = max(x.size(0) for x in seqs) # max len in batch
        out = torch.full(
            (len(seqs), max_len),
            self.pad_token_id,
            dtype=torch.long
        )
        for i, x in enumerate(seqs):
            out[i, :x.size(0)] = x
        return out

    def _pad_abs_encs(self, abs_encs, x_tokens):
        diff = x_tokens.shape[1] - abs_encs.shape[0]
        if diff > 0:
            pad = torch.zeros(diff, abs_encs.shape[1], device=abs_encs.device, dtype=abs_encs.dtype)
            abs_encs = torch.cat([abs_encs, pad], dim=0)
        return abs_encs

    def _pad_abs_encs_new(self, batch, target_len, kind='src'):
        abs_encs = [sample[f'{kind}_abs_encs'] for sample in batch]
    
        max_depth = 0
        for sample_abs in abs_encs:
            for pos in sample_abs:
                if len(pos) > max_depth:
                    max_depth = len(pos)
    
        out = torch.zeros(
            len(abs_encs),
            target_len,
            max_depth,
            dtype=torch.long
        )
    
        for i, sample_abs in enumerate(abs_encs):
            cur_len = min(len(sample_abs), target_len)
    
            for j in range(cur_len):
                pos = sample_abs[j]
                if torch.is_tensor(pos):
                    out[i, j, :len(pos)] = pos.to(dtype=torch.long)
                else:
                    out[i, j, :len(pos)] = torch.tensor(pos, dtype=torch.long)
    
        return out

    def _pad_rel_encs(self, rel_encs, x_tokens):
        target_len = x_tokens.shape[1]
        pad_value = 1
    
        out = torch.full(
            (target_len, target_len),
            pad_value,
            device=rel_encs.device,
            dtype=rel_encs.dtype
        )
    
        cur_len = min(rel_encs.shape[0], target_len)
        out[:cur_len, :cur_len] = rel_encs[:cur_len, :cur_len]
    
        return out

    def collate_fn(self, batch):
        
        src_seqs = [x['src'] for x in batch]
        tgt_in_seqs = [x['tgt'][:-1] for x in batch]
        tgt_out_seqs = [x['tgt'][1:] for x in batch]
        
        src = self._pad_batch(src_seqs) 
        tgt_in = self._pad_batch(tgt_in_seqs)
        tgt_out = self._pad_batch(tgt_out_seqs)

        if 'src_abs_encs' in batch[0]: 
            src_abs_encs = [self._pad_abs_encs(x['src_abs_encs'], src) for x in batch]
            src_abs_encs = torch.stack(src_abs_encs, dim=0)
        else:
            src_abs_encs = None
        if 'tgt_abs_encs' in batch[0]:
            tgt_abs_encs = [self._pad_abs_encs(x['tgt_abs_encs'][:-1], tgt_in) for x in batch]
            tgt_abs_encs = torch.stack(tgt_abs_encs, dim=0)
        else:
            tgt_abs_encs = [None for x in batch]
        if 'src_rel_encs' in batch[0]:
            src_rel_encs = [self._pad_rel_encs(x['src_rel_encs'], src) for x in batch]
            src_rel_encs = torch.stack(src_rel_encs, dim=0)
        else:
            src_rel_encs = [None for x in batch]
        if 'tgt_rel_encs' in batch[0]:
            tgt_rel_encs = [self._pad_rel_encs(x['tgt_rel_encs'][:-1, :-1], tgt_in) for x in batch]
            tgt_rel_encs = torch.stack(tgt_rel_encs, dim=0)
        else:
            tgt_rel_encs = [None for x in batch]

        return {
            'src': src,
            'tgt_in': tgt_in,
            'tgt_out': tgt_out,
            'src_abs_encs':src_abs_encs,
            'src_rel_encs':src_rel_encs,
            'tgt_abs_encs':tgt_abs_encs,
            'tgt_rel_encs':tgt_rel_encs,
        }

    def evaluate(self, beta=1.0, split='es'):
        
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        if split == 'es':
            loader = self.es_loader
        elif split == 'val':
            loader = self.val_loader

        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.model.device) if not isinstance(v, list) else v
                    for k, v in batch.items()
                }
                
                logits, mu, logvar, _ = self.model(src=batch['src'], 
                                                   tgt=batch['tgt_in'],
                                                   src_abs_encs=batch['src_abs_encs'], 
                                                   src_rel_encs=batch['src_rel_encs'],
                                                   tgt_abs_encs=batch['tgt_abs_encs'],
                                                   tgt_rel_encs=batch['tgt_rel_encs'])  # TTVAE forward

                loss, recon, kl = self.vae_loss(
                    logits=logits,
                    tgt_out=batch['tgt_out'],
                    mu=mu,
                    logvar=logvar,
                    beta=beta
                )

                total_loss += loss.item()
                total_recon += recon.item()
                total_kl += kl.item()

        n = len(loader)
        avg_loss = total_loss / n
        avg_recon = total_recon / n
        avg_kl = total_kl / n

        return avg_loss, avg_recon, avg_kl

    def _beta_schedule(
        self,
        step,
        total_steps,
        schedule,      
        beta_start,
        beta_end,
        beta_warmup_frac,
        cyclic=False,
        n_cycles=4,
        k_sigmoid=10.0
    ):
    
        if cyclic:
            cycle_len = max(1, total_steps // n_cycles)
            cycle_pos = (step % cycle_len) / cycle_len
    
            ramp_frac = beta_warmup_frac if beta_warmup_frac is not None else 0.5
    
            if cycle_pos >= ramp_frac:
                return beta_end
    
            t = cycle_pos / ramp_frac
    
        else:
            if beta_warmup_frac is None:
                return beta_end
    
            warmup_steps = max(1, int(total_steps * beta_warmup_frac))
    
            if step >= warmup_steps:
                return beta_end
    
            t = step / warmup_steps
    
        if schedule == 'linear':
            s = t
    
        elif schedule == 'sigmoid':
            s = 1.0 / (1.0 + math.exp(-k_sigmoid * (t - 0.5)))
            s0 = 1.0 / (1.0 + math.exp(-k_sigmoid * (0.0 - 0.5)))
            s1 = 1.0 / (1.0 + math.exp(-k_sigmoid * (1.0 - 0.5)))
            s = (s - s0) / (s1 - s0)
    
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
        return beta_start + s * (beta_end - beta_start)

    def vae_loss(self, logits, tgt_out, mu, logvar, beta, free_bits=0.0):

        ignore = self.pad_token_id if self.pad_token_id is not None else -100

        recon_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=ignore,
            reduction='sum' 
        )

        if self.pad_token_id is not None:
            num_tokens = (tgt_out != self.pad_token_id).sum() 
        else:
            num_tokens = torch.tensor(tgt_out.numel(), device=tgt_out.device)
    
        num_tokens = num_tokens.clamp(min=1) 
        recon = recon_sum / num_tokens 

        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        kl_per_seq = kl_per_dim.sum(dim=-1)
        kl_sum = kl_per_seq.sum()  
        kl = kl_sum / num_tokens # per token
    
        loss = recon + beta * kl
        
        return loss, recon, kl 

    def mask_context(
        self,
        tgt_in: torch.Tensor,
        mask_prob: float,
    ):
        
        if mask_prob <= 0:
            return tgt_in
    
        out = tgt_in.clone()
    
        rand = torch.rand(out.shape, device=out.device)
        mask = rand < mask_prob
    
        mask &= (out != self.bos_token_id)
        mask &= (out != self.pad_token_id)
    
        out[mask] = self.unk_token_id
        return out

    def fit(
        self,
        lr,
        epochs,
        early_stopping=True,
        start_early_stop = 0,
        patience=2,
        min_delta=0.01,
        save_path="best_model.pt",
        beta_config = {
            'schedule':'linear', 
            'beta_start':0.0,
            'beta_end':1.0,
            'beta_warmup_frac':0.3, 
            'k_sigmoid':10.0,
            'cyclic':False,
            'n_cycles':None,
        },
        beta_epoch=0.0,
        beta_eval = 1.0,
        lr_start=0.0,
        lr_end=1e-1,
        lr_warmup_frac=0.3,
        free_bits=0.0,
        mask_prob=0.0,
        max_kl=0.1,
        min_delta_recon=0.05,
        patience_recon=1.0
        
    ):

        optimizer = AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

        history = {
                'train_loss_epoch': list(),
                'train_loss_step': list(),
                'betas':list(),
                'train_beta_loss_epoch': list(),
                'train_beta_loss_step': list(),
                'val_loss_epoch': list(),
                'train_recon_epoch': list(),
                'train_recon_step':list(),
                'val_recon_epoch': list(),
                'train_kl_epoch':list(),
                'train_kl_step':list(),
                'val_kl_epoch':list(),
                'time_per_epoch':list(),
                'global_step':None
                 }

        config = dict()

        es = None
        if early_stopping:
            es = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                min_delta=min_delta,
                save_path=save_path,
            )
    
        total_steps = epochs * len(self.train_loader)
        global_step = 0
        best_avg_recon = None
    
        for epoch in tqdm(range(epochs)):
            
            start_time = time.time()
            
            self.model.train()
    
            total_loss = 0.0
            total_recon = 0.0
            total_kl = 0.0
            total_elbo = 0.0 
    
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch = {
                    k: v.to(self.model.device) if not isinstance(v, list) else v
                    for k, v in batch.items()
                }
                
                src, tgt_in, tgt_out, src_abs_encs, src_rel_encs, tgt_abs_encs, tgt_rel_encs = batch['src'], batch['tgt_in'], batch['tgt_out'], batch['src_abs_encs'], batch['src_rel_encs'], batch['tgt_abs_encs'], batch['tgt_rel_encs']

                if mask_prob:
                    tgt_in = self.mask_context(tgt_in, mask_prob)

                if epoch >= beta_epoch:

                    beta = self._beta_schedule(
                        step=global_step,
                        total_steps=total_steps,
                        **beta_config
                    )

                    global_step += 1
                else:
                    beta = beta_config['beta_start']

                logits, mu, logvar, _ = self.model(src=src, 
                                                   tgt=tgt_in, 
                                                   src_abs_encs=src_abs_encs, 
                                                   src_rel_encs=src_rel_encs,
                                                   tgt_abs_encs=tgt_abs_encs,
                                                   tgt_rel_encs=tgt_rel_encs)
    
                loss, recon, kl = self.vae_loss(
                    logits=logits,
                    tgt_out=tgt_out,
                    mu=mu,
                    logvar=logvar,
                    beta=beta,
                    free_bits=free_bits
                )
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon += recon.item()
                total_kl += kl.item()

                history['train_loss_step'].append(-(recon.item() + kl.item()))
                history['train_beta_loss_step'].append(loss.item())
                history['train_recon_step'].append(recon.item())
                history['train_kl_step'].append(kl.item())
                history['betas'].append(beta)
                

            n = len(self.train_loader)
            avg_loss = total_loss / n
            avg_recon = total_recon / n
            avg_kl = total_kl / n

            val_loss, val_recon, val_kl = self.evaluate()

            history['train_loss_epoch'].append(-(avg_recon + avg_kl))
            history['train_beta_loss_epoch'].append(avg_loss)
            history['train_recon_epoch'].append(avg_recon)
            history['train_kl_epoch'].append(avg_kl)
            history['val_loss_epoch'].append(val_loss)
            history['val_recon_epoch'].append(val_recon)
            history['val_kl_epoch'].append(val_kl)

            end_time = time.time()
            total_time = end_time - start_time
            history['time_per_epoch'].append(total_time)   
    
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Train Recon: {avg_recon:.4f} | Train KL: {avg_kl:.4f} | Beta: {beta:.4f} | Val Loss: {val_loss:.4f} | Val Recon: {val_recon:.4f} | Val KL: {val_kl:.4f}"
            )

            if early_stopping and epoch >= start_early_stop:
                
                es(epoch=epoch, 
                   current_loss=val_loss, 
                   current_recon=val_recon, 
                   current_kl=val_kl, 
                   max_kl=max_kl, 
                   beta=beta, 
                   model=self.model, 
                   optimizer=optimizer, 
                   history=history, 
                   config=config)
                
                if es.early_stop:
                    print(f"Early stopping at epoch {epoch}.")
                    break
   
        if early_stopping and epoch >= start_early_stop:
            if os.path.isfile(save_path):
                es.load_best_model(self.model)
                print(f"Best model loaded from {save_path} with validation loss {es.best_score:.4f}")
            else:
                print(f"No checkpoint found at {save_path}")
                
        else: # save at last epoch
            last_ckpt = {
                'epoch': epoch, 
                'monitor': None,
                'best_score': None,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'config': config,
                }    
            torch.save(last_ckpt, save_path)
            

        
        
