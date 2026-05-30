import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from ttvae.positional import SinusoidalPositionalEncoder, AbsoluteTreeEncoder, RelativeTreeEncoder
import math

class MHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by n_heads'

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads 

        self.W_q = nn.Linear(d_model, d_model, bias=False) # queries
        self.W_k = nn.Linear(d_model, d_model, bias=False) # keys
        self.W_v = nn.Linear(d_model, d_model, bias=False) # values
        self.W_o = nn.Linear(d_model, d_model, bias=False) 

    def _build_causal_mask(self, S_q, S_k_total, past_len, device, dtype):
        q_pos = torch.arange(S_q, device=device).unsqueeze(1) + past_len      # (S_q, 1)
        k_pos = torch.arange(S_k_total, device=device).unsqueeze(0)           # (1, S_k_total)
        mask = k_pos > q_pos  # True dove va mascherato
        causal_mask = torch.zeros((S_q, S_k_total), device=device, dtype=dtype)
        causal_mask = causal_mask.masked_fill(mask, float("-inf"))
        return causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S_q, S_k_total)

    def forward(self, 
                q, 
                k,
                v,
                mask=None, 
                is_causal=False, 
                enable_flash=False, 
                enable_gqa=False,
                use_cache=False,
                kv_cache=None):

        B, S_q, D = q.shape 
        _, S_k, _ = k.shape
        _, S_v, _ = v.shape

        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        
        Q = Q.view(B, S_q, self.num_heads, self.d_k).transpose(1, 2) 
        K = K.view(B, S_k, self.num_heads, self.d_k).transpose(1, 2) 
        V = V.view(B, S_v, self.num_heads, self.d_k).transpose(1, 2)

        # KV-Cache
        past_len = 0
        if use_cache and kv_cache is not None and len(kv_cache) == 2:
            past_k, past_v = kv_cache
            if past_k is not None and past_v is not None and past_k.numel() > 0:
                past_len = past_k.size(2)
                K = torch.cat((past_k, K), dim=2)
                V = torch.cat((past_v, V), dim=2)

        S_k_total = K.size(2)
        present_kv = (K, V) if use_cache else None

        if enable_flash and mask is None: 
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                
                output = F.scaled_dot_product_attention(Q, 
                                                        K, 
                                                        V, 
                                                        attn_mask=mask, 
                                                        dropout_p=0.0, 
                                                        is_causal=is_causal, 
                                                        enable_gqa=enable_gqa)
                weights = None

        else: # manual computation
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if is_causal:
                causal_mask = self._build_causal_mask(
                    S_q=S_q,
                    S_k_total=S_k_total,
                    past_len=past_len,
                    device=q.device,
                    dtype=scores.dtype,
                )
                scores = scores + causal_mask 
            if mask is not None:
                scores = scores + mask 
                
            weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(weights, V)
            
            
        output = output.transpose(1, 2).contiguous().view(B, S_q, D)
        output = self.W_o(output)

        return output, weights, present_kv 

class TransformerBlock(nn.Module): 
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.mhsa = MHA(d_model, num_heads) 
        self.norm1 = nn.LayerNorm(d_model) 
        self.norm2 = nn.LayerNorm(d_model) 
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.ReLU(),
            nn.Linear(d_ff, d_model) 
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x, 
                mask=None,
                is_causal=False,
                enable_flash=False, 
                enable_gqa=False,
                kv_cache=None):
        
        attn_out, attn_weights, _ = self.mhsa(q=x, k=x, v=x, 
                                              mask=mask, 
                                              is_causal=is_causal, 
                                              enable_flash=enable_flash,
                                              enable_gqa=enable_gqa,
                                              kv_cache=kv_cache)
        
        x = x + self.dropout(attn_out) 
        x = self.norm1(x) 
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out) 
        x = self.norm2(x)
        return x, attn_weights

class TransformerEncoder(nn.Module):
    def __init__(self,  
                 tree_encoder,
                 max_depth,
                 d_model=512, 
                 num_heads=8, 
                 d_ff=2048,
                 num_layers=6, 
                 max_len=512, 
                 num_scales=10,
                 dropout=0.0, 
                 rel_pos_enc=False,
                 abs_pos_enc='sinusoidal'):
        super().__init__()

        self.tree_encoder = tree_encoder
        self.vocab_size = len(self.tree_encoder.id_to_token)
        self.d_model = d_model
        
        self.token_embed = nn.Embedding(self.vocab_size, d_model) 

        self.abs_pos_enc = abs_pos_enc
        if self.abs_pos_enc == 'sinusoidal':
            self.abs_pos_encoder = SinusoidalPositionalEncoder(self.d_model, max_len)
        elif self.abs_pos_enc == 'tree':
            self.abs_pos_encoder = AbsoluteTreeEncoder(num_scales=num_scales) 
        elif self.abs_pos_enc is None:
            self.abs_pos_encoder = None
        else:
            raise ValueError('Invalid positional.')

        self.rel_pos_enc = rel_pos_enc
        if rel_pos_enc:
            self.rel_pos_encoder = RelativeTreeEncoder()

        
        self.layers = nn.ModuleList([
            TransformerBlock(self.d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)

    def _make_padding_mask(self, x_tokens):
        mask = (x_tokens == self.tree_encoder.token_to_id['<PAD>'])  
        key_padding_mask = torch.zeros_like(x_tokens, dtype=torch.float32) 
        key_padding_mask = key_padding_mask.masked_fill(mask, float("-inf"))
        return key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)

    def forward(self, 
                x_tokens, 
                src_abs_encs, 
                src_rel_encs,
                padding=True,
                enable_flash=False,  
                enable_gqa=False
                ): 

        self_mask = None
        node_ids = None
        
        if padding:
            self_mask = self._make_padding_mask(x_tokens)

        if self.rel_pos_enc:
            rel_mask = self.rel_pos_encoder(src_rel_encs)
            rel_mask = rel_mask.unsqueeze(1)   # (B, 1, S, S) 
            if self_mask is None:
                self_mask = rel_mask
            else:
                self_mask = self_mask + rel_mask
         
        x = self.token_embed(x_tokens)
        if self.abs_pos_enc == 'sinusoidal': 
            x = x + self.abs_pos_encoder(x) 
        elif self.abs_pos_enc == 'tree':
            x = x + self.abs_pos_encoder(src_abs_encs)
        
        x = self.dropout(x)
        
        attn_weights = list()
        
        for layer in self.layers:
            x, attn = layer(x, 
                            mask=self_mask,
                            is_causal=False,
                            enable_flash=enable_flash,
                            enable_gqa=enable_gqa,
                            kv_cache=None)
            attn_weights.append(attn)
          
        return x, attn_weights

class TransformerDecoderBlock(nn.Module): 
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.mhsa = MHA(d_model, num_heads) # self
        self.mhca = MHA(d_model, num_heads) # cross
        
        self.norm1 = nn.LayerNorm(d_model) 
        self.norm2 = nn.LayerNorm(d_model) 
        self.norm3 = nn.LayerNorm(d_model) 
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.ReLU(),
            nn.Linear(d_ff, d_model) 
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x, 
                z, 
                self_mask=None,
                query_mask=None,
                enable_flash=False, 
                enable_gqa=False,
                use_cache=False,
                kv_cache=None):
        
        self_attn_out, self_attn_weights, present_kv = self.mhsa(q=x, k=x, v=x, 
                                                                 mask=self_mask, 
                                                                 is_causal=True, 
                                                                 enable_flash=enable_flash,
                                                                 enable_gqa=enable_gqa,
                                                                 use_cache=use_cache,
                                                                 kv_cache=kv_cache)
        
        x = x + self.dropout(self_attn_out) # residual + norm 1
        x = self.norm1(x) 

        cross_attn_out, cross_attn_weights, _ = self.mhca(q=x, k=z, v=z, 
                                                          mask=None, 
                                                          is_causal=False, 
                                                          enable_flash=enable_flash,
                                                          enable_gqa=enable_gqa,
                                                          use_cache=False,
                                                          kv_cache=None)

        if query_mask is not None:
            cross_attn_out = cross_attn_out * query_mask 

        x = x + self.dropout(cross_attn_out) 
        x = self.norm2(x) 

        ff_out = self.ff(x)
        
        x = x + self.dropout(ff_out)
        x = self.norm3(x)
        
        return x, self_attn_weights, cross_attn_weights, present_kv
    
class TransformerDecoder(nn.Module):
    def __init__(self, 
                 tree_encoder,
                 max_depth,
                 device,
                 d_model=512, 
                 num_heads=8, 
                 d_ff=2048,
                 num_layers=6, 
                 max_len=512, 
                 num_scales=10,
                 dropout=0.0, 
                 rel_pos_enc=False,
                 abs_pos_enc='sinusoidal'):
        super().__init__()


        self.tree_encoder = tree_encoder
        self.vocab_size = len(self.tree_encoder.token_to_id)
        self.num_layers = num_layers
        self.device = device

        self.d_model = d_model
        
        self.token_embed = nn.Embedding(self.vocab_size, d_model) 

        self.abs_pos_enc = abs_pos_enc
        if self.abs_pos_enc == 'sinusoidal':
            self.abs_pos_encoder = SinusoidalPositionalEncoder(self.d_model, max_len)
        elif self.abs_pos_enc == 'tree':
            self.abs_pos_encoder = AbsoluteTreeEncoder(num_scales=num_scales) 
        elif self.abs_pos_enc is None:
            self.abs_pos_encoder = None
        else:
            raise ValueError('Invalid positional.')

        self.rel_pos_enc = rel_pos_enc
        if rel_pos_enc:
            self.rel_pos_encoder = RelativeTreeEncoder()
        
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(self.d_model, num_heads, d_ff, dropout)
            for _ in range(self.num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size) 
        
    def _make_padding_mask(self, x_tokens):
        mask = (x_tokens == self.tree_encoder.token_to_id['<PAD>'])  
        key_padding_mask = torch.zeros_like(x_tokens, dtype=torch.float32)  
        key_padding_mask = key_padding_mask.masked_fill(mask, float("-inf"))
        return key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)

    def forward(self, 
                x_tokens, 
                z,
                tgt_abs_encs=None,
                tgt_rel_encs=None,
                padding=True,
                enable_flash=False, 
                enable_gqa=False, 
                use_cache=False,
                kv_caches=None,
                injection='mhca',
                return_hidden=False
                ): 

        self_mask = None
        node_ids = None

        if kv_caches is None:
            kv_caches = [None] * self.num_layers

        if padding and not use_cache: 
            self_mask = self._make_padding_mask(x_tokens) 

        if self.abs_pos_enc == 'tree' and tgt_abs_encs is None:
            outputs = self.tree_encoder.ids_to_pos_encoding(x_tokens)
            tgt_abs_encs = outputs['tgt_abs_encs'].unsqueeze(0)
            tgt_rel_encs = outputs['tgt_rel_encs'].unsqueeze(0)
            tgt_rel_encs = tgt_rel_encs.to(self.device)
            tgt_abs_encs = tgt_abs_encs.to(self.device)
                
        if self.rel_pos_enc:
            if self_mask is None:
                self_mask = self.rel_pos_encoder(tgt_rel_encs).unsqueeze(1)
            else:
                self_mask = self_mask + self.rel_pos_encoder(tgt_rel_encs).unsqueeze(1)
         
        x = self.token_embed(x_tokens)
        
        if self.abs_pos_enc == 'sinusoidal':
            if use_cache and kv_caches[0] is not None: 
                past_len = kv_caches[0][0].size(2) 
            else:
                past_len = 0
            x = x + self.abs_pos_encoder(x, start_pos=past_len) 
        elif self.abs_pos_enc == 'tree':
            x = x + self.abs_pos_encoder(tgt_abs_encs)
        
        x = self.dropout(x)
        
        self_weights = list()
        cross_weights = list()
        present_kvs = list()

        if injection == 'mhca':

            query_mask = (x_tokens != self.tree_encoder.token_to_id['<PAD>']).unsqueeze(-1).to(x.dtype)

            for i, layer in enumerate(self.layers):
                kv_cache = kv_caches[i]
                z_i = z[:, i, :].unsqueeze(1) 
                x, self_attn, cross_attn, present_kv = layer(x, 
                                                             z_i,
                                                             self_mask=self_mask,
                                                             query_mask=query_mask,
                                                             enable_flash=enable_flash,
                                                             enable_gqa=enable_gqa,
                                                             use_cache=use_cache,
                                                             kv_cache=kv_cache)
                self_weights.append(self_attn)
                cross_weights.append(cross_attn)
                present_kvs.append(present_kv)
        
        elif injection == 'input':
            raise NotImplementedError
        elif injection == 'layers':
            raise NotImplementedError
        else:
            raise NotImplementedError

        logits = self.lm_head(x)

        if return_hidden:
            return logits, x, self_weights, cross_weights, present_kvs
        return logits, present_kvs

class TTVAE(nn.Module):

    def __init__(self, 
                 tree_encoder,
                 max_depth=None,
                 d_model_encoder=512,
                 d_model_decoder=512,
                 num_heads_encoder=8,
                 num_heads_decoder=8,
                 num_layers_encoder=6,
                 num_layers_decoder=6,
                 d_ff_encoder=2048,
                 d_ff_decoder=2048,
                 dropout_encoder=0.0, 
                 dropout_decoder=0.0, 
                 abs_pos_enc='sinusoidal',
                 rel_pos_enc=False,
                 latent_dim=512,
                 max_len=512,
                 weight_tying=False,
                 num_scales=10,
                 device=None):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
            
        self.latent_dim = latent_dim
        
        self.encoder = TransformerEncoder(
            tree_encoder=tree_encoder,
            max_depth=max_depth,
            d_model=d_model_encoder,
            num_heads=num_heads_encoder,
            num_layers=num_layers_encoder,
            d_ff=d_ff_encoder,
            dropout=dropout_encoder,
            abs_pos_enc=abs_pos_enc,
            rel_pos_enc=rel_pos_enc,
            max_len=max_len,
            num_scales=num_scales
        )

        self.to_mu = nn.Linear(d_model_encoder, latent_dim) 
        self.to_logvar = nn.Linear(d_model_encoder, latent_dim)
        self.z_proj = nn.Linear(latent_dim, num_layers_decoder * d_model_decoder) 

        self.decoder = TransformerDecoder(
            tree_encoder=tree_encoder,
            max_depth=max_depth,
            device=device,
            d_model=d_model_decoder,
            num_heads=num_heads_decoder,
            num_layers=num_layers_decoder,
            d_ff=d_ff_decoder,
            dropout=dropout_decoder,
            abs_pos_enc=abs_pos_enc,
            rel_pos_enc=rel_pos_enc,
            max_len=max_len,
            num_scales=num_scales
        )

        if weight_tying and d_model_encoder == d_model_decoder:
            self.decoder.token_embed.weight = self.encoder.token_embed.weight

        self.to(self.device)

    def _reparameterize(self, 
                        mu, 
                        logvar):
        if self.training:
            eps = torch.randn_like(mu) 
            return mu + eps * torch.exp(0.5 * logvar) 
        return mu
            
    def encode(self, 
               x_tokens, 
               src_abs_encs, 
               src_rel_encs,
               padding=True, 
               strategy='cls'): 
        
        h, attn_weights = self.encoder(x_tokens, src_abs_encs, src_rel_encs, padding=padding)

        if strategy == 'cls':
            h = h[:, 0, :] 
        elif strategy == 'mean_pooling':
            if padding: 
                mask = (x_tokens != self.tree_encoder.token_to_id['<PAD>']).unsqueeze(-1)
                h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                h = h.mean(dim=1)
        
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        z = self._reparameterize(mu, logvar)

        return z, mu, logvar, h

    def forward(self,  
                tgt, 
                src=None,
                z=None,
                src_abs_encs=None, 
                src_rel_encs=None,
                tgt_abs_encs=None,
                tgt_rel_encs=None,
                return_hidden=False, 
                injection='mhca',
                use_cache=False,
                kv_caches=None):

        mu = None
        logvar = None
        h = None

        if z is None:
            assert src is not None, 'null src'
        
            z, mu, logvar, h = self.encode(src, src_abs_encs, src_rel_encs)
            B, _ = z.shape 
            z = self.z_proj(z).view(B, self.decoder.num_layers, self.decoder.d_model) 

        if return_hidden:
            logits, x, self_weights, cross_weights, present_kvs = self.decoder(tgt, 
                                                                               z, 
                                                                               tgt_abs_encs=tgt_abs_encs,
                                                                               tgt_rel_encs=tgt_rel_encs,
                                                                               injection=injection, 
                                                                               return_hidden=return_hidden,
                                                                               kv_caches=kv_caches,
                                                                               use_cache=use_cache)
            
            return logits, x, self_weights, cross_weights, present_kvs, z, mu, logvar, h

        logits, present_kvs = self.decoder(tgt, 
                                           z, 
                                           tgt_abs_encs=tgt_abs_encs,
                                           tgt_rel_encs=tgt_rel_encs,
                                           injection=injection, 
                                           return_hidden=return_hidden, 
                                           kv_caches=kv_caches,
                                           use_cache=use_cache)
        
        return logits, mu, logvar, present_kvs

    def _sample_next_token(self, next_logits, do_sample=True, temperature=1.0, top_k=None, top_p=None):
        next_logits = next_logits / max(temperature, 1e-8)

        if top_k is not None:
            k = min(top_k, next_logits.size(-1))
            values, _ = torch.topk(next_logits, k)
            min_values = values[:, -1].unsqueeze(-1)
            next_logits = torch.where(
                next_logits < min_values,
                torch.full_like(next_logits, float("-inf")),
                next_logits,
            )

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_mask = cumulative_probs > top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False

            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            next_logits = torch.full_like(next_logits, float("-inf"))
            next_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

        probs = torch.softmax(next_logits, dim=-1)

        if do_sample:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

        return next_token

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        z=None,
        src=None,
        max_new_tokens=50,
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
        eos_token_id=None,
        injection="mhca",
        use_cache=True,
        z_mean=None,
        z_std=None,
        return_z=False
    ):
        self.eval()
    
        if eos_token_id is None:
            eos_token_id = self.decoder.tree_encoder.token_to_id['<EOS>']
    
        if input_ids is None:
            B = 1
            input_ids = torch.full(
                (B, 1),
                self.decoder.tree_encoder.token_to_id['<BOS>'],
                dtype=torch.long,
                device=self.device
            )
    
        if z is None:
            if src:
                z, _, _, _ = self.encode(src)
                B, _ = z.shape
            else: # sample z
                z = torch.randn(B, self.latent_dim, device=self.device)
        
        if z_mean is not None and z_std is not None:
            z = z * z_std + z_mean 
            
        B = 1
        z = self.z_proj(z).view(B, self.decoder.num_layers, self.decoder.d_model)
            
        generated = input_ids
    
        if use_cache:
            kv_caches = [None] * self.decoder.num_layers
    
            tgt_abs_encs = None
            if self.decoder.abs_pos_enc == "tree":
                outputs = self.decoder.tree_encoder.ids_to_pos_encoding(generated)
                tgt_abs_encs = outputs["tgt_abs_encs"].unsqueeze(0).to(self.device)
    
            logits, mu, logvar, kv_caches = self.forward(
                tgt=generated,
                z=z,
                tgt_abs_encs=tgt_abs_encs,
                injection=injection,
                kv_caches=kv_caches,
                use_cache=True,
            )
    
            finished = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)
    
            for _ in range(max_new_tokens):
                next_logits = logits[:, -1, :]
                next_token = self._sample_next_token(
                    next_logits,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
    
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )
    
                generated = torch.cat([generated, next_token], dim=1)
                finished = finished | (next_token.squeeze(1) == eos_token_id)
    
                if finished.all():
                    break
    
                tgt_abs_encs_next = None
                if self.decoder.abs_pos_enc == "tree":
                    outputs = self.decoder.tree_encoder.ids_to_pos_encoding(generated)
                    tgt_abs_encs_full = outputs["tgt_abs_encs"].unsqueeze(0).to(self.device)
                    tgt_abs_encs_next = tgt_abs_encs_full[:, -1:, :]
    
                logits, mu, logvar, kv_caches = self.forward(
                    tgt=next_token,  
                    z=z,
                    tgt_abs_encs=tgt_abs_encs_next,
                    injection=injection,
                    kv_caches=kv_caches,
                    use_cache=True,
                )
    
        else:
            finished = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)
    
            for _ in range(max_new_tokens):
                
                logits, mu, logvar, _ = self.forward(
                    tgt=generated,
                    z=z,
                    injection=injection,
                    kv_caches=None,
                    use_cache=False,
                )
    
                next_logits = logits[:, -1, :]
                
                next_token = self._sample_next_token(
                    next_logits,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
    
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )
    
                generated = torch.cat([generated, next_token], dim=1)
                finished = finished | (next_token.squeeze(1) == eos_token_id)
    
                if finished.all():
                    break
        if return_z:
            return generated, z
        return generated

        
        
    
    
    