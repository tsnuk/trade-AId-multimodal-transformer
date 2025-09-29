"""model.py

Neural network classes and model architecture for the multimodal transformer system.

Contains all PyTorch model classes:
- Head (single attention head)
- MultiHeadAttention (multi-head self-attention)
- CrossAttention (cross-modal attention)
- FeedForward (feedforward layer)
- MultimodalBlock (transformer block)
- FixedEmbedding (custom embedding layer - experimental, not currently used)
- MultimodalPreBlock (input processing)
- MultimodalPostBlock (output processing)
- MultimodalTransformer (main model)

Extracted from mm_final_4.py for better code organization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Import configuration utilities
from config_utils import (
    _get_block_size, _get_n_embd, _get_n_head, _get_n_layer, _get_dropout
)


class Head(nn.Module):
    """Single attention head within a multi-head attention mechanism."""

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Sequential(
            nn.Linear(_get_n_embd(), head_size // 2),
            nn.Tanh(),
            nn.Linear(head_size // 2, head_size, bias=False)
        )
        self.query = nn.Sequential(
            nn.Linear(_get_n_embd(), head_size // 2),
            nn.Tanh(),
            nn.Linear(head_size // 2, head_size, bias=False)
        )
        self.value = nn.Sequential(
            nn.Linear(_get_n_embd(), head_size // 2),
            nn.Tanh(),
            nn.Linear(head_size // 2, head_size, bias=False)
        )



        # Lower triangular matrix for causal masking
        self.register_buffer('tril', torch.tril(torch.ones(_get_block_size(), _get_block_size())))

        self.dropout = nn.Dropout(_get_dropout())


    def forward(self, x):
        Ba, Bl, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Scaled dot-product attention
        aff = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        # Apply causal mask to prevent attending to future tokens
        aff = aff.masked_fill(self.tril[:Bl, :Bl] == 0, float('-inf'))
        aff = F.softmax(aff, dim=-1)
        aff = self.dropout(aff)
        # Weighted aggregation of values
        v = self.value(x)
        att_out = aff @ v
        return att_out


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism for transformer blocks."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for cur_head in range(num_heads)])
        self.proj = nn.Sequential(
            nn.Linear(head_size * num_heads, _get_n_embd()//2),
            nn.Tanh(),
            nn.Linear(_get_n_embd()//2, _get_n_embd())
        )
        self.dropout = nn.Dropout(_get_dropout())

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class CrossAttention(nn.Module):
    """Cross-modal attention mechanism for attending across different modalities."""

    def __init__(self, num_heads, head_size, num_kv_modalities):
        super().__init__()
        self.num_kv_modalities = num_kv_modalities
        self.heads = nn.ModuleList([self.Head(head_size, num_kv_modalities) for _ in range(num_heads)])
        self.proj = nn.Sequential(
            nn.Linear(head_size * num_heads, _get_n_embd() // 2),
            nn.Tanh(),
            nn.Linear(_get_n_embd() // 2, _get_n_embd())
        )
        self.dropout = nn.Dropout(_get_dropout())

    def forward(self, query_x, key_value_x_list):
        """
        Args:
            query_x: The querying modality (batch_size, block_size, n_embd)
            key_value_x_list: List of key-value modality tensors
        """
        out = torch.cat([head(query_x, key_value_x_list) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

    class Head(nn.Module):
        """Cross-attention head for processing queries from one modality against keys/values from others."""

        def __init__(self, head_size, num_kv_modalities):
            super().__init__()
            self.num_kv_modalities = num_kv_modalities
            self.head_size = head_size

            self.query = nn.Linear(_get_n_embd(), head_size, bias=False)
            # Separate key and value projections for each key/value modality
            self.kv_projections = nn.ModuleList([
                nn.Linear(_get_n_embd(), 2 * head_size, bias=False) for _ in range(num_kv_modalities)
            ])
            # Causal mask for preventing attention to future positions
            self.register_buffer('tril', torch.tril(torch.ones(_get_block_size(), _get_block_size())))
            self.dropout = nn.Dropout(_get_dropout())

        def forward(self, query_x, key_value_x_list):
            Ba, Bl, C = query_x.shape
            q = self.query(query_x)

            all_kv_outputs = []
            for i, kv_x in enumerate(key_value_x_list):
                # Project KV modality to keys and values
                kv_projection = self.kv_projections[i](kv_x)
                k, v = kv_projection.split(self.head_size, dim=-1)

                # Calculate attention affinities with scaling
                aff = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
                # Apply causal mask
                aff = aff.masked_fill(self.tril[:Bl, :Bl] == 0, float('-inf'))
                aff = F.softmax(aff, dim=-1)
                aff = self.dropout(aff)

                # Apply attention weights to values
                kv_output = aff @ v
                all_kv_outputs.append(kv_output)

            # Combine outputs from all KV modalities
            combined_output = sum(all_kv_outputs)
            return combined_output


class FeedForward(nn.Module):
    """Feedforward network with expansion and contraction."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_get_n_embd(), 4 * _get_n_embd()),
            nn.ReLU(),
            nn.Linear(4 * _get_n_embd(), _get_n_embd()),
            nn.Dropout(_get_dropout()),
        )

    def forward(self, x):
        return self.net(x)


class MultimodalBlock(nn.Module):
    """Single transformer block with self-attention, cross-attention, and feed-forward layers."""

    def __init__(self, n_embd, n_head, num_modalities, all_modality_params):
        super().__init__()
        self.num_modalities = num_modalities
        self.all_modality_params = all_modality_params

        head_size = _get_n_embd() // _get_n_head()
        self.sa_layers = nn.ModuleList([MultiHeadAttention(_get_n_head(), head_size) for _ in range(num_modalities)])
        self.ffwd_layers = nn.ModuleList([FeedForward() for _ in range(num_modalities)])
        self.ln1_layers = nn.ModuleList([nn.LayerNorm(_get_n_embd()) for _ in range(num_modalities)])
        self.ln2_layers = nn.ModuleList([nn.LayerNorm(_get_n_embd()) for _ in range(num_modalities)])

        # Initialize cross-attention layers based on modality configuration
        self.cross_attention_layers = nn.ModuleList()
        for i in range(num_modalities):
            modality_params = all_modality_params[i]
            cross_attention_enabled = modality_params[8]  # Index 8 is cross_attention status
            if cross_attention_enabled:
                other_modality_indices = [j for j in range(num_modalities) if j != i]
                num_kv_modalities = len(other_modality_indices)
                self.cross_attention_layers.append(CrossAttention(_get_n_head(), head_size, num_kv_modalities))
            else:
                self.cross_attention_layers.append(None)

        # Layer norms for cross-attention
        self.ln_cross_layers = nn.ModuleList()
        for i in range(num_modalities):
            modality_params = all_modality_params[i]
            cross_attention_enabled = modality_params[8]
            if cross_attention_enabled:
                self.ln_cross_layers.append(nn.LayerNorm(_get_n_embd()))
            else:
                self.ln_cross_layers.append(None)

    def forward(self, x_list):
        """
        Args:
            x_list: List of tensors, each shape (batch_size, block_size, n_embd)
        """
        attended_x_list = []

        # Self-attention and feedforward for each modality
        for i in range(self.num_modalities):
            x = x_list[i]
            x = x + self.sa_layers[i](self.ln1_layers[i](x))
            x = x + self.ffwd_layers[i](self.ln2_layers[i](x))
            attended_x_list.append(x)

        # Cross-attention between modalities
        cross_attended_x_list = []
        for i in range(self.num_modalities):
            x = attended_x_list[i]
            modality_params = self.all_modality_params[i]
            cross_attention_enabled = modality_params[8]

            if cross_attention_enabled and self.cross_attention_layers[i] is not None:
                other_modality_indices = [j for j in range(self.num_modalities) if j != i]
                # Only apply cross-attention if there are other modalities to attend to
                if other_modality_indices:
                    key_value_x_list = [attended_x_list[j] for j in other_modality_indices]
                    x = x + self.cross_attention_layers[i](self.ln_cross_layers[i](x), key_value_x_list)

            cross_attended_x_list.append(x)

        return cross_attended_x_list


class FixedEmbedding(nn.Module):
    """
    Custom embedding layer with fixed (non-learnable) values.

    NOTE: This is an alternative embedding approach not currently used by the main model.
    The active model uses standard nn.Embedding layers. This class is preserved for
    research/experimentation purposes.
    """

    def __init__(self, vocab_size, embed_size, fixed_values):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        # Create embedding table with random values from fixed_values
        embedding_table = torch.zeros(vocab_size, embed_size)
        for i in range(vocab_size):
            for j in range(embed_size):
                embedding_table[i, j] = random.choice(fixed_values)

        # Register as buffer (not learnable parameter)
        self.register_buffer('embedding_table', embedding_table)

    def forward(self, input_tokens):
        """
        Args:
            input_tokens (torch.Tensor): Token indices. Shape: [batch_size, seq_len]
        Returns:
            torch.Tensor: Fixed embeddings. Shape: [batch_size, seq_len, n_embd]
        """
        return self.embedding_table[input_tokens]


def long_tanh(x):
    """Apply tanh activation then convert to long integers (-1, 0, or 1)."""
    return x.tanh().long()


class MultimodalPreBlock(nn.Module):
    """Converts input tokens to embeddings and adds positional information."""

    def __init__(self, num_modalities, vocab_sizes):
        super().__init__()
        self.num_modalities = num_modalities
        self.vocab_sizes = vocab_sizes

        # Token embeddings for each modality
        self.token_embedding_tables = nn.ModuleList([
            nn.Embedding(vocab_sizes[i], _get_n_embd()) for i in range(num_modalities)
        ])
        # Shared positional embeddings
        self.position_embedding_table = nn.Embedding(_get_block_size(), _get_n_embd())

    def forward(self, idx_list):
        """
        Args:
            idx_list: List of token tensors, each shape (batch_size, block_size)
        Returns:
            List of embedded tensors with positional information
        """
        embedded_output_list = []
        for i in range(self.num_modalities):
            B, T = idx_list[i].shape
            tok_emb = self.token_embedding_tables[i](idx_list[i])

            # Add positional embeddings
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx_list[i].device))
            pos_emb = pos_emb.expand_as(tok_emb)

            embedded_output = tok_emb + pos_emb
            embedded_output_list.append(embedded_output)

        return embedded_output_list


class MultimodalPostBlock(nn.Module):
    """Transforms processed outputs into logits for next token prediction."""

    def __init__(self, num_modalities, vocab_sizes):
        super().__init__()
        self.num_modalities = num_modalities
        self.vocab_sizes = vocab_sizes

        self.fin_norm_layers = nn.ModuleList([nn.LayerNorm(_get_n_embd()) for _ in range(num_modalities)])
        self.soft_score_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(_get_n_embd(), vocab_sizes[i] // 2),
                nn.Tanh(),
                nn.Linear(vocab_sizes[i] // 2, vocab_sizes[i])
            ) for i in range(self.num_modalities)
        ])

    def forward(self, x_list):
        """
        Args:
            x_list: List of processed tensors from transformer blocks
        Returns:
            List of logit tensors for each modality
        """
        logits_list = []
        for i in range(self.num_modalities):
            x = self.fin_norm_layers[i](x_list[i])
            logits = self.soft_score_layers[i](x)
            logits_list.append(logits)

        return logits_list


class MultimodalTransformer(nn.Module):
    """Main multimodal transformer model with cross-attention between modalities."""

    def __init__(self, num_modalities, vocab_sizes, all_modality_params):
        super().__init__()
        self.num_modalities = num_modalities
        self.vocab_sizes = vocab_sizes
        self.all_modality_params = all_modality_params

        self.pre_block = MultimodalPreBlock(num_modalities, vocab_sizes)
        self.blocks = nn.Sequential(*[
            MultimodalBlock(_get_n_embd(), _get_n_head(), num_modalities, all_modality_params)
            for _ in range(_get_n_layer())
        ])
        self.post_block = MultimodalPostBlock(num_modalities, vocab_sizes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx_list, targets_list=None):
        """
        Args:
            idx_list: List of input tensors, one per modality
            targets_list: Optional list of target tensors for training
        Returns:
            Tuple of (logits_list, losses_list)
        """
        x_list = self.pre_block(idx_list)
        x_list = self.blocks(x_list)
        logits_list = self.post_block(x_list)

        if targets_list is not None:
            losses_list = []
            for i in range(self.num_modalities):
                B, T, C_i = logits_list[i].shape
                logits_reshaped = logits_list[i].view(B*T, C_i)
                targets_reshaped = targets_list[i].view(B*T)
                loss = F.cross_entropy(logits_reshaped, targets_reshaped)
                losses_list.append(loss)
            return logits_list, losses_list
        else:
            return logits_list, None

    def generate(self, idx_list, max_new_tokens=1, modality_to_generate=0):
        """
        Generate new tokens for a specified modality.

        Args:
            idx_list: List of initial input tensors
            max_new_tokens: Number of tokens to generate
            modality_to_generate: Index of modality to generate for
        Returns:
            List of updated sequence tensors
        """
        generated_sequences_list = [idx.clone() for idx in idx_list]

        for _ in range(max_new_tokens):
            # Crop to context window size
            idx_cond_list = [idx[:, -_get_block_size():] for idx in generated_sequences_list]

            # Get predictions
            logits_list, _ = self(idx_cond_list)

            # Focus on last time step for the target modality
            logits = logits_list[modality_to_generate][:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to target modality
            generated_sequences_list[modality_to_generate] = torch.cat(
                (generated_sequences_list[modality_to_generate], idx_next), dim=1
            )

            # Maintain consistent sequence lengths for other modalities
            for i in range(self.num_modalities):
                if i != modality_to_generate:
                    if generated_sequences_list[i].shape[1] < generated_sequences_list[modality_to_generate].shape[1]:
                        # Pad with last token
                        last_token = generated_sequences_list[i][:, -1:]
                        generated_sequences_list[i] = torch.cat((generated_sequences_list[i], last_token), dim=1)
                    elif generated_sequences_list[i].shape[1] > generated_sequences_list[modality_to_generate].shape[1]:
                        # Crop to match
                        target_length = generated_sequences_list[modality_to_generate].shape[1]
                        generated_sequences_list[i] = generated_sequences_list[i][:, :target_length]

        return generated_sequences_list