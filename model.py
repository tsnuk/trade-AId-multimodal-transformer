"""model.py

Neural network classes and model architecture for the multimodal transformer system.

Contains all PyTorch model classes:
- Head (single attention head)
- MultiHeadAttention (multi-head self-attention)
- CrossAttention (cross-modal attention)
- FeedForward (feedforward layer)
- MultimodalBlock (transformer block)
- FixedEmbedding (custom embedding layer)
- MultimodalPreBlock (input processing)
- MultimodalPostBlock (output processing)
- MultimodalTransformer (main model)

Extracted from mm_final_4.py for better code organization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Configuration will be loaded lazily when needed
from config import fixed_values

# Global configuration cache - will be populated when first accessed
_config_cache = None

def _get_config():
    """Lazy load configuration through compatibility layer"""
    global _config_cache
    if _config_cache is None:
        from compatibility_layer import get_system_configuration
        _config_cache = get_system_configuration()
    return _config_cache

# Create property-like accessors for configuration values
def _get_n_embd(): return _get_config()['n_embd']
def _get_n_head(): return _get_config()['n_head']
def _get_n_layer(): return _get_config()['n_layer']
def _get_dropout(): return _get_config()['dropout']
def _get_block_size(): return _get_config()['block_size']
def _get_device(): return _get_config()['device']


class Head(nn.Module):  # The Head class represents a single attention head within a multi-head attention mechanism
                        # Head inheriting from nn.Module is a standard way to define a custom layer/component in PyTorch

    def __init__(self, head_size):  # head_size determines the dimensionality of the output of the head.
                                    # ie, the dimensionality of the output of a single attention head.
                                    # it determines how many "features" or "dimensions" the attention head will use to represent the relationships between tokens in the input sequence.

        super().__init__()  # constructor

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


        '''
        self.key = nn.Sequential(   # nn.Sequential is a container that allows to define a sequence of neural network modules in a sequential order,
                                    # it is used to define the structure of the key, query, and value transformations.
                                    # when we pass an input to an nn.Sequential instance, it will be passed through each module in the defined order, and the output of the last module will be returned.


            nn.Linear(_get_n_embd(), head_size // 2),  # a linear transformation that maps the input embedding (_get_n_embd()) to an intermediate size (head_size // 2).
                                                # the intermediate size in the attention head (head_size // 2) aims to balance dimensionality reduction and information preservation.
                                                # this is particularly beneficial when dealing with large embedding dimensions (_get_n_embd()) or long input sequences.
                                                # while head_size // 2 is a common choice, it's worth considering other values based on the specific application.
                                                # // divides and rounds down to the nearest integer. For example, 7 // 2 would result in 3.
                                                # head_size // 2 is effectively half the value of head_size, rounded down to the nearest whole number.
                                                # we can experiment with other intermediate sizes, such as head_size // 4 or head_size * 3 // 4, and observe their impact on the model's performance.
                                                #
                                                # what is a linear transformation? a linear transformation takes an input vector and applies a weighted sum to produce an output vector.
                                                # this operation can be represented mathematically as:
                                                # y = x * W^T + b
                                                # where:
                                                # x is the input vector, W is the weight matrix (^T denotes the transpose operation), b is the bias vector, y is the output vector
                                                #
                                                # how nn.Linear is applied:
                                                # 1. Input: The input to the first nn.Linear layer is the token embedding (_get_n_embd() dimensions).
                                                # 2. Transformation: The nn.Linear layer applies a linear transformation using a learned weight matrix and a bias vector (if bias=True, which is the default).
                                                # this maps the input to an intermediate size (head_size // 2).
                                                # 3. Non-linearity: The nn.Tanh activation function introduces non-linearity.
                                                # 4. Output: The second nn.Linear layer further transforms the intermediate representation to the final head_size dimension.
                                                # in this case, bias=False, meaning no bias term is used.
                                                #
                                                # nn.Linear is a fundamental module in PyTorch that performs linear transformations.
                                                # it's used within the Head class and other parts of the Transformer model to
                                                # extract features, connect neural network layers, and enable the model to learn and make predictions.


            nn.Tanh(),  # An activation function (hyperbolic tangent) to introduce non-linearity.
                        # it's a non-linear function that maps input values to a range between -1 and 1.
                        # this prevents the activations from becoming too large or too small, which can lead to issues like vanishing or exploding gradients during training.
                        # by introducing a non-linear activation function, the model gains the ability to learn more complex and nuanced patterns in the data.
                        # alternative activation functions that could instead be used are: ReLU, Sigmoid, etc.


            nn.Linear(head_size // 2, head_size, bias=False)  # another linear transformation that maps the intermediate representation to the final head_size dimension.
                                                              # bias=False indicates that the linear layer will not use a bias term.
        )
        '''

        self.register_buffer('tril', torch.tril(torch.ones(_get_block_size(), _get_block_size())))
        # This line registers a buffer called tril
        # torch.tril creates a lower triangular matrix (all elements above the main diagonal are zero) with size block_size x block_size.
        # this is used for masking in the attention mechanism to prevent the model from attending to future tokens in the sequence.

        self.dropout = nn.Dropout(_get_dropout())
        # This adds a _get_dropout() layer to the attention head.
        # Dropout helps to prevent overfitting by randomly setting a fraction of the input units to zero during training.
        # (_get_dropout()) is a hyperparameter that controls the _get_dropout() rate.


    def forward(self, x):   # This function is the core of the attention mechanism within a single "head" of a Transformer model.
                            # This function defines how the attention head processes its input during the forward pass of the neural network.

        Ba,Bl,C = x.shape # x is the input to the attention head
                          # x has shape (batch_size, block_size, _get_n_embd())
                          # batch_size (Ba): The number of independent sequences processed in parallel
                          # block_size (Bl): The length of each sequence (context window)
                          # _get_n_embd() (C): The dimensionality of the token embeddings
        k = self.key(x)   # k has shape (batch_size, block_size, head_size)
                          # The input x is transformed into "keys" (k) using a linear layer (self.key)
        q = self.query(x) # q has shape (batch_size, block_size, head_size)
                          # The input x is transformed into "queries" (q) using a linear layer (self.query).
        aff = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # (Ba, Bl, hs) @ (Ba, hs, Bl) -> (Ba, Bl, Bl)
                                                          # Calculate attention affinities using scaled dot-product attention

        aff = aff.masked_fill(self.tril[:Bl, :Bl] == 0, float('-inf'))  # (Ba, Bl, Bl)
                                                                        # Apply causal mask to prevent attending to future tokens

        aff = F.softmax(aff, dim=-1)  # (Ba, Bl, Bl)
                                      # Convert affinities to attention probabilities

        aff = self.dropout(aff)   # Apply _get_dropout() for regularization

        # Perform weighted aggregation of values
        v = self.value(x) # (Ba,Bl,head_size)
        att_out = aff @ v # (Ba, Bl, Bl) @ (Ba, Bl, head_size) -> (Ba, Bl, head_size)
                          # Weighted aggregation based on attention weights

        return att_out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()  # initializes the parent class (nn.Module), which is a standard practice in PyTorch when defining custom modules
        self.heads = nn.ModuleList([Head(head_size) for cur_head in range(num_heads)])
        # Multiple attention heads for parallel processing
        self.proj = nn.Sequential(
            nn.Linear(head_size * num_heads, _get_n_embd()//2),
            nn.Tanh(),
            nn.Linear(_get_n_embd()//2, _get_n_embd())
        )
        self.dropout = nn.Dropout(_get_dropout())

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # Concatenate outputs from all attention heads
        out = self.dropout(self.proj(out))
        # Project and apply _get_dropout()
        return out


class CrossAttention(nn.Module):
    def __init__(self, num_heads, head_size, num_kv_modalities):
        super().__init__()
        self.num_kv_modalities = num_kv_modalities
        # Each head now needs to process keys/values from multiple modalities
        self.heads = nn.ModuleList([self.Head(head_size, num_kv_modalities) for _ in range(num_heads)])

        # The projection layer needs to handle the concatenated output from all heads
        # The output from each head is head_size, and there are num_heads.
        # The input to proj is the concatenation of head outputs, which will be num_heads * head_size
        self.proj = nn.Sequential(
            nn.Linear(head_size * num_heads, _get_n_embd() // 2),
            nn.Tanh(),
            nn.Linear(_get_n_embd() // 2, _get_n_embd())
        )
        self.dropout = nn.Dropout(_get_dropout())

    def forward(self, query_x, key_value_x_list):
        # query_x: the modality that is querying, shape (batch_size, block_size, _get_n_embd())
        # key_value_x_list: List of tensors, one for each modality providing keys and values
        # Each tensor in key_value_x_list should have shape (batch_size, block_size, _get_n_embd())

        # Concatenate outputs from all heads
        # Each head now takes query_x and the list of key_value_x_list
        out = torch.cat([head(query_x, key_value_x_list) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

    # Modified Head forward for Cross-Attention
    class Head(nn.Module):
        def __init__(self, head_size, num_kv_modalities):
            super().__init__()
            self.num_kv_modalities = num_kv_modalities
            self.head_size = head_size # Store head_size

            # Query projection for the querying modality
            self.query = nn.Linear(_get_n_embd(), head_size, bias=False)

            # Separate key and value projections for each key/value modality
            self.kv_projections = nn.ModuleList([
                nn.Linear(_get_n_embd(), 2 * head_size, bias=False) for _ in range(num_kv_modalities)
            ])

            # Define mask for causal attention (prevent attending to future positions)
            self.register_buffer('tril', torch.tril(torch.ones(_get_block_size(), _get_block_size())))
            self.dropout = nn.Dropout(_get_dropout())

        def forward(self, query_x, key_value_x_list):
            # query_x: shape (batch_size, block_size, _get_n_embd())
            # key_value_x_list: List of tensors, each shape (batch_size, block_size, _get_n_embd())

            Ba, Bl, C = query_x.shape

            q = self.query(query_x) # Queries from query_x, shape (batch_size, block_size, head_size)

            # Calculate affinities and weighted sums for each KV modality separately
            all_kv_outputs = []

            for i, kv_x in enumerate(key_value_x_list):
                # Project KV modality to keys and values
                kv_projection = self.kv_projections[i](kv_x) # shape (batch_size, block_size, 2 * head_size)
                k, v = kv_projection.split(self.head_size, dim=-1) # Each shape (batch_size, block_size, head_size)

                # Calculate affinities between query modality and this KV modality
                aff = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # shape (batch_size, block_size, block_size)

                # Apply causal mask to prevent attending to future positions
                aff = aff.masked_fill(self.tril[:Bl, :Bl] == 0, float('-inf'))

                # Apply softmax to get attention weights
                aff = F.softmax(aff, dim=-1)
                aff = self.dropout(aff)

                # Apply attention weights to values
                kv_output = aff @ v # shape (batch_size, block_size, head_size)
                all_kv_outputs.append(kv_output)

            # Combine outputs from all KV modalities (simple summation for now)
            # Alternative: could use learned weights, concatenation + projection, etc.
            combined_output = sum(all_kv_outputs) # shape (batch_size, block_size, head_size)

            return combined_output


class FeedForward(nn.Module): # Feedforward network with expansion and contraction

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_get_n_embd(), 4 * _get_n_embd()),  # Expand dimensionality
            nn.ReLU(),  # Non-linear activation
            nn.Linear(4 * _get_n_embd(), _get_n_embd()),  # Contract back to original size
            nn.Dropout(_get_dropout()),  # Regularization
        )

    def forward(self, x):
        return self.net(x)  # Forward pass through the network


class MultimodalBlock(nn.Module):
    def __init__(self, n_embd, n_head, num_modalities, all_modality_params):
        super().__init__()
        self.num_modalities = num_modalities
        self.all_modality_params = all_modality_params # Store all_modality_params

        head_size = _get_n_embd() // _get_n_head()
        # Multi-head attention for each modality (self-attention)
        self.sa_layers = nn.ModuleList([MultiHeadAttention(_get_n_head(), head_size) for _ in range(num_modalities)])

        # Feedforward layers for each modality
        self.ffwd_layers = nn.ModuleList([FeedForward() for _ in range(num_modalities)])

        # Layer norms for each modality
        self.ln1_layers = nn.ModuleList([nn.LayerNorm(_get_n_embd()) for _ in range(num_modalities)])
        self.ln2_layers = nn.ModuleList([nn.LayerNorm(_get_n_embd()) for _ in range(num_modalities)])

        # Cross-attention layers: only for modalities with cross_attention enabled
        self.cross_attention_layers = nn.ModuleList()
        for i in range(num_modalities):
            modality_params = all_modality_params[i]
            cross_attention_enabled = modality_params[8] # Index 8 is cross_attention status
            if cross_attention_enabled:
                # Get the indices of other modalities (all except the current one)
                other_modality_indices = [j for j in range(num_modalities) if j != i]
                num_kv_modalities = len(other_modality_indices)
                self.cross_attention_layers.append(CrossAttention(_get_n_head(), head_size, num_kv_modalities))
            else:
                self.cross_attention_layers.append(None) # No cross-attention for this modality

        # Additional layer norms for cross-attention (only for modalities with cross-attention enabled)
        self.ln_cross_layers = nn.ModuleList()
        for i in range(num_modalities):
            modality_params = all_modality_params[i]
            cross_attention_enabled = modality_params[8] # Index 8 is cross_attention status
            if cross_attention_enabled:
                self.ln_cross_layers.append(nn.LayerNorm(_get_n_embd()))
            else:
                self.ln_cross_layers.append(None) # No cross-attention layer norm for this modality

    def forward(self, x_list): # x_list is a list of tensors, one for each modality
        # x_list: List of tensors, each shape (batch_size, block_size, _get_n_embd())

        attended_x_list = []

        # Self-attention and feedforward for each modality
        for i in range(self.num_modalities):
            x = x_list[i]
            # Self-attention with residual connection and layer norm
            x = x + self.sa_layers[i](self.ln1_layers[i](x))
            # Feedforward with residual connection and layer norm
            x = x + self.ffwd_layers[i](self.ln2_layers[i](x))
            attended_x_list.append(x)

        # Cross-attention: Each modality can attend to other modalities
        cross_attended_x_list = []
        for i in range(self.num_modalities):
            x = attended_x_list[i]
            modality_params = self.all_modality_params[i]
            cross_attention_enabled = modality_params[8] # Index 8 is cross_attention status

            if cross_attention_enabled and self.cross_attention_layers[i] is not None:
                # Prepare key-value list from other modalities
                other_modality_indices = [j for j in range(self.num_modalities) if j != i]
                key_value_x_list = [attended_x_list[j] for j in other_modality_indices]

                # Apply cross-attention with residual connection and layer norm
                x = x + self.cross_attention_layers[i](self.ln_cross_layers[i](x), key_value_x_list)

            cross_attended_x_list.append(x)

        return cross_attended_x_list


class FixedEmbedding(nn.Module):  # this class defines a custom embedding layer where the embedding values are fixed and not learned during training
                                  # standard embedding layers in PyTorch have learnable parameters, but this class uses predefined fixed values

    def __init__(self, vocab_size, embed_size, fixed_values):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        # Create embedding table with random values from fixed_values
        embedding_table = torch.zeros(vocab_size, embed_size)
        for i in range(vocab_size):
            for j in range(embed_size):
                embedding_table[i, j] = random.choice(fixed_values)

        # Register as a buffer (not a parameter, so it won't be updated during training)
        self.register_buffer('embedding_table', embedding_table)

    def forward(self, input_tokens):
        # this forward method of this class defines how the embedding layer processes its input
        # it takes input_tokens (a tensor representing token indices) as input
        # and retrieves the corresponding fixed embeddings from the embedding_table based on the input_tokens and returns them as output
        """
        Args:
            input_tokens (torch.Tensor): Indices of tokens. Shape: [batch_size, seq_len]
        Returns:
            torch.Tensor: Fixed embeddings. Shape: [batch_size, seq_len, _get_n_embd()]
        """
        return self.embedding_table[input_tokens]


def long_tanh(x):
    return x.tanh().long()
# Apply tanh activation then convert to long integers (values become -1, 0, or 1)


class MultimodalPreBlock(nn.Module):
    '''
    MultimodalPreBlock is responsible for converting input tokens from multiple modalities into numerical representations called embeddings.
    It also adds information about the position of each token in the sequence, consistently across all modalities.
    '''
    def __init__(self, num_modalities, vocab_sizes):
        super().__init__()
        self.num_modalities = num_modalities
        self.vocab_sizes = vocab_sizes # list of vocab sizes, one for each modality

        # Token embeddings for each modality
        self.token_embedding_tables = nn.ModuleList([nn.Embedding(vocab_sizes[i], _get_n_embd()) for i in range(num_modalities)])

        # Positional embedding table (shared across modalities)
        self.position_embedding_table = nn.Embedding(_get_block_size(), _get_n_embd())

    def forward(self, idx_list): # idx_list is a list of tensors, one for each modality
        # idx_list: List of tensors, each shape (batch_size, block_size)

        embedded_output_list = []
        for i in range(self.num_modalities):
            B, T = idx_list[i].shape
            tok_emb = self.token_embedding_tables[i](idx_list[i]) # Token embeddings for modality i

            # Positional embeddings (shared and expanded)
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx_list[i].device))
            pos_emb = pos_emb.expand_as(tok_emb)

            embedded_output = tok_emb + pos_emb
            embedded_output_list.append(embedded_output)

        return embedded_output_list


class MultimodalPostBlock(nn.Module):
    '''
    MultimodalPostBlock takes the processed output from the multimodal transformer blocks
    and transforms it into logits for each modality for predicting the next token.
    '''
    def __init__(self, num_modalities, vocab_sizes):
        super().__init__()
        self.num_modalities = num_modalities
        self.vocab_sizes = vocab_sizes

        # Layer normalization and linear layers for each modality
        self.fin_norm_layers = nn.ModuleList([nn.LayerNorm(_get_n_embd()) for _ in range(num_modalities)])
        self.soft_score_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(_get_n_embd(), vocab_sizes[i] // 2),
                nn.Tanh(),
                nn.Linear(vocab_sizes[i] // 2, vocab_sizes[i])
            ) for i in range(self.num_modalities)
        ])

    def forward(self, x_list): # x_list is a list of tensors, one for each modality
        # x_list: List of tensors, each shape (batch_size, block_size, _get_n_embd())

        logits_list = []
        for i in range(self.num_modalities):
            x = self.fin_norm_layers[i](x_list[i])
            logits = self.soft_score_layers[i](x)
            logits_list.append(logits)

        return logits_list


'''
The MultimodalTransformer class performs the following operations:
1. MultimodalPreBlock: Prepares input from multiple modalities by converting them into embeddings and adding positional information.
2. MultimodalBlocks: These are the core processing units. Each block performs self-attention within each modality and selective cross-attention between specified modalities.
3. forward: Defines the entire multimodal transformer process.
4. generate: Is used to generate new tokens for a specified modality based on the context from all modalities.
'''
class MultimodalTransformer(nn.Module):

    def __init__(self, num_modalities, vocab_sizes, all_modality_params):
        super().__init__()
        self.num_modalities = num_modalities
        self.vocab_sizes = vocab_sizes
        self.all_modality_params = all_modality_params # Store all_modality_params

        self.pre_block = MultimodalPreBlock(num_modalities, vocab_sizes)
        # Pass all_modality_params to the MultimodalBlock
        self.blocks = nn.Sequential(*[MultimodalBlock(_get_n_embd(), _get_n_head(), num_modalities, all_modality_params) for _ in range(_get_n_layer())])
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
        # idx_list: List of tensors, one for each modality, each shape (batch_size, block_size)
        # targets_list: List of tensors (optional), one for each modality, each shape (batch_size, block_size)

        x_list = self.pre_block(idx_list) # Process input through PreBlock
        x_list = self.blocks(x_list) # Process through transformer blocks
        logits_list = self.post_block(x_list) # Process through PostBlock

        if targets_list is not None:
            losses_list = []
            for i in range(self.num_modalities):
                B, T, C_i = logits_list[i].shape # C_i = vocab_size for modality i
                # Reshape logits and targets for cross_entropy
                logits_reshaped = logits_list[i].view(B*T, C_i)
                targets_reshaped = targets_list[i].view(B*T)
                loss = F.cross_entropy(logits_reshaped, targets_reshaped)
                losses_list.append(loss)
            return logits_list, losses_list
        else:
            return logits_list, None

    def generate(self, idx_list, max_new_tokens=1, modality_to_generate=0):
        # idx_list: List of initial input tensors, one for each modality, each shape (batch_size, initial_seq_len)
        # max_new_tokens: Number of tokens to generate
        # modality_to_generate: Index of the modality for which to generate tokens

        generated_sequences_list = [idx.clone() for idx in idx_list] # Start with the initial sequences

        for _ in range(max_new_tokens):
            # Crop idx_list to the last block_size tokens
            # Need to apply cropping to each tensor in the list
            idx_cond_list = [idx[:, -_get_block_size():] for idx in generated_sequences_list]

            # Get predictions
            logits_list, _ = self(idx_cond_list)

            # Focus only on the last time step (the predicted next token) for each modality
            logits_last_step_list = [logits[:, -1, :] for logits in logits_list] # List of (B, vocab_size)

            # Apply softmax to get probabilities for each modality
            # (focusing only on the modality_to_generate)
            logits = logits_last_step_list[modality_to_generate] # (B, vocab_size)
            probs = F.softmax(logits, dim=-1) # apply softmax to get probabilities

            idx_next = torch.multinomial(probs, num_samples=1) # get next token, shape is (batch_size, 1)

            # Append sampled index only to the specified modality
            generated_sequences_list[modality_to_generate] = torch.cat((generated_sequences_list[modality_to_generate], idx_next), dim=1)

            # For other modalities, update their lists to maintain the same sequence length
            # This is a design choice; you might handle this differently based on your specific use case
            for i in range(self.num_modalities):
                if i != modality_to_generate:
                    # Option 1: Keep the same sequences (no new tokens)
                    # generated_sequences_list[i] remains unchanged
                    # Option 2: Duplicate the last token
                    # last_token = generated_sequences_list[i][:, -1:]
                    # generated_sequences_list[i] = torch.cat((generated_sequences_list[i], last_token), dim=1)
                    # For now, let's use option 1 (keep the same sequences)
                    # But let's crop them to maintain consistent lengths if needed
                    if generated_sequences_list[i].shape[1] < generated_sequences_list[modality_to_generate].shape[1]:
                        # Pad with the last token to match the length of the generated modality
                        last_token = generated_sequences_list[i][:, -1:]
                        generated_sequences_list[i] = torch.cat((generated_sequences_list[i], last_token), dim=1)
                    elif generated_sequences_list[i].shape[1] > generated_sequences_list[modality_to_generate].shape[1]:
                        # Crop to match the length of the generated modality
                        target_length = generated_sequences_list[modality_to_generate].shape[1]
                        generated_sequences_list[i] = generated_sequences_list[i][:, :target_length]
                    # If lengths are equal, do nothing

        return generated_sequences_list # Return the updated list of modality tensors