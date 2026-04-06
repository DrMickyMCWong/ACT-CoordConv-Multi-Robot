# ========================================
# TRANSFORMER.PY LINE-BY-LINE ANALYSIS
# ========================================

# LINE 1-10: File header and imports
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy                                    # For deep copying modules
from typing import Optional, List              # Type hints
import torch                                   # PyTorch core
import torch.nn.functional as F               # Activation functions
from torch import nn, Tensor                 # Neural network modules

import IPython                                # For debugging
e = IPython.embed                            # Debugging function

# ========================================
# MAIN TRANSFORMER CLASS (LINE 20-77)
# ========================================

class Transformer(nn.Module):
    """
    Main transformer class - this is what gets called in ACT's forward pass
    Combines encoder (processes visual features) + decoder (generates actions)
    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        
        # LINE 30-33: CREATE ENCODER STACK
        # The encoder processes visual features from the CNN backbone
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None     # Optional final normalization
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        # LINE 34-37: CREATE DECODER STACK  
        # The decoder generates action predictions using encoder output + queries
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)                                  # Always have decoder norm
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        
        # LINE 39-42: INITIALIZE PARAMETERS
        self._reset_parameters()                # Xavier initialization for better training
        self.d_model = d_model                  # Store model dimension (256 in ACT)
        self.nhead = nhead                      # Store number of heads (8 in ACT)

    def _reset_parameters(self):
        """Initialize all parameters with Xavier uniform distribution"""
        for p in self.parameters():
            if p.dim() > 1:                     # Only initialize multi-dimensional parameters
                nn.init.xavier_uniform_(p)      # Xavier initialization for stable training

    def forward(self, src, mask, query_embed, pos_embed, latent_input=None, proprio_input=None, additional_pos_embed=None):
        """
        MAIN FORWARD PASS - This is the heart of ACT!
        
        Args:
            src: [batch, 256, H, W] - Visual features from CNN backbone
            mask: [batch, H*W] - Attention mask (usually None)
            query_embed: [10, 256] - Learnable action queries
            pos_embed: [batch, 256, H, W] - Visual position embeddings
            latent_input: [batch, 256] - Projected latent variable Z from CVAE
            proprio_input: [batch, 256] - Projected robot state (qpos)
            additional_pos_embed: [2, 256] - Position embeddings for latent+proprio
        
        Returns:
            hs: [batch, 10, 256] - Features for action prediction
        """
        
        # LINE 51-62: HANDLE 4D VISUAL FEATURES (Standard ACT case)
        if len(src.shape) == 4: # has H and W - visual features from CNN
            # STEP 1: RESHAPE VISUAL FEATURES
            bs, c, h, w = src.shape                           # [batch, 256, H, W]
            src = src.flatten(2).permute(2, 0, 1)            # [H*W, batch, 256] - Flatten spatial dims
            
            # STEP 2: RESHAPE VISUAL POSITION EMBEDDINGS  
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)  # [H*W, batch, 256]
            
            # STEP 3: PREPARE ACTION QUERIES
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)             # [10, batch, 256]
            
            # STEP 4: ADD LATENT + PROPRIOCEPTION TO SEQUENCE
            # This is KEY to ACT: we prepend latent Z and robot state to visual features
            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1) # [2, batch, 256]
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)    # [2+H*W, batch, 256]
            
            # STEP 5: CREATE ENCODER INPUT SEQUENCE
            # Sequence: [latent_Z, robot_state, visual_feature_1, ..., visual_feature_HW]
            addition_input = torch.stack([latent_input, proprio_input], axis=0) # [2, batch, 256]
            src = torch.cat([addition_input, src], axis=0)                      # [2+H*W, batch, 256]
            
        # LINE 63-68: HANDLE 3D FEATURES (Alternative format)
        else:
            assert len(src.shape) == 3                        # [batch, H*W, 256] format
            bs, hw, c = src.shape
            src = src.permute(1, 0, 2)                        # [H*W, batch, 256]
            pos_embed = pos_embed.unsqueeze(1).repeat(1, bs, 1)  # [pos_len, batch, 256]
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [10, batch, 256]

        # LINE 70-77: TRANSFORMER PROCESSING
        # STEP 6: INITIALIZE TARGET SEQUENCE (for decoder)
        tgt = torch.zeros_like(query_embed)                   # [10, batch, 256] - Start with zeros
        
        # STEP 7: ENCODER FORWARD PASS
        # Processes: [latent_Z, robot_state, visual_features] → encoded_memory
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        
        # STEP 8: DECODER FORWARD PASS  
        # Combines: encoded_memory + action_queries → action_features
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        
        # STEP 9: RESHAPE OUTPUT
        hs = hs.transpose(1, 2)                               # [batch, 10, 256] - Ready for action heads
        return hs

# ========================================
# TRANSFORMER ENCODER (LINE 79-103)
# ========================================

class TransformerEncoder(nn.Module):
    """
    Stack of encoder layers - processes the multimodal input sequence
    Input: [latent_Z, robot_state, visual_feature_1, ..., visual_feature_HW]
    Output: Encoded memory for the decoder
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)  # Create multiple identical layers
        self.num_layers = num_layers                          # Store number of layers (6 in ACT)
        self.norm = norm                                      # Optional final normalization

    def forward(self, src,
                mask: Optional[Tensor] = None,                # Attention mask
                src_key_padding_mask: Optional[Tensor] = None, # Padding mask
                pos: Optional[Tensor] = None):                # Positional embeddings
        """
        Process sequence through all encoder layers
        
        src: [seq_len, batch, 256] - Input sequence (latent + proprio + visual)
        pos: [seq_len, batch, 256] - Positional embeddings
        """
        output = src                                          # Start with input sequence

        # Apply each encoder layer sequentially
        for layer in self.layers:                             # 6 layers in ACT
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:                             # Apply final normalization if specified
            output = self.norm(output)

        return output                                         # [seq_len, batch, 256] - Encoded memory

# ========================================
# TRANSFORMER DECODER (LINE 105-153)
# ========================================

class TransformerDecoder(nn.Module):
    """
    Stack of decoder layers - generates action predictions
    Uses cross-attention to combine action queries with encoded memory
    """

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)  # Create multiple identical layers
        self.num_layers = num_layers                          # Store number of layers (6 in ACT)
        self.norm = norm                                      # Final normalization layer
        self.return_intermediate = return_intermediate        # Whether to return all layer outputs

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,            # Target sequence mask
                memory_mask: Optional[Tensor] = None,         # Memory mask
                tgt_key_padding_mask: Optional[Tensor] = None, # Target padding mask
                memory_key_padding_mask: Optional[Tensor] = None, # Memory padding mask
                pos: Optional[Tensor] = None,                 # Memory positional embeddings
                query_pos: Optional[Tensor] = None):          # Query positional embeddings
        """
        Generate action predictions using queries and encoded memory
        
        tgt: [10, batch, 256] - Target sequence (starts as zeros)
        memory: [seq_len, batch, 256] - Encoded memory from encoder
        query_pos: [10, batch, 256] - Action query embeddings
        """
        output = tgt                                          # Start with target sequence (zeros)
        intermediate = []                                     # Store intermediate outputs

        # Apply each decoder layer sequentially
        for layer in self.layers:                             # 6 layers in ACT
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:                      # Store output if requested
                intermediate.append(self.norm(output))

        if self.norm is not None:                             # Apply final normalization
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()                            # Remove last intermediate
                intermediate.append(output)                   # Add final output

        if self.return_intermediate:                          # Return all layer outputs
            return torch.stack(intermediate)                  # [num_layers, 10, batch, 256]

        return output.unsqueeze(0)                           # [1, 10, batch, 256] - Single output

# ========================================
# TRANSFORMER ENCODER LAYER (LINE 155-219)
# ========================================

class TransformerEncoderLayer(nn.Module):
    """
    Single encoder layer with self-attention and feedforward
    Processes the multimodal sequence: [latent_Z, robot_state, visual_features]
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # ATTENTION MECHANISM
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)  # Multi-head self-attention
        
        # FEEDFORWARD NETWORK
        self.linear1 = nn.Linear(d_model, dim_feedforward)    # First linear layer: 256 → 2048
        self.dropout = nn.Dropout(dropout)                    # Dropout for regularization
        self.linear2 = nn.Linear(dim_feedforward, d_model)    # Second linear layer: 2048 → 256

        # NORMALIZATION AND REGULARIZATION
        self.norm1 = nn.LayerNorm(d_model)                    # Layer norm after attention
        self.norm2 = nn.LayerNorm(d_model)                    # Layer norm after feedforward
        self.dropout1 = nn.Dropout(dropout)                   # Dropout after attention
        self.dropout2 = nn.Dropout(dropout)                   # Dropout after feedforward

        self.activation = _get_activation_fn(activation)       # Activation function (ReLU)
        self.normalize_before = normalize_before               # Pre-norm vs post-norm

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Add positional encoding to tensor"""
        return tensor if pos is None else tensor + pos        # Element-wise addition

    def forward_post(self,
                     src,                                     # [seq_len, batch, 256] - Input sequence
                     src_mask: Optional[Tensor] = None,       # Attention mask
                     src_key_padding_mask: Optional[Tensor] = None, # Padding mask
                     pos: Optional[Tensor] = None):           # Positional embeddings
        """
        POST-NORMALIZATION: Apply operation then normalize
        Standard transformer architecture
        """
        # STEP 1: SELF-ATTENTION WITH POSITIONAL ENCODING
        q = k = self.with_pos_embed(src, pos)                # Add position to queries and keys
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]  # Self-attention
        
        # STEP 2: RESIDUAL CONNECTION + DROPOUT + LAYER NORM
        src = src + self.dropout1(src2)                      # Residual connection with dropout
        src = self.norm1(src)                                # Layer normalization
        
        # STEP 3: FEEDFORWARD NETWORK
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # FF: Linear→ReLU→Dropout→Linear
        
        # STEP 4: RESIDUAL CONNECTION + DROPOUT + LAYER NORM
        src = src + self.dropout2(src2)                      # Residual connection with dropout
        src = self.norm2(src)                                # Layer normalization
        return src

    def forward_pre(self, src,                               # [seq_len, batch, 256] - Input sequence
                    src_mask: Optional[Tensor] = None,       # Attention mask
                    src_key_padding_mask: Optional[Tensor] = None, # Padding mask
                    pos: Optional[Tensor] = None):           # Positional embeddings
        """
        PRE-NORMALIZATION: Normalize then apply operation
        Modern transformer architecture (more stable training)
        """
        # STEP 1: NORMALIZE THEN SELF-ATTENTION
        src2 = self.norm1(src)                               # Normalize BEFORE attention
        q = k = self.with_pos_embed(src2, pos)               # Add position to normalized input
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]  # Self-attention
        src = src + self.dropout1(src2)                      # Residual connection (no norm after)
        
        # STEP 2: NORMALIZE THEN FEEDFORWARD
        src2 = self.norm2(src)                               # Normalize BEFORE feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))  # Feedforward
        src = src + self.dropout2(src2)                      # Residual connection (no norm after)
        return src

    def forward(self, src,                                   # Main forward function
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """Choose pre-norm or post-norm based on configuration"""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)    # Pre-normalization
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)       # Post-normalization

# ========================================
# TRANSFORMER DECODER LAYER (LINE 221-317)
# ========================================

class TransformerDecoderLayer(nn.Module):
    """
    Single decoder layer with self-attention, cross-attention, and feedforward
    Generates action features by attending to encoded memory
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # ATTENTION MECHANISMS
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)      # Self-attention on queries
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # Cross-attention: queries→memory
        
        # FEEDFORWARD NETWORK
        self.linear1 = nn.Linear(d_model, dim_feedforward)    # First linear: 256 → 2048
        self.dropout = nn.Dropout(dropout)                    # Dropout for regularization
        self.linear2 = nn.Linear(dim_feedforward, d_model)    # Second linear: 2048 → 256

        # NORMALIZATION AND REGULARIZATION (3 layers: self-attn, cross-attn, feedforward)
        self.norm1 = nn.LayerNorm(d_model)                    # After self-attention
        self.norm2 = nn.LayerNorm(d_model)                    # After cross-attention
        self.norm3 = nn.LayerNorm(d_model)                    # After feedforward
        self.dropout1 = nn.Dropout(dropout)                   # After self-attention
        self.dropout2 = nn.Dropout(dropout)                   # After cross-attention
        self.dropout3 = nn.Dropout(dropout)                   # After feedforward

        self.activation = _get_activation_fn(activation)       # Activation function
        self.normalize_before = normalize_before               # Pre-norm vs post-norm

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Add positional encoding to tensor"""
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,                       # POST-NORMALIZATION version
                     tgt_mask: Optional[Tensor] = None,       # Target mask
                     memory_mask: Optional[Tensor] = None,    # Memory mask
                     tgt_key_padding_mask: Optional[Tensor] = None,     # Target padding
                     memory_key_padding_mask: Optional[Tensor] = None,  # Memory padding
                     pos: Optional[Tensor] = None,            # Memory positions
                     query_pos: Optional[Tensor] = None):     # Query positions
        """
        POST-NORMALIZATION decoder layer
        
        tgt: [10, batch, 256] - Target sequence (action queries)
        memory: [seq_len, batch, 256] - Encoded memory (latent + proprio + visual)
        query_pos: [10, batch, 256] - Query positional embeddings
        pos: [seq_len, batch, 256] - Memory positional embeddings
        """
        # STEP 1: SELF-ATTENTION ON ACTION QUERIES
        q = k = self.with_pos_embed(tgt, query_pos)          # Add position to queries
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]  # Self-attention among queries
        tgt = tgt + self.dropout1(tgt2)                      # Residual connection
        tgt = self.norm1(tgt)                                # Layer normalization
        
        # STEP 2: CROSS-ATTENTION: QUERIES → MEMORY
        # This is where action queries "look at" the encoded multimodal memory!
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),      # Queries (with position)
                                   key=self.with_pos_embed(memory, pos),           # Memory keys (with position)
                                   value=memory, attn_mask=memory_mask,             # Memory values
                                   key_padding_mask=memory_key_padding_mask)[0]    # Cross-attention
        tgt = tgt + self.dropout2(tgt2)                      # Residual connection
        tgt = self.norm2(tgt)                                # Layer normalization
        
        # STEP 3: FEEDFORWARD NETWORK
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))  # Feedforward processing
        tgt = tgt + self.dropout3(tgt2)                      # Residual connection
        tgt = self.norm3(tgt)                                # Layer normalization
        return tgt

    def forward_pre(self, tgt, memory,                       # PRE-NORMALIZATION version
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        """PRE-NORMALIZATION decoder layer (same logic but normalize first)"""
        # STEP 1: NORMALIZE THEN SELF-ATTENTION
        tgt2 = self.norm1(tgt)                               # Normalize BEFORE self-attention
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)                      # Residual (no norm after)
        
        # STEP 2: NORMALIZE THEN CROSS-ATTENTION
        tgt2 = self.norm2(tgt)                               # Normalize BEFORE cross-attention
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)                      # Residual (no norm after)
        
        # STEP 3: NORMALIZE THEN FEEDFORWARD
        tgt2 = self.norm3(tgt)                               # Normalize BEFORE feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)                      # Residual (no norm after)
        return tgt

    def forward(self, tgt, memory,                           # Main forward function
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """Choose pre-norm or post-norm based on configuration"""
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

# ========================================
# UTILITY FUNCTIONS (LINE 319-343)
# ========================================

def _get_clones(module, N):
    """Create N identical copies of a module"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])  # Deep copy for independence

def build_transformer(args):
    """
    Factory function to create transformer with specified arguments
    This is called by ACT to create the main transformer
    """
    return Transformer(
        d_model=args.hidden_dim,                  # 256 in ACT
        dropout=args.dropout,                     # 0.1 in ACT
        nhead=args.nheads,                        # 8 in ACT
        dim_feedforward=args.dim_feedforward,     # 2048 in ACT
        num_encoder_layers=args.enc_layers,       # 6 in ACT
        num_decoder_layers=args.dec_layers,       # 6 in ACT
        normalize_before=args.pre_norm,           # False in ACT (post-norm)
        return_intermediate_dec=True,             # Return all decoder layer outputs
    )

def _get_activation_fn(activation):
    """Return activation function based on string name"""
    if activation == "relu":
        return F.relu                             # ReLU activation (default in ACT)
    if activation == "gelu":
        return F.gelu                             # GELU activation
    if activation == "glu":
        return F.glu                              # GLU activation
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")