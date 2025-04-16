import flax.linen as nn
import jax.numpy as jnp
from .moe import SparseMoE, FeedForwardExpert # Import MoE components
from .positional_encoding import RotaryPositionalEncoding # Import RoPE

# Standard Multi-Head Attention
class MultiHeadAttention(nn.Module):
    d_model: int
    num_heads: int
    max_seq_length: int # Add max_seq_length attribute
    dropout_rate: float

    def setup(self):
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = self.d_model // self.num_heads
        self.q_proj = nn.Dense(features=self.d_model, name="q_proj")
        self.k_proj = nn.Dense(features=self.d_model, name="k_proj")
        self.v_proj = nn.Dense(features=self.d_model, name="v_proj")
        self.out_proj = nn.Dense(features=self.d_model, name="out_proj")
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        # RoPE for applying rotary embeddings to Q and K
        # Use the passed max_seq_length instead of hardcoded value
        self.rope = RotaryPositionalEncoding(dim=self.head_dim, max_len=self.max_seq_length)

    def __call__(self, x, mask=None, deterministic=False, seq_start_index=0):
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        # (batch, seq_len, num_heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply Rotary Positional Embeddings (RoPE)
        q = self.rope(q, seq_start_index=seq_start_index)
        k = self.rope(k, seq_start_index=seq_start_index)

        # Transpose for attention calculation: (batch, num_heads, seq_len, head_dim)
        q = q.transpose((0, 2, 1, 3))
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))

        # Calculate attention scores manually to ensure correct shapes
        # q, k shapes: (batch, num_heads, seq_len, head_dim) -> (1, 4, 4096, 512)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k)

        # Scale scores
        scale = jnp.sqrt(q.shape[-1]).astype(jnp.float32) # Use float32 for scaling
        attn_weights = attn_weights / scale

        # Apply mask (using where for stability)
        # mask shape: (1, 1, 4096, 4096) - broadcasts correctly
        if mask is not None:
            attn_weights = jnp.where(mask == 0, jnp.finfo(attn_weights.dtype).min, attn_weights)

        # Apply softmax
        attn_weights = nn.softmax(attn_weights, axis=-1).astype(q.dtype) # Cast back to original dtype

        # Apply dropout to attention weights
        if not deterministic:
            # Use the existing dropout layer, assuming it's suitable. Pass the dropout RNG.
            attn_weights = self.dropout(attn_weights, deterministic=deterministic, rng=self.make_rng('dropout'))

        scores = attn_weights # Assign to scores variable used later

        # Apply attention scores to values
        # (batch, num_heads, seq_len, head_dim)
        attention_output = jnp.einsum('bhij,bhjd->bhid', scores, v)

        # Concatenate heads and project back to d_model
        # (batch, seq_len, d_model)
        attention_output = attention_output.transpose((0, 2, 1, 3)).reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(attention_output)
        output = self.dropout(output, deterministic=deterministic)

        return output

# Transformer Block
class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    max_seq_length: int # Add max_seq_length attribute
    d_ff: int
    dropout_rate: float
    use_moe: bool = False # Flag to switch between Dense FFN and MoE
    # MoE specific params (only used if use_moe is True)
    num_experts: int = 8
    num_experts_per_token: int = 2
    capacity_factor: float = 1.25
    router_z_loss_coef: float = 1e-3

    def setup(self):
        self.attention = MultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            max_seq_length=self.max_seq_length, # Pass max_seq_length
            dropout_rate=self.dropout_rate,
            name="attention"
        )
        self.norm1 = nn.LayerNorm(epsilon=1e-6, name="norm1")
        self.norm2 = nn.LayerNorm(epsilon=1e-6, name="norm2")
        self.dropout = nn.Dropout(rate=self.dropout_rate)

        # Conditionally define the FeedForward or MoE layer
        if self.use_moe:
            self.feed_forward_or_moe = SparseMoE(
                d_model=self.d_model,
                d_ff=self.d_ff,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                dropout_rate=self.dropout_rate,
                capacity_factor=self.capacity_factor,
                router_z_loss_coef=self.router_z_loss_coef,
                name="moe_layer"
            )
        else:
            self.feed_forward_or_moe = FeedForwardExpert(
                d_model=self.d_model,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                name="feed_forward_dense"
            )

    def __call__(self, x, mask=None, deterministic=False):
        # Attention sub-layer with pre-normalization
        residual = x
        x_norm = self.norm1(x)
        attn_output = self.attention(x_norm, mask=mask, deterministic=deterministic)
        x = residual + self.dropout(attn_output, deterministic=deterministic)

        # FeedForward/MoE sub-layer with pre-normalization
        residual = x
        x_norm = self.norm2(x)
        if self.use_moe:
            ff_or_moe_output, aux_loss = self.feed_forward_or_moe(x_norm, deterministic=deterministic)
        else:
            ff_or_moe_output = self.feed_forward_or_moe(x_norm, deterministic=deterministic)
            aux_loss = 0.0 # No aux loss for dense FFN

        x = residual + self.dropout(ff_or_moe_output, deterministic=deterministic)

        return x, aux_loss # Return aux_loss (0 if not MoE)
