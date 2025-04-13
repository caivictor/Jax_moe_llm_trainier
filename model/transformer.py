import flax.linen as nn
import jax.numpy as jnp
from .moe import SparseMoE, FeedForwardExpert # Import MoE components
from .positional_encoding import RotaryPositionalEncoding # Import RoPE

# Standard Multi-Head Attention
class MultiHeadAttention(nn.Module):
    d_model: int
    num_heads: int
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
        self.rope = RotaryPositionalEncoding(dim=self.head_dim, max_len=2000000) # Increase max_len for 1M context

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

        # Calculate attention scores
        # (batch, num_heads, seq_len, seq_len)
        scores = nn.dot_product_attention_weights(
            q, k,
            bias=mask, # Apply causal mask here if needed
            dropout_rng=self.make_rng('dropout') if not deterministic else None,
            dropout_rate=self.dropout_rate if not deterministic else 0.0,
            deterministic=deterministic,
            dtype=jnp.float32 # Use float32 for precision in attention scores
        )

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
