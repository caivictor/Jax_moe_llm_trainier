import flax.linen as nn
import jax.numpy as jnp
from .transformer import TransformerBlock

# The main Language Model combining embeddings, transformer blocks, and output layer
class LanguageModel(nn.Module):
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    dropout_rate: float
    max_seq_length: int # Needed for causal mask shape
    # MoE Configuration
    use_moe: bool
    moe_layer_freq: int
    num_experts: int
    num_experts_per_token: int
    capacity_factor: float
    router_z_loss_coef: float

    def setup(self):
        self.token_embed = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)
        self.dropout = nn.Dropout(rate=self.dropout_rate)

        # Create Transformer Blocks, alternating between Dense and MoE if configured
        blocks = []
        for i in range(self.num_layers):
            is_moe_layer = self.use_moe and (i + 1) % self.moe_layer_freq == 0
            blocks.append(TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                max_seq_length=self.max_seq_length, # Pass max_seq_length here
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                use_moe=is_moe_layer,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                capacity_factor=self.capacity_factor,
                router_z_loss_coef=self.router_z_loss_coef,
                name=f"transformer_block_{i}"
            ))
        self.blocks = blocks

        self.output_norm = nn.LayerNorm(epsilon=1e-6, name="output_norm")
        self.output_proj = nn.Dense(features=self.vocab_size, name="output_proj")

    def __call__(self, input_ids, attention_mask=None, deterministic=False, train=True):
        """
        Forward pass of the language model.

        Args:
            input_ids (jnp.ndarray): Input token IDs (batch_size, seq_len)
            attention_mask (jnp.ndarray, optional): Mask to avoid attending to padding tokens.
                                                   (batch_size, seq_len)
            deterministic (bool): If True, disable dropout.
            train (bool): Flag indicating if the model is in training mode.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
                - Logits (batch_size, seq_len, vocab_size)
                - Total auxiliary MoE loss (scalar)
        """
        batch_size, seq_len = input_ids.shape

        # 1. Token Embeddings
        x = self.token_embed(input_ids)
        x = self.dropout(x, deterministic=deterministic)

        # 2. Create Causal Mask
        # (1, 1, seq_len, seq_len) - Broadcasts across batch and heads
        causal_mask = nn.make_causal_mask(input_ids, dtype=jnp.bool_) # Use input_ids shape

        # Combine with padding mask if provided
        if attention_mask is not None:
            # Attention mask shape: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            padding_mask = attention_mask[:, None, None, :]
            causal_mask = nn.combine_masks(causal_mask, padding_mask)


        # 3. Transformer Blocks
        total_aux_loss = 0.0
        for i, block in enumerate(self.blocks):
            x, aux_loss = block(x, mask=causal_mask, deterministic=deterministic)
            total_aux_loss += aux_loss

        # 4. Final Layer Norm
        x = self.output_norm(x)

        # 5. Output Projection
        logits = self.output_proj(x)

        # Normalize aux loss by number of MoE layers
        num_moe_layers = self.num_layers // self.moe_layer_freq if self.use_moe else 0
        if num_moe_layers > 0:
             total_aux_loss /= num_moe_layers

        return logits, total_aux_loss
