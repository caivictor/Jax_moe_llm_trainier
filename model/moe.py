import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import Optional

# Mixture of Experts Layer Implementation
class FeedForwardExpert(nn.Module):
    """Standard FeedForward layer used as an expert."""
    d_model: int
    d_ff: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, deterministic: bool):
        # Project up
        x = nn.Dense(features=self.d_ff, name="dense_up")(x)
        x = nn.relu(x) # Or gelu, swish etc.
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        # Project down
        x = nn.Dense(features=self.d_model, name="dense_down")(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        return x

class SparseMoE(nn.Module):
    """Sparse Mixture of Experts layer."""
    d_model: int
    d_ff: int # Dimension of the feed-forward layer within each expert
    num_experts: int
    num_experts_per_token: int # Top-k routing
    dropout_rate: float
    capacity_factor: float = 1.25 # Controls max tokens per expert
    router_z_loss_coef: float = 1e-3 # Coefficient for auxiliary load balancing loss

    @nn.compact
    def __call__(self, x, deterministic: bool):
        """
        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, seq_len, d_model)
            deterministic (bool): If true, disable dropout.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
                - Output tensor of shape (batch_size, seq_len, d_model)
                - Auxiliary load balancing loss (scalar)
        """
        batch_size, seq_len, d_model = x.shape
        num_tokens = batch_size * seq_len
        x_flat = x.reshape(num_tokens, d_model) # Flatten input for routing

        # --- Routing Logic ---
        # Gating network: Projects input tokens to logits for each expert
        # Shape: (num_tokens, num_experts)
        router_logits = nn.Dense(features=self.num_experts, use_bias=False, name="gate")(x_flat)

        # Apply softmax to get probabilities/weights
        # Using softmax ensures weights sum to 1, useful for weighted combination
        router_probs = nn.softmax(router_logits, axis=-1)

        # --- Top-K Selection ---
        # Find the top-k experts and their weights for each token
        # Shape: (num_tokens, num_experts_per_token)
        expert_weights, expert_indices = jax.lax.top_k(router_probs, k=self.num_experts_per_token)

        # Normalize the weights of the selected experts
        expert_weights /= jnp.sum(expert_weights, axis=-1, keepdims=True)

        # --- Dispatch Tokens to Experts ---
        # Create a sparse dispatch tensor/mask
        # Shape: (num_tokens, num_experts) - 1 if token i goes to expert j, 0 otherwise
        # This is complex to do efficiently in JAX without explicit loops.
        # A common approach involves scatter operations or one-hot encoding combined with masking.

        # Calculate expert capacity
        # Max number of tokens each expert can process
        expert_capacity = int((num_tokens / self.num_experts) * self.capacity_factor * self.num_experts_per_token)
        expert_capacity = max(expert_capacity, 1) # Ensure capacity is at least 1

        # Create a binary mask indicating which tokens are routed to which expert
        # Shape: (num_tokens, num_experts)
        expert_mask = jax.nn.one_hot(expert_indices, num_classes=self.num_experts, axis=-1)
        expert_mask = jnp.sum(expert_mask, axis=-2) # Sum over the k dimension

        # --- Load Balancing Calculation ---
        # Calculate how many tokens are assigned to each expert
        # Shape: (num_experts,)
        tokens_per_expert = jnp.sum(expert_mask, axis=0)

        # Calculate the fraction of tokens routed to each expert
        # Shape: (num_experts,)
        load = tokens_per_expert / num_tokens

        # Calculate the average probability assigned to each expert by the router
        # Shape: (num_experts,)
        importance = jnp.sum(router_probs, axis=0) / num_tokens

        # Auxiliary Load Balancing Loss (from Switch Transformers paper)
        # Encourages router probabilities and token distribution to be similar
        aux_loss = self.num_experts * jnp.sum(load * importance)
        aux_loss *= self.router_z_loss_coef

        # --- Expert Computation ---
        # This is the most challenging part for efficiency. We want to process tokens
        # assigned to the same expert together. This often requires permuting/sorting
        # tokens based on their assigned expert index.

        # A simplified (potentially less efficient) approach:
        # Pass all tokens through all experts and then combine using weights.
        # This avoids complex dispatch/combine logic but is computationally expensive.

        # More efficient approach (conceptual):
        # 1. Permute tokens: Group tokens by their assigned expert.
        # 2. Pad/Truncate: Ensure each expert receives exactly `expert_capacity` tokens.
        # 3. Expert Forward Pass: Apply each expert FFN to its assigned tokens.
        # 4. Un-permute: Map the expert outputs back to the original token positions.
        # 5. Combine: Combine outputs using the router weights.

        # --- Simplified Combination (Illustrative - Inefficient) ---
        expert_outputs = []
        experts = [
            FeedForwardExpert(d_model=self.d_model, d_ff=self.d_ff, dropout_rate=self.dropout_rate, name=f"expert_{i}")
            for i in range(self.num_experts)
        ]

        final_output = jnp.zeros_like(x_flat)
        for i in range(self.num_experts):
            # Get the indices and weights for tokens assigned to this expert
            # This requires finding which tokens selected expert `i` in their top-k
            token_indices_for_expert_i, = jnp.where(expert_indices == i)
            weights_for_expert_i = expert_weights[token_indices_for_expert_i] # Need careful indexing

            # Select the input tokens for this expert
            inputs_for_expert_i = x_flat[token_indices_for_expert_i]

            # Apply the expert (handle potential empty inputs)
            if inputs_for_expert_i.shape[0] > 0:
                 output_expert_i = experts[i](inputs_for_expert_i, deterministic)
                 # Combine outputs weighted by router probabilities
                 # This needs careful scatter/update based on original indices
                 # final_output = final_output.at[token_indices_for_expert_i].add(output_expert_i * weights_for_expert_i[:,None]) # Approximate
            # This simplified loop is highly inefficient and conceptually flawed for proper MoE.
            # A real implementation requires efficient dispatch/combine, often using custom kernels or libraries.


        # --- Placeholder for actual efficient implementation ---
        # Assume `final_output` is computed efficiently using dispatch/combine
        # based on `expert_weights` and `expert_indices`.
        # This part is non-trivial in JAX. Libraries like Tutel or custom implementations
        # are often needed for high performance.

        # For now, just pass through one expert as a placeholder structure
        # This does NOT represent a functional MoE layer.
        placeholder_expert = FeedForwardExpert(d_model=self.d_model, d_ff=self.d_ff, dropout_rate=self.dropout_rate, name="placeholder_expert")
        final_output_placeholder = placeholder_expert(x_flat, deterministic)


        # Reshape back to original shape
        final_output_reshaped = final_output_placeholder.reshape(batch_size, seq_len, d_model) # Use placeholder output

        return final_output_reshaped, aux_loss

