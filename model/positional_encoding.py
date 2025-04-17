import jax.numpy as jnp
import flax.linen as nn

# Implementation of Rotary Positional Embeddings (RoPE)
# Based on https://arxiv.org/pdf/2104.09864.pdf
class RotaryPositionalEncoding(nn.Module):
    """Applies Rotary Positional Embeddings to the input."""
    dim: int
    max_len: int = 5000 # Default max length, adjust as needed
    theta: float = 10000.0

    def setup(self):
        # Precompute frequencies
        freqs = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2) / self.dim))
        self.seq = jnp.arange(self.max_len)
        freqs_cis = jnp.outer(self.seq, freqs)
        # Shape: (max_len, dim / 2)
        self.freqs_cis_real = jnp.cos(freqs_cis)
        self.freqs_cis_imag = jnp.sin(freqs_cis)

    def __call__(self, x, seq_start_index=0):
        """
        Apply RoPE to the input tensor.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch, seq_len, num_heads, head_dim)
                             or (batch, seq_len, embed_dim). Assumes dim = head_dim or embed_dim.
            seq_start_index (int): The starting index for the sequence dimension,
                                   useful for caching in autoregressive generation.

        Returns:
            jnp.ndarray: Tensor with RoPE applied.
        """
        seq_len = x.shape[1]
        # Ensure head_dim is even
        assert self.dim % 2 == 0, "Dimension must be even for RoPE."

        # --- Add Check for Slice Indices ---
        end_index = seq_start_index + seq_len
        if end_index > self.max_len:
            # Raise a clearer error instead of letting JAX crash later
            raise ValueError(
                f"RoPE slicing error: Calculated end index {end_index} (start={seq_start_index}, len={seq_len}) "
                f"exceeds the precomputed max_len {self.max_len} for positional embeddings. "
                f"Ensure generation length doesn't exceed model's max sequence length."
            )
        # --- End Check ---

        # Reshape x if it's the full embedding dim (e.g., before splitting heads)
        original_shape = x.shape
        if len(original_shape) == 3: # (batch, seq_len, embed_dim)
             x = x.reshape(*original_shape[:-1], -1, self.dim) # Assume last dim can be factored

        # Split into real and imaginary parts (conceptual)
        x_real = x[..., 0::2]
        x_imag = x[..., 1::2]

        # Get the corresponding frequencies for the current sequence length and start index
        # Shape: (seq_len, dim / 2)
        # Use the calculated end_index which is guaranteed to be within bounds now
        freqs_real = self.freqs_cis_real[seq_start_index : end_index, :]
        freqs_imag = self.freqs_cis_imag[seq_start_index : end_index, :]

        # Add sequence dimension for broadcasting: (1, seq_len, 1, dim / 2)
        freqs_real = freqs_real[None, :, None, :]
        freqs_imag = freqs_imag[None, :, None, :]

        # Apply rotation
        # x' = x * cos(m*theta) - rotate_half(x) * sin(m*theta)
        x_rotated_real = x_real * freqs_real - x_imag * freqs_imag
        x_rotated_imag = x_real * freqs_imag + x_imag * freqs_imag

        # Combine back: Interleave real and imaginary parts
        # Create an output tensor of the same shape as x
        rotated_x = jnp.empty_like(x)
        rotated_x = rotated_x.at[..., 0::2].set(x_rotated_real)
        rotated_x = rotated_x.at[..., 1::2].set(x_rotated_imag)

        # Reshape back to original if necessary
        if len(original_shape) == 3:
            rotated_x = rotated_x.reshape(*original_shape)

        return rotated_x

# Placeholder for other positional encoding techniques if needed (e.g., ALiBi, YaRN)
