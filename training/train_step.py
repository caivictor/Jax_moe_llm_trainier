import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import struct
import functools

# Define the training state, adding dropout RNG
class TrainState(train_state.TrainState):
    dropout_rng: jax.random.PRNGKey

# Loss Function (Cross-Entropy for Language Modeling)
def cross_entropy_loss(logits, labels, ignore_id=-100):
    """
    Calculates cross-entropy loss, ignoring padding tokens.

    Args:
        logits (jnp.ndarray): Model output logits (batch, seq_len, vocab_size).
        labels (jnp.ndarray): Target token IDs (batch, seq_len).
        ignore_id (int): Label value to ignore (e.g., for padding).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]:
            - Average loss per non-ignored token (scalar).
            - Number of non-ignored tokens (scalar).
    """
    # Get vocab size from logits
    vocab_size = logits.shape[-1]
    # One-hot encode labels, assign 0 probability to ignored labels
    labels_one_hot = jax.nn.one_hot(labels, num_classes=vocab_size)
    # Create mask for non-ignored labels
    mask = (labels != ignore_id)
    # Calculate loss, applying mask
    log_softmax_logits = jax.nn.log_softmax(logits, axis=-1)
    raw_loss = -jnp.sum(labels_one_hot * log_softmax_logits, axis=-1)
    # Mask the loss for ignored tokens
    masked_loss = raw_loss * mask
    # Calculate average loss over non-ignored tokens
    total_loss = jnp.sum(masked_loss)
    num_non_ignored_tokens = jnp.sum(mask)
    # Avoid division by zero if mask is all False
    average_loss = jax.lax.cond(
        num_non_ignored_tokens > 0,
        lambda: total_loss / num_non_ignored_tokens,
        lambda: 0.0, # Return 0 loss if no valid tokens
    )
    return average_loss, num_non_ignored_tokens


# Define the training step function
# Remove dropout_rng from args, access from state instead
# Add is_distributed flag
def train_step(state: TrainState, batch, model, config, is_distributed: bool):
    """Performs a single training step."""

    # Generate a new dropout RNG key from the one stored in the state
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def compute_loss(params):
        # Get input_ids and attention_mask from the batch
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask') # Optional

        # Prepare labels (assuming causal LM: shift inputs)
        labels = input_ids # Use input_ids as labels initially
        # Shift labels to the left, padding the last position
        labels = jnp.pad(labels[:, 1:], ((0, 0), (0, 1)), constant_values=-100) # Use -100 for ignored token

        # Ensure labels have the same sequence length as logits will have
        if labels.shape[1] != input_ids.shape[1]:
             # This might happen if padding was done differently, adjust as needed
             # Example: pad labels to match input_ids length if necessary
             pad_width = input_ids.shape[1] - labels.shape[1]
             labels = jnp.pad(labels, ((0, 0), (0, pad_width)), constant_values=-100)


        # Run the model forward pass
        logits, aux_loss = model.apply(
            {'params': params},
            input_ids=input_ids,
            attention_mask=attention_mask,
            deterministic=False, # Use dropout during training
            train=True,
            rngs={'dropout': dropout_rng}
        )

        # Calculate the primary loss (cross-entropy)
        lm_loss, num_tokens = cross_entropy_loss(logits, labels, ignore_id=-100)

        # Combine LM loss and MoE auxiliary loss
        total_loss = lm_loss + aux_loss

        # Return loss and metrics
        metrics = {
            'loss': total_loss,
            'lm_loss': lm_loss,
            'moe_aux_loss': aux_loss,
            'num_tokens': num_tokens,
            'perplexity': jnp.exp(lm_loss) # Perplexity based on LM loss
        }
        return total_loss, metrics

    # Compute gradients
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)

    # --- Distributed Training: Gradient Averaging ---
    # Average gradients across devices only if actually distributed
    if is_distributed:
         grads = jax.lax.pmean(grads, axis_name='batch')
         metrics = jax.lax.pmean(metrics, axis_name='batch')


    # Update model state (apply gradients)
    # Also update the dropout RNG in the state
    new_state = state.apply_gradients(grads=grads).replace(dropout_rng=new_dropout_rng)

    # Return updated state and metrics
    return new_state, metrics

# Define the evaluation step function (similar to train_step but without gradients)
# Add is_distributed flag
def eval_step(state, batch, model, config, is_distributed: bool):
    """Performs a single evaluation step."""
    input_ids = batch['input_ids']
    attention_mask = batch.get('attention_mask')
    labels = input_ids
    labels = jnp.pad(labels[:, 1:], ((0, 0), (0, 1)), constant_values=-100)
    if labels.shape[1] != input_ids.shape[1]:
         pad_width = input_ids.shape[1] - labels.shape[1]
         labels = jnp.pad(labels, ((0, 0), (0, pad_width)), constant_values=-100)


    logits, aux_loss = model.apply(
        {'params': state.params},
        input_ids=input_ids,
        attention_mask=attention_mask,
        deterministic=True, # No dropout during evaluation
        train=False
    )

    lm_loss, num_tokens = cross_entropy_loss(logits, labels, ignore_id=-100)
    total_loss = lm_loss + aux_loss # Include aux loss in eval metric if desired

    metrics = {
        'eval_loss': total_loss,
        'eval_lm_loss': lm_loss,
        'eval_moe_aux_loss': aux_loss,
        'eval_perplexity': jnp.exp(lm_loss),
        'eval_num_tokens': num_tokens
    }

    # Average metrics across devices only if actually distributed
    if is_distributed:
        metrics = jax.lax.pmean(metrics, axis_name='batch')

    return metrics
