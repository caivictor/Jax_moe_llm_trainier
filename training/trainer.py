import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import struct
import functools

# Define the training state
class TrainState(train_state.TrainState):
    # Add any additional state variables if needed, e.g., RNGs
    pass

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
def train_step(state, batch, model, dropout_rng, config):
    """Performs a single training step."""

    # Generate a new dropout RNG key for each step
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

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
    # If using pmap, average gradients across devices
    if config.training.use_distributed:
         grads = jax.lax.pmean(grads, axis_name='batch')
         metrics = jax.lax.pmean(metrics, axis_name='batch')


    # Update model state (apply gradients)
    new_state = state.apply_gradients(grads=grads)

    # Return updated state, metrics, and new RNG
    return new_state, metrics, new_dropout_rng

# Define the evaluation step function (similar to train_step but without gradients)
def eval_step(state, batch, model, config):
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

    # Average metrics across devices if distributed
    if config.training.use_distributed:
        metrics = jax.lax.pmean(metrics, axis_name='batch')

    return metrics


# --- training/trainer.py ---
import jax
import jax.numpy as jnp
from flax.training import train_state
from .train_step import TrainState, train_step, eval_step
from .checkpointing import create_checkpoint_manager, save_checkpoint, restore_checkpoint
from ..data.dataloader import get_datasets, data_generator # Relative imports
from ..model.language_model import LanguageModel
from ..optimizer.shampoo import get_optimizer
from ..utils.logging import Logger # Placeholder for logging utility
from ..utils.distributed import setup_distributed # Placeholder for distributed setup
import time
import numpy as np
from flax.training.common_utils import shard # For sharding data

def train(config):
    """Main training loop."""

    # --- Setup (Random Keys, Distributed Env) ---
    key = jax.random.PRNGKey(config.training.seed)
    model_key, params_key, dropout_key, data_key = jax.random.split(key, 4)

    # Setup distributed environment (if applicable)
    if config.training.use_distributed:
        setup_distributed(config) # Placeholder function
        num_devices = jax.local_device_count()
        # Ensure batch size is divisible by num_devices
        assert config.data.batch_size_per_device * num_devices == config.data.batch_size_per_device * jax.device_count(), \
            "Total batch size must be divisible by number of devices"
    else:
        num_devices = 1

    print(f"Running on {jax.device_count()} devices.")

    # --- Load Data ---
    print("Loading datasets...")
    # Data loaders yield batches already sized for the total number of devices
    train_loader, eval_loader, tokenizer = get_datasets(config)
    # Update vocab size in config based on tokenizer
    config.model.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocabulary size: {config.model.vocab_size}")

    # --- Initialize Model and Optimizer ---
    print("Initializing model...")
    model = LanguageModel(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        dropout_rate=config.model.dropout_rate,
        max_seq_length=config.data.max_seq_length,
        use_moe=config.model.use_moe,
        moe_layer_freq=config.model.moe_layer_freq,
        num_experts=config.model.num_experts,
        num_experts_per_token=config.model.num_experts_per_token,
        capacity_factor=config.model.capacity_factor,
        router_z_loss_coef=config.model.router_z_loss_coef
    )

    # Dummy input for initialization
    dummy_input_ids = jnp.ones((config.data.batch_size_per_device * num_devices, config.data.max_seq_length), dtype=jnp.int32)
    dummy_attention_mask = jnp.ones_like(dummy_input_ids)

    # Initialize parameters
    # Use an initialization function or model.init
    params = model.init(params_key, input_ids=dummy_input_ids, attention_mask=dummy_attention_mask, deterministic=True, train=False)['params']
    print("Model initialized.")

    # Initialize Optimizer
    optimizer = get_optimizer(config)

    # Create TrainState
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    print("Optimizer and TrainState created.")

    # --- Checkpointing ---
    checkpoint_manager = create_checkpoint_manager(config)
    # Restore checkpoint if available
    state, start_step = restore_checkpoint(checkpoint_manager, state, config)

    # --- Distributed Training: Replicate State and JIT ---
    # Replicate state across devices using pmap
    if config.training.use_distributed:
        state = jax.device_put_replicated(state, jax.local_devices())
        dropout_keys = jax.random.split(dropout_key, jax.device_count())
        # Define pmapped train and eval steps
        p_train_step = jax.pmap(
            functools.partial(train_step, model=model, config=config),
            axis_name='batch', # Name for collective operations (pmean)
            # donate_argnums=(0,) # Donate state for potential memory optimization
        )
        p_eval_step = jax.pmap(
            functools.partial(eval_step, model=model, config=config),
            axis_name='batch'
        )
        print("State replicated and JIT functions prepared for distributed training.")
    else:
        # If not distributed, use the original functions
        dropout_keys = dropout_key # Single key
        p_train_step = jax.jit(functools.partial(train_step, model=model, config=config))
        p_eval_step = jax.jit(functools.partial(eval_step, model=model, config=config))
        print("JIT functions prepared for single-device training.")


    # --- Logging ---
    logger = Logger(config) # Initialize logger (TensorBoard/WandB)

    # --- Training Loop ---
    print(f"Starting training from step {start_step + 1}...")
    train_metrics = []
    last_log_time = time.time()

    # Use the data loader directly
    for step, batch in enumerate(train_loader, start=start_step + 1):
        # Stop training if max steps reached
        if step > config.training.num_train_steps:
            break

        # Shard data across devices if using pmap
        if config.training.use_distributed:
            batch = shard(batch) # Shard the PyTree batch across devices
            # Ensure dropout keys are also sharded or handled correctly per device
            # dropout_keys might need to be managed within the pmapped function if stateful RNGs are used.
            # For stateless, splitting outside and passing is okay.

        # Perform a training step
        state, metrics, dropout_keys = p_train_step(state, batch, dropout_keys) # Pass dropout_keys

        train_metrics.append(metrics)

        # --- Logging ---
        if step % config.training.log_steps == 0:
            # Aggregate metrics (average over log_steps)
            # Need to handle metrics potentially being replicated if using pmap
            if config.training.use_distributed:
                # Metrics are already averaged by pmean, take the value from the first device
                 aggregated_metrics = jax.tree_util.tree_map(lambda x: x[0].item(), metrics)
            else:
                 aggregated_metrics = jax.tree_util.tree_map(lambda x: x.item(), metrics) # Get scalar values

            current_time = time.time()
            steps_per_sec = config.training.log_steps / (current_time - last_log_time)
            last_log_time = current_time

            print(f"Step: {step}/{config.training.num_train_steps} | "
                  f"Loss: {aggregated_metrics['loss']:.4f} | "
                  f"LM Loss: {aggregated_metrics['lm_loss']:.4f} | "
                  f"Aux Loss: {aggregated_metrics['moe_aux_loss']:.4f} | "
                  f"Perplexity: {aggregated_metrics['perplexity']:.2f} | "
                  f"Steps/sec: {steps_per_sec:.2f}")

            # Log to TensorBoard/WandB
            logger.log_metrics(aggregated_metrics, step, prefix="train")
            train_metrics = [] # Reset metrics buffer


        # --- Evaluation ---
        if step % config.training.eval_steps == 0:
            print(f"Running evaluation at step {step}...")
            eval_metrics_list = []
            eval_start_time = time.time()
            # Limited number of eval steps or iterate through eval_loader once
            eval_steps_limit = 50 # Example limit
            eval_count = 0
            for eval_batch in eval_loader:
                 if config.training.use_distributed:
                     eval_batch = shard(eval_batch)

                 metrics = p_eval_step(state, eval_batch)

                 if config.training.use_distributed:
                     # Take metrics from first device after pmean
                     metrics = jax.tree_util.tree_map(lambda x: x[0], metrics)
                 eval_metrics_list.append(metrics)

                 eval_count += 1
                 if eval_count >= eval_steps_limit:
                     break # Stop after a fixed number of eval batches

            # Aggregate eval metrics
            if eval_metrics_list:
                # Average metrics across eval batches
                avg_eval_metrics = jax.tree_util.tree_map(lambda *x: jnp.mean(jnp.array(x)), *eval_metrics_list)
                avg_eval_metrics = jax.tree_util.tree_map(lambda x: x.item(), avg_eval_metrics) # To scalar

                eval_duration = time.time() - eval_start_time
                print(f"Evaluation finished in {eval_duration:.2f}s")
                print(f"Step: {step} | Eval Loss: {avg_eval_metrics['eval_loss']:.4f} | Eval Perplexity: {avg_eval_metrics['eval_perplexity']:.2f}")
                logger.log_metrics(avg_eval_metrics, step, prefix="eval")
            else:
                 print("No data found for evaluation.")


        # --- Checkpointing ---
        # Orbax handles the saving interval via CheckpointManagerOptions
        # We just need to call save periodically or let the manager handle it.
        # To ensure saving happens *exactly* at save_steps intervals:
        if step % config.training.save_steps == 0:
             print(f"Attempting to save checkpoint at step {step}...")
             # If distributed, save the unreplicated state from device 0
             state_to_save = state
             if config.training.use_distributed:
                  state_to_save = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state)) # Get state from first device

             save_checkpoint(checkpoint_manager, step, state_to_save, config)


    # --- Final Checkpoint ---
    print("Training finished. Saving final checkpoint...")
    final_state_to_save = state
    if config.training.use_distributed:
         final_state_to_save = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    save_checkpoint(checkpoint_manager, step, final_state_to_save, config) # Save final state
    checkpoint_manager.wait_until_finished()
    logger.close()
    print("Training complete.")
