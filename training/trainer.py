import jax
import jax.numpy as jnp
from flax.training import train_state
import flax.jax_utils as flax_utils # For replicating/unreplicated state
import functools
import time
import numpy as np
import os # For path joining

# Import using absolute paths from project root
from training.train_step import TrainState, train_step, eval_step
from training.checkpointing import create_checkpoint_manager, save_checkpoint, restore_checkpoint
from data.dataloader import get_datasets # Corrected import
from model.language_model import LanguageModel # Corrected import
from optimizer.shampoo import get_optimizer # Corrected import
from utils.logging import Logger # Corrected import
from utils.distributed import setup_distributed # Corrected import
from flax.training.common_utils import shard # For sharding data

def train(config):
    """Main training loop."""

    # --- Setup (Random Keys, Distributed Env) ---
    key = jax.random.PRNGKey(config.training.seed)
    model_key, params_key, dropout_key, data_key = jax.random.split(key, 4)

    # Setup distributed environment (if applicable)
    # setup_distributed(config) # Call setup function if needed
    num_devices = jax.local_device_count()
    is_distributed = config.training.use_distributed and num_devices > 1
    global_device_count = jax.device_count()

    print(f"Running on {num_devices} local devices ({global_device_count} global devices). Distributed: {is_distributed}")
    if is_distributed:
        # Ensure total batch size is divisible by number of global devices for pmap
        total_batch_size_per_step = config.data.batch_size_per_device * global_device_count
        print(f"Global batch size per step: {total_batch_size_per_step}")
    else:
        total_batch_size_per_step = config.data.batch_size_per_device
        print(f"Single device batch size per step: {total_batch_size_per_step}")


    # --- Load Data ---
    print("Loading datasets...")
    # Pass the correct total batch size needed by the generator
    train_loader, eval_loader, tokenizer = get_datasets(config)
    # Update vocab size in config based on tokenizer
    # Ensure config is mutable or handle this appropriately
    # config.model.vocab_size = tokenizer.vocab_size # This might fail if config is SimpleNamespace
    effective_vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocabulary size: {effective_vocab_size}")

    # --- Initialize Model and Optimizer ---
    print("Initializing model...")
    model = LanguageModel(
        vocab_size=effective_vocab_size, # Use loaded vocab size
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

    # Dummy input for initialization (use total batch size for shape)
    dummy_input_ids = jnp.ones((total_batch_size_per_step, config.data.max_seq_length), dtype=jnp.int32)
    dummy_attention_mask = jnp.ones_like(dummy_input_ids)

    # Initialize parameters
    params = model.init(params_key, input_ids=dummy_input_ids, attention_mask=dummy_attention_mask, deterministic=True, train=False)['params']
    print("Model parameters initialized.")

    # Initialize Optimizer
    optimizer = get_optimizer(config)

    # Create TrainState, including the initial dropout RNG
    # The dropout_key will be split across devices if distributed
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        dropout_rng=dropout_key # Pass the key here, TrainState now accepts it
    )
    print("Optimizer and TrainState created.")

    # --- Checkpointing ---
    checkpoint_manager = create_checkpoint_manager(config)
    # Restore checkpoint if available (restores unreplicated state)
    state, start_step = restore_checkpoint(checkpoint_manager, state, config)

    # --- Distributed Training: Replicate State and JIT ---
    if is_distributed:
        # Split the dropout key in the state across devices *after* potential restore
        state = state.replace(dropout_rng=jax.random.split(state.dropout_rng, global_device_count))
        # Replicate state across devices
        state = flax_utils.replicate(state)
        # Define pmapped train and eval steps
        p_train_step = jax.pmap(
            functools.partial(train_step, model=model, config=config, is_distributed=is_distributed),
            axis_name='batch',
            donate_argnums=(0,) # Donate state for potential memory optimization
        )
        p_eval_step = jax.pmap(
            functools.partial(eval_step, model=model, config=config, is_distributed=is_distributed),
            axis_name='batch'
        )
        print("State replicated and pmapped functions prepared for distributed training.")
    else:
        # If not distributed, JIT the functions
        # No need to split dropout key, it's used directly
        p_train_step = jax.jit(functools.partial(train_step, model=model, config=config, is_distributed=is_distributed))
        p_eval_step = jax.jit(functools.partial(eval_step, model=model, config=config, is_distributed=is_distributed))
        print("JIT functions prepared for single-device training.")


    # --- Logging ---
    # Ensure output dir exists for logger
    os.makedirs(config.training.output_dir, exist_ok=True)
    logger = Logger(config) # Initialize logger (TensorBoard/WandB)

    # --- Training Loop ---
    print(f"Starting training from step {start_step + 1}...")
    train_metrics_list = [] # Collect metrics over log_steps
    last_log_time = time.time()

    # Use the data loader generator
    for step, batch in enumerate(train_loader, start=start_step + 1):
        # Stop training if max steps reached
        if step > config.training.num_train_steps:
            break

        # Shard data across devices if using pmap
        if is_distributed:
            batch = shard(batch) # Shard the PyTree batch across devices

        # Perform a training step
        # train_step now updates the RNG within the state and returns only state, metrics
        state, metrics = p_train_step(state, batch)

        # Collect metrics (metrics are potentially replicated if distributed)
        train_metrics_list.append(metrics)

        # --- Logging ---
        if step % config.training.log_steps == 0:
            # Aggregate metrics collected since last log
            if train_metrics_list:
                # If distributed, metrics are replicated. Unreplicate and average.
                if is_distributed:
                    # Unreplicate first, then average over the list
                    unreplicated_metrics = [flax_utils.unreplicate(m) for m in train_metrics_list]
                    aggregated_metrics = jax.tree_util.tree_map(lambda *x: np.mean([i.item() for i in x]), *unreplicated_metrics)
                else:
                    # If not distributed, metrics are not replicated. Average over the list.
                    aggregated_metrics = jax.tree_util.tree_map(lambda *x: np.mean([i.item() for i in x]), *train_metrics_list)

                current_time = time.time()
                steps_per_sec = config.training.log_steps / (current_time - last_log_time)
                last_log_time = current_time

                print(f"Step: {step}/{config.training.num_train_steps} | "
                      f"Loss: {aggregated_metrics['loss']:.4f} | "
                    #   f"LM Loss: {aggregated_metrics['lm_loss']:.4f} | " # Optional print
                    #   f"Aux Loss: {aggregated_metrics['moe_aux_loss']:.4f} | " # Optional print
                      f"Perplexity: {aggregated_metrics['perplexity']:.2f} | "
                      f"Steps/sec: {steps_per_sec:.2f}")

                # Log to TensorBoard/WandB
                logger.log_metrics(aggregated_metrics, step, prefix="train")
                train_metrics_list = [] # Reset metrics buffer
            else:
                print(f"Step: {step}/{config.training.num_train_steps} | No metrics collected for logging interval.")


        # --- Evaluation ---
        if step % config.training.eval_steps == 0:
            print(f"Running evaluation at step {step}...")
            eval_metrics_list = []
            eval_start_time = time.time()
            # Limited number of eval steps or iterate through eval_loader once
            eval_steps_limit = 50 # Example limit, configure if needed
            eval_count = 0
            try:
                for eval_batch in eval_loader:
                    if eval_count >= eval_steps_limit:
                        break # Stop after a fixed number of eval batches

                    if is_distributed:
                        eval_batch = shard(eval_batch)

                    metrics = p_eval_step(state, eval_batch)
                    eval_metrics_list.append(metrics)
                    eval_count += 1

            except StopIteration:
                 print("Evaluation data loader finished.") # Handle case where eval loader is exhausted
            except Exception as e:
                 print(f"Error during evaluation loop: {e}") # Catch other potential errors


            # Aggregate eval metrics
            if eval_metrics_list:
                 # Unreplicate/average metrics similar to training log
                 if is_distributed:
                      unreplicated_eval_metrics = [flax_utils.unreplicate(m) for m in eval_metrics_list]
                      avg_eval_metrics = jax.tree_util.tree_map(lambda *x: np.mean([i.item() for i in x]), *unreplicated_eval_metrics)
                 else:
                      avg_eval_metrics = jax.tree_util.tree_map(lambda *x: np.mean([i.item() for i in x]), *eval_metrics_list)


                 eval_duration = time.time() - eval_start_time
                 print(f"Evaluation finished in {eval_duration:.2f}s over {eval_count} batches.")
                 print(f"Step: {step} | Eval Loss: {avg_eval_metrics['eval_loss']:.4f} | Eval Perplexity: {avg_eval_metrics['eval_perplexity']:.2f}")
                 logger.log_metrics(avg_eval_metrics, step, prefix="eval")
            else:
                 print("No metrics collected during evaluation.")


        # --- Checkpointing ---
        if step % config.training.save_steps == 0:
             print(f"Attempting to save checkpoint at step {step}...")
             # If distributed, get the state from device 0 to save
             state_to_save = state
             if is_distributed:
                  state_to_save = flax_utils.unreplicate(state)

             save_checkpoint(checkpoint_manager, step, state_to_save, config)
             # Optional: Wait for async save to complete if needed before proceeding far
             # checkpoint_manager.wait_until_finished()


    # --- Final Checkpoint ---
    print("Training finished. Saving final checkpoint...")
    final_state_to_save = state
    if is_distributed:
         final_state_to_save = flax_utils.unreplicate(state)
    save_checkpoint(checkpoint_manager, step, final_state_to_save, config) # Save final state
    checkpoint_manager.wait_until_finished() # Wait for final save
    logger.close()
    print("Training complete.")
