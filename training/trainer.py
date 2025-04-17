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


# --- Generation using jax.lax.scan ---
@functools.partial(jax.jit, static_argnums=(0, 4, 5)) # JIT compile, static args for model, max_length, temperature
def generate_text_scan(model, params, prng_key, input_ids, max_length=50, temperature=1.0, pad_token_id=0):
    """Generates text using sampling with jax.lax.scan and fixed-size carry."""

    # Ensure input_ids has a batch dimension
    if input_ids.ndim == 1:
        input_ids_batched = input_ids[None, :]
        was_1d = True
    else:
        input_ids_batched = input_ids
        was_1d = False

    batch_size = input_ids_batched.shape[0]
    initial_seq_len = input_ids_batched.shape[1]
    num_steps_to_generate = max_length - initial_seq_len

    # Pre-allocate the full sequence tensor
    # Shape: (batch_size, max_length)
    # Initialize with pad tokens, then fill with the prompt
    full_sequence = jnp.full((batch_size, max_length), pad_token_id, dtype=input_ids_batched.dtype)
    full_sequence = full_sequence.at[:, :initial_seq_len].set(input_ids_batched)

    def generation_step(carry, _):
        """One step of the generation loop with fixed-size carry."""
        # Carry: (current_full_sequence, current_generation_index)
        # Carry: (current_full_sequence, current_generation_index, current_prng_key)
        current_seq_tensor, current_index, current_key = carry
        # Split key for this step's sampling
        step_key, next_key = jax.random.split(current_key)

        # No need to slice dynamically. Pass the full tensor.
        # The model's internal causal mask handles attending only to valid tokens.
        model_input_ids = current_seq_tensor

        # Model forward pass
        outputs = model.apply(
            {'params': params},
            input_ids=model_input_ids, # Pass the full tensor
            attention_mask=None, # Causal mask handled internally
            train=False,
            deterministic=True
        )
        logits, _ = outputs # Shape: (batch_size, current_index, vocab_size)

        # Get logits for the *last valid token* position (index - 1)
        last_token_logits = logits[:, -1, :] / temperature # Shape: (batch_size, vocab_size)

        # Sample from the distribution
        next_token_id = jax.random.categorical(step_key, last_token_logits, axis=-1) # Shape: (batch_size,)

        # Update the pre-allocated tensor at the current index
        # Shape of next_token_id: (batch_size,) -> needs reshape for update
        # Shape of slice to update: (batch_size, 1) at index `current_index`
        updated_seq_tensor = current_seq_tensor.at[:, current_index].set(next_token_id)

        # Return the updated tensor, the *next* index, and the *next* key as the carry
        # The shape of the carry (tensor shape, scalar shape, key shape) remains constant
        return (updated_seq_tensor, current_index + 1, next_key), next_token_id

    # Initial carry state (add the initial PRNG key)
    initial_carry = (full_sequence, initial_seq_len, prng_key)

    # Run the scan loop
    final_carry, _ = jax.lax.scan(
        generation_step,
        initial_carry,      # Initial carry: (tensor, start_index)
        None,               # No per-step input needed
        length=num_steps_to_generate
    )

    # The final generated sequence is the first element of the final carry
    final_sequence_batched = final_carry[0]

    # Remove batch dimension if the original input was 1D
    if was_1d:
        final_sequence = final_sequence_batched[0]
    else:
        final_sequence = final_sequence_batched

    return final_sequence


# --- Evaluation Function ---
def run_evaluation(step, state, model, eval_loader, tokenizer, p_eval_step, logger, config, is_distributed):
    """Runs the evaluation loop and logs metrics."""
    print(f"Running evaluation at step {step}...")
    eval_metrics_list = []
    eval_start_time = time.time()
    # Limited number of eval steps or iterate through eval_loader once
    # Use hasattr for SimpleNamespace compatibility
    eval_steps_limit = config.training.eval_steps_limit if hasattr(config.training, 'eval_steps_limit') else 50
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

         # --- Add Inference Here ---
         # Get unreplicated parameters for generation
         inference_params = state.params
         if is_distributed:
             # Ensure we get the parameters from the first device if replicated
             inference_params = flax_utils.unreplicate(state).params

         # Define the prompt and tokenize it
         prompt = "I am a "
         # Ensure tokenizer has 'encode' method and returns 'input_ids' or similar
         # Adjust based on your actual tokenizer implementation (e.g., Hugging Face, SentencePiece)
         try:
             # Attempt common encoding patterns
             if hasattr(tokenizer, 'encode') and callable(tokenizer.encode):
                 # Check if it returns a dict (like HF tokenizers)
                 encoded_output = tokenizer.encode(prompt, return_tensors="np")
                 if isinstance(encoded_output, dict) and 'input_ids' in encoded_output:
                     tokenized_prompt = encoded_output['input_ids']
                 else: # Assume it returns IDs directly
                     tokenized_prompt = jnp.array(encoded_output)

                 # Remove batch dimension if tokenizer adds one and we only want 1D
                 if tokenized_prompt.ndim > 1 and tokenized_prompt.shape[0] == 1:
                     tokenized_prompt = tokenized_prompt[0]
             else:
                 raise ValueError("Tokenizer does not have a standard 'encode' method.")

         except Exception as e:
             print(f"Error tokenizing prompt: {e}. Skipping generation.")
             tokenized_prompt = None # Handle error case

         if tokenized_prompt is not None:
             print(f"\n--- Generating text from prompt: '{prompt}' ---")
             # Generate text (adjust max_length as needed)
             # Use jax.device_get to bring result back to host if needed.
             # Use hasattr for SimpleNamespace compatibility
             default_generate_length = config.data.max_seq_length // 2
             generate_max_length = config.training.generate_max_length if hasattr(config.training, 'generate_max_length') else default_generate_length

             # --- Use the new scan-based generation function ---
             generated_ids = generate_text_scan(
                 model,
                 inference_params, # Use unreplicated params
                 jax.random.PRNGKey(int(time.time())), # Pass a PRNG key for sampling
                 tokenized_prompt,
                 max_length=generate_max_length, # Use configured/default length
                 temperature=1.0 # Explicitly pass temperature (can be configured later)
             )
             generated_ids = jax.device_get(generated_ids) # Ensure result is on host CPU

             # Decode the generated IDs
             # Ensure tokenizer has 'decode' method
             try:
                 if hasattr(tokenizer, 'decode') and callable(tokenizer.decode):
                     generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                     print(f"Generated Text: {generated_text}")
                 else:
                     raise ValueError("Tokenizer does not have a standard 'decode' method.")
             except Exception as e:
                 print(f"Error decoding generated text: {e}")
             print("-------------------------------------------------\n")

    else:
         print("No metrics collected during evaluation.")


# --- Main Training Function ---
def train(config, eval_only=False): # Add eval_only parameter
    """Main training and evaluation function."""

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

    # --- Check if running evaluation only ---
    if eval_only:
        print("Running in evaluation-only mode.")
        if start_step == 0:
            print("Warning: No checkpoint found to evaluate. Exiting.")
            return # Or raise an error

        # Run evaluation using the restored state (start_step is the step of the loaded checkpoint)
        run_evaluation(start_step, state, model, eval_loader, tokenizer, p_eval_step, logger, config, is_distributed)
        logger.close()
        print("Evaluation complete.")
        return # Exit after evaluation

    # --- Training Loop ---
    print(f"Starting training from step {start_step + 1}...")
    train_metrics_list = [] # Collect metrics over log_steps
    last_log_time = time.time()
    final_step = 0 # Keep track of the last step reached

    # Use the data loader generator
    for step, batch in enumerate(train_loader, start=start_step + 1):
        final_step = step # Update final step
        # Stop training if max steps reached
        if step > config.training.num_train_steps:
            break

        # Shard data across devices if using pmap
        if is_distributed:
            batch = shard(batch) # Shard the PyTree batch across devices

        # Perform a training step
        state, metrics = p_train_step(state, batch)

        # Collect metrics
        train_metrics_list.append(metrics)

        # --- Logging ---
        if step % config.training.log_steps == 0:
            if train_metrics_list:
                if is_distributed:
                    unreplicated_metrics = [flax_utils.unreplicate(m) for m in train_metrics_list]
                    aggregated_metrics = jax.tree_util.tree_map(lambda *x: np.mean([i.item() for i in x]), *unreplicated_metrics)
                else:
                    aggregated_metrics = jax.tree_util.tree_map(lambda *x: np.mean([i.item() for i in x]), *train_metrics_list)

                current_time = time.time()
                steps_per_sec = config.training.log_steps / (current_time - last_log_time)
                last_log_time = current_time

                print(f"Step: {step}/{config.training.num_train_steps} | "
                      f"Loss: {aggregated_metrics['loss']:.4f} | "
                      f"Perplexity: {aggregated_metrics['perplexity']:.2f} | "
                      f"Steps/sec: {steps_per_sec:.2f}")

                logger.log_metrics(aggregated_metrics, step, prefix="train")
                train_metrics_list = []
            else:
                print(f"Step: {step}/{config.training.num_train_steps} | No metrics collected for logging interval.")

        # --- Evaluation ---
        if step % config.training.eval_steps == 0:
            # Call the refactored evaluation function
            run_evaluation(step, state, model, eval_loader, tokenizer, p_eval_step, logger, config, is_distributed)

        # --- Checkpointing ---
        if step % config.training.save_steps == 0:
             print(f"Attempting to save checkpoint at step {step}...")
             state_to_save = state
             if is_distributed:
                  state_to_save = flax_utils.unreplicate(state)
             save_checkpoint(checkpoint_manager, step, state_to_save, config)

    # --- Final Actions After Training Loop ---
    print("Training loop finished.")

    # Save final checkpoint if training occurred
    if final_step > start_step: # Check if any training steps were actually run
        print(f"Saving final checkpoint at step {final_step}...")
        final_state_to_save = state
        if is_distributed:
             final_state_to_save = flax_utils.unreplicate(state)
        save_checkpoint(checkpoint_manager, final_step, final_state_to_save, config)
        checkpoint_manager.wait_until_finished() # Wait for final save

    logger.close()
    print("Training process complete.")
