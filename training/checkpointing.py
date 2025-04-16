import jax
import orbax.checkpoint as ocp
import os
from etils import epath # For path manipulation

# Orbax Checkpointing Utilities
def create_checkpoint_manager(config):
    """Creates an Orbax CheckpointManager."""
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=config.training.save_steps,
        max_to_keep=config.checkpointing.keep,
        create=True, # Create directory if it doesn't exist
        # best_fn=lambda metrics: metrics['eval_loss'], # Example: Save best based on eval loss
        # best_mode='min'
    )
    # Ensure the checkpoint directory path is absolute
    absolute_checkpoint_dir = os.path.abspath(config.checkpointing.checkpoint_dir)
    mngr = ocp.CheckpointManager(
        epath.Path(absolute_checkpoint_dir), # Use absolute path
        options=options
    )
    print(f"Checkpoint manager initialized at: {absolute_checkpoint_dir}")
    return mngr

def save_checkpoint(manager, step, train_state, config):
    """Saves the training state using Orbax."""
    # Orbax expects the state and args to save
    # We save the replicated train_state directly if using pmap
    # If using sharded data parallelism (FSDP style), state might already be sharded
    # Orbax handles saving sharded arrays automatically.
    manager.save(step, args=ocp.args.StandardSave(train_state))
    manager.wait_until_finished() # Wait for async save to complete if needed
    print(f"Checkpoint saved for step {step}")

    # Optionally save the config alongside the checkpoint
    config_path = os.path.join(manager.directory, f'config_step_{step}.yaml')
    try:
        import yaml
        # Convert SimpleNamespace to dict recursively
        def namespace_to_dict(ns):
            if isinstance(ns, type(config)):  # SimpleNamespace
                return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
            elif isinstance(ns, dict):
                return {k: namespace_to_dict(v) for k, v in ns.items()}
            elif isinstance(ns, list):
                return [namespace_to_dict(item) for item in ns]
            else:
                return ns
        
        # Convert config to dict and save
        config_dict = namespace_to_dict(config)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        print(f"Config saved to {config_path}")
    except Exception as e:
        print(f"Warning: Could not save config: {e}")


def restore_checkpoint(manager, target_state, config):
    """Restores the latest checkpoint or a specific one."""
    latest_step = manager.latest_step()
    if latest_step is None:
        print("No checkpoint found. Starting from scratch.")
        return target_state, 0 # Return initial state and step 0

    print(f"Restoring checkpoint from step {latest_step}...")
    # Create abstract structure for Orbax to know what to restore
    abstract_state = jax.eval_shape(lambda: target_state)

    restored_state = manager.restore(
        latest_step,
        args=ocp.args.StandardRestore(abstract_state)
    )
    print(f"Checkpoint restored successfully from step {latest_step}.")
    return restored_state, latest_step
