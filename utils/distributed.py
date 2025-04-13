import jax
import os

# Placeholder for distributed setup logic
def setup_distributed(config):
    """Initializes JAX distributed environment if necessary."""
    # JAX often automatically discovers devices.
    # For multi-host setups (TPU Pods), initialization might be needed.
    # Example: jax.distributed.initialize()
    # This might depend on the specific cluster environment (GCP, Slurm, etc.)
    print("Setting up distributed environment...")
    try:
        # Try initializing (may not be needed for single-host multi-GPU)
        # jax.distributed.initialize() # Uncomment if required for your setup
        print(f"JAX Global Device Count: {jax.device_count()}")
        print(f"JAX Local Device Count: {jax.local_device_count()}")
        # Set environment variables if needed (e.g., for NCCL)
        # os.environ['NCCL_DEBUG'] = 'INFO'
    except Exception as e:
        print(f"Could not initialize JAX distributed: {e}")
        print("Proceeding with available local devices.")

    if jax.device_count() > 1:
        print("Distributed training enabled.")
    else:
        print("Warning: Distributed training requested but only one device found.")
        config.training.use_distributed = False # Fallback to single device

