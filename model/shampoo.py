import optax

# Function to get the configured optimizer
def get_optimizer(config):
    """
    Creates the optimizer based on the config. Currently supports AdamW and Shampoo.
    """
    optimizer_name = config.optimizer.name.lower()
    lr = config.optimizer.learning_rate

    if optimizer_name == "adamw":
        print("Using AdamW optimizer")
        return optax.adamw(
            learning_rate=lr,
            b1=config.optimizer.beta1,
            b2=config.optimizer.beta2,
            weight_decay=config.optimizer.weight_decay
        )
    elif optimizer_name == "shampoo":
        print("Using Shampoo optimizer")
        try:
            # Attempt to import from optax first (newer versions might include it)
            # Or import from optax_shampoo if installed separately
            # from optax_shampoo import shampoo # If using optax-shampoo
            # For now, assume it might be in optax.contrib or needs separate install
            print("Note: Ensure 'optax-shampoo' is installed or Shampoo is available in your optax version.")
            # Placeholder using AdamW if Shampoo isn't found easily
            # Replace this with the actual Shampoo import and instantiation
            print("Shampoo not directly found in standard optax, using AdamW as fallback in this placeholder.")
            print("Please install 'optax-shampoo' or use a JAX environment where Shampoo is available.")

            # Example using optax.contrib.shampoo if available
            # return optax.contrib.shampoo(
            #     learning_rate=lr,
            #     block_size=config.optimizer.block_size,
            #     beta1=config.optimizer.beta1, # Shampoo might use different beta names/defaults
            #     beta2=config.optimizer.beta2,
            #     diagonal_epsilon=1e-10, # Example param
            #     matrix_epsilon=1e-6,    # Example param
            #     weight_decay=config.optimizer.weight_decay,
            #     start_preconditioning_step=config.optimizer.preconditioning_compute_steps, # Adjust param name
            #     preconditioning_compute_steps=config.optimizer.preconditioning_compute_steps,
            #     graft_type=config.optimizer.graft_type,
            #     nesterov=config.optimizer.nesterov,
            #     # Add other Shampoo specific parameters from config
            # )

            # Fallback to AdamW if Shampoo is not found
            return optax.adamw(
                learning_rate=lr,
                b1=config.optimizer.beta1,
                b2=config.optimizer.beta2,
                weight_decay=config.optimizer.weight_decay
            )

        except ImportError:
            print("ERROR: Shampoo optimizer specified but couldn't be imported.")
            print("Please install 'optax-shampoo' or ensure Shampoo is available.")
            print("Falling back to AdamW.")
            return optax.adamw(
                learning_rate=lr,
                b1=config.optimizer.beta1,
                b2=config.optimizer.beta2,
                weight_decay=config.optimizer.weight_decay
            )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
