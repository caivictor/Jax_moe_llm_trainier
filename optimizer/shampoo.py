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
            # Attempt to import from optax.contrib first
            from optax.contrib import shampoo
            print("Imported Shampoo from optax.contrib")
            # Instantiate Shampoo using parameters from config
            return shampoo(
                learning_rate=lr,
                block_size=config.optimizer.block_size,
                beta1=config.optimizer.beta1, # Shampoo might use different beta names/defaults
                beta2=config.optimizer.beta2,
                diagonal_epsilon=1e-10, # Default or configure
                matrix_epsilon=1e-6,    # Default or configure
                weight_decay=config.optimizer.weight_decay,
                start_preconditioning_step=max(1, config.optimizer.preconditioning_compute_steps // 2), # Example logic
                preconditioning_compute_steps=config.optimizer.preconditioning_compute_steps,
                statistics_compute_steps=1, # Usually 1
                best_effort_shape_interpretation=True, # Recommended
                graft_type=config.optimizer.graft_type,
                nesterov=config.optimizer.nesterov,
                exponent_override=0, # Default
                batch_axis_name='batch' # Important if using pmap
            )
        except ImportError:
            print("Warning: Could not import Shampoo from optax.contrib.")
            print("Attempting to import from 'optax-shampoo' library...")
            try:
                from optax_shampoo import shampoo # External library
                print("Imported Shampoo from 'optax-shampoo' library.")
                # Instantiate Shampoo using parameters from config
                # Note: Parameter names might differ slightly from optax.contrib version
                return shampoo(
                    learning_rate=lr,
                    block_size=config.optimizer.block_size,
                    beta1=config.optimizer.beta1,
                    beta2=config.optimizer.beta2,
                    # diagonal_epsilon=1e-10, # Check library specifics
                    # matrix_epsilon=1e-6,    # Check library specifics
                    weight_decay=config.optimizer.weight_decay,
                    # start_preconditioning_step=..., # Check library specifics
                    preconditioning_compute_steps=config.optimizer.preconditioning_compute_steps,
                    # graft_type=config.optimizer.graft_type, # Check library specifics
                    # nesterov=config.optimizer.nesterov, # Check library specifics
                    batch_axis_name='batch' # Important if using pmap
                 )
            except ImportError:
                print("ERROR: Shampoo optimizer specified but couldn't be imported from optax.contrib or optax-shampoo.")
                print("Please install 'optax-shampoo' or ensure Shampoo is available in optax.contrib.")
                print("Falling back to AdamW.")
                return optax.adamw(
                    learning_rate=lr,
                    b1=config.optimizer.beta1,
                    b2=config.optimizer.beta2,
                    weight_decay=config.optimizer.weight_decay
                )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
