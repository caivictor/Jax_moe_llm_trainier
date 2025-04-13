from torch.utils.tensorboard import SummaryWriter # Using PyTorch's TensorBoard logger for convenience
import wandb
import os

# Simple Logger using TensorBoard or WandB
class Logger:
    def __init__(self, config):
        self.config = config
        self.writer = None
        self.use_wandb = config.logging.use_wandb

        if config.logging.use_tensorboard:
            log_dir = os.path.join(config.training.output_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging initialized at: {log_dir}")

        if self.use_wandb:
            try:
                wandb.init(
                    project=config.logging.wandb_project,
                    entity=config.logging.wandb_entity,
                    config=config.to_dict() # Log the entire config
                )
                print("Weights & Biases initialized.")
            except Exception as e:
                print(f"Could not initialize WandB: {e}")
                self.use_wandb = False

    def log_metrics(self, metrics, step, prefix=""):
        """Logs metrics to TensorBoard and/or WandB."""
        log_dict = {}
        for key, value in metrics.items():
            log_key = f"{prefix}/{key}" if prefix else key
            log_dict[log_key] = value # For WandB

            if self.writer:
                self.writer.add_scalar(log_key, value, step)

        if self.use_wandb:
            wandb.log(log_dict, step=step)

    def close(self):
        """Closes the logger resources."""
        if self.writer:
            self.writer.close()
        if self.use_wandb:
            wandb.finish()
        print("Logger closed.")

