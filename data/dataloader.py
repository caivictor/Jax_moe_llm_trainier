import jax
import jax.numpy as jnp
from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer
import numpy as np
from flax.training.common_utils import shard # For distributed training

# Placeholder for data loading and preprocessing functions
def get_datasets(config):
    """
    Loads the FineWeb dataset (or specified dataset) and tokenizer.
    Applies tokenization and prepares datasets for JAX training.

    Args:
        config: Configuration object/dictionary.

    Returns:
        Tuple: (train_dataset, eval_dataset, tokenizer)
               Datasets are typically iterators yielding batches of JAX arrays.
    """
    print(f"Loading tokenizer: {config.data.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    # Set padding token if it doesn't exist (e.g., for GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")

    print(f"Loading dataset: {config.data.dataset_name}")
    # Use streaming for very large datasets like FineWeb
    # Adjust split names as needed based on the dataset structure
    train_dataset = load_dataset(
        config.data.dataset_name,
        name=config.data.dataset_config_name,
        split='train',
        streaming=config.data.streaming,
        trust_remote_code=True # Be cautious with this flag
    )
    eval_dataset = load_dataset(
        config.data.dataset_name,
        name=config.data.dataset_config_name,
        split='validation', # Or 'test', adjust as needed
        streaming=config.data.streaming,
        trust_remote_code=True
    )

    # Ensure datasets are iterable if streaming
    if not isinstance(train_dataset, IterableDataset):
         print("Warning: Training dataset is not iterable. Consider streaming for large datasets.")
    if not isinstance(eval_dataset, IterableDataset):
         print("Warning: Evaluation dataset is not iterable. Consider streaming for large datasets.")


    def tokenize_function(examples):
        # Tokenize the text. FineWeb might have a 'text' field. Adjust if needed.
        # Handle potential errors if 'text' field is missing or empty
        texts = [text for text in examples.get("text", []) if text]
        if not texts:
            return {"input_ids": [], "attention_mask": []}

        output = tokenizer(
            texts,
            padding="max_length", # Pad to max_seq_length
            truncation=True,
            max_length=config.data.max_seq_length,
            return_tensors="np" # Return NumPy arrays initially
        )
        # JAX expects labels = input_ids shifted for causal LM
        # Create labels by shifting input_ids
        # output["labels"] = np.roll(output["input_ids"], shift=-1, axis=-1)
        # Handle the last token label (often set to -100 or ignored in loss)
        # output["labels"][:, -1] = -100 # Common practice to ignore loss on padding/last token

        # Alternative: Prepare inputs and targets explicitly if needed by the model/loss
        # inputs = output["input_ids"][:, :-1]
        # targets = output["input_ids"][:, 1:]
        # masks = output["attention_mask"][:, :-1] # Adjust mask accordingly
        # return {"input_ids": inputs, "attention_mask": masks, "labels": targets}

        return output # Return dict with input_ids, attention_mask

    # Apply tokenization lazily if streaming
    # Note: map might behave differently on IterableDataset vs Dataset
    # Consider using a custom generator or tf.data pipeline for complex preprocessing
    print("Tokenizing datasets...")
    # num_proc is not applicable for streaming datasets in the same way
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        # remove_columns=train_dataset.column_names # Keep only needed columns
    )
    eval_tokenized = eval_dataset.map(
        tokenize_function,
        batched=True,
        # remove_columns=eval_dataset.column_names
    )

    # Shuffle the training data (important for non-streaming)
    # For streaming, shuffling is often handled by bufferring
    if config.data.streaming:
        train_tokenized = train_tokenized.shuffle(
            seed=config.training.seed,
            buffer_size=config.data.shuffle_buffer_size
        )
        print(f"Streaming dataset shuffled with buffer size: {config.data.shuffle_buffer_size}")
    # else: # If not streaming
    #     train_tokenized = train_tokenized.shuffle(seed=config.training.seed)


    # --- Batching and JAX Conversion ---
    # This part needs careful handling, especially for distributed training.
    # A common pattern is to use a generator that yields batches of JAX arrays.

    def numpy_to_jax(batch):
        """Converts a batch of numpy arrays to JAX arrays."""
        return jax.tree_util.tree_map(jnp.array, batch)

    def collate_fn(batch_list):
        """Collates a list of examples into a batch dictionary of numpy arrays."""
        # Example assumes batch_list is a list of dicts like {'input_ids': ..., 'attention_mask': ...}
        # This needs robust implementation based on actual data structure
        keys = batch_list[0].keys()
        batch = {k: np.stack([example[k] for example in batch_list]) for k in keys}
        return batch

    def data_generator(dataset, batch_size):
        """Generator yielding batches of JAX arrays."""
        buffer = []
        for example in dataset:
            # Filter out potentially empty examples after tokenization/mapping
            if example and 'input_ids' in example and len(example['input_ids']) > 0:
                 buffer.append(example)
                 if len(buffer) == batch_size:
                     batch = collate_fn(buffer)
                     yield numpy_to_jax(batch)
                     buffer = []
        # Yield any remaining examples if buffer is not empty
        # if buffer:
        #     batch = collate_fn(buffer)
        #     yield numpy_to_jax(batch) # Potentially smaller last batch


    # Calculate total batch size across devices
    total_batch_size = config.data.batch_size_per_device * jax.device_count()
    print(f"Total batch size across {jax.device_count()} devices: {total_batch_size}")

    train_loader = data_generator(train_tokenized, total_batch_size)
    eval_loader = data_generator(eval_tokenized, total_batch_size) # Use same batch size for eval

    return train_loader, eval_loader, tokenizer
