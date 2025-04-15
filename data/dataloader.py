import jax
import jax.numpy as jnp
from datasets import load_dataset, IterableDataset, get_dataset_split_names
from transformers import AutoTokenizer
import numpy as np
from flax.training.common_utils import shard # For distributed training
import itertools # For islice

# Placeholder for data loading and preprocessing functions
def get_datasets(config):
    """
    Loads the FineWeb dataset (or specified dataset) and tokenizer.
    Applies tokenization and prepares datasets for JAX training.
    If 'validation' split is not available, it splits the 'train' set
    using take/skip for streaming compatibility.

    Args:
        config: Configuration object/dictionary (expected as SimpleNamespace).

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
    dataset_name = config.data.dataset_name
    dataset_config = config.data.dataset_config_name
    streaming = config.data.streaming

    # Check available splits
    available_splits = get_dataset_split_names(dataset_name, config_name=dataset_config)
    print(f"Available splits in dataset: {available_splits}")

    if 'validation' in available_splits:
        print("Found 'validation' split.")
        train_dataset = load_dataset(
            dataset_name, name=dataset_config, split='train',
            streaming=streaming, trust_remote_code=True
        )
        eval_dataset = load_dataset(
            dataset_name, name=dataset_config, split='validation',
            streaming=streaming, trust_remote_code=True
        )
    elif 'train' in available_splits:
        print("Validation split not found. Splitting 'train' dataset using take/skip.")
        if not streaming:
            print("Warning: Splitting non-streaming dataset using take/skip. Consider using train_test_split.")
            # For non-streaming, train_test_split is better, but we implement take/skip for consistency
            # as the config defaults to streaming=True.

        num_validation_samples = getattr(config.data, 'num_validation_samples', 50000) # Get from config or default
        print(f"Taking {num_validation_samples} samples for validation.")

        # Load the base 'train' split
        # Important: Load *once* then use take/skip on the *same* iterable object
        base_train_dataset = load_dataset(
            dataset_name, name=dataset_config, split='train',
            streaming=streaming, trust_remote_code=True
        )

        # Create eval and train sets using take and skip
        # Note: For streaming, this processes the beginning of the stream for both.
        eval_dataset = base_train_dataset.take(num_validation_samples)
        train_dataset = base_train_dataset.skip(num_validation_samples)

        # Ensure the results are still IterableDatasets if streaming
        if streaming:
             if not isinstance(train_dataset, IterableDataset):
                  print("Warning: train_dataset is not IterableDataset after skip().")
             if not isinstance(eval_dataset, IterableDataset):
                  print("Warning: eval_dataset is not IterableDataset after take().")

    else:
        raise ValueError(f"Dataset {dataset_name} (config: {dataset_config}) must contain at least a 'train' split. Found: {available_splits}")


    def tokenize_function(examples):
        # Tokenize the text. FineWeb might have a 'text' field. Adjust if needed.
        # Handle potential errors if 'text' field is missing or empty
        texts = [text for text in examples.get("text", []) if text]
        if not texts:
            # Return empty dict matching expected keys if no text
            return {"input_ids": np.array([], dtype=np.int32),
                    "attention_mask": np.array([], dtype=np.int32)}

        output = tokenizer(
            texts,
            padding="max_length", # Pad to max_seq_length
            truncation=True,
            max_length=config.data.max_seq_length,
            return_tensors="np" # Return NumPy arrays initially
        )

        return output # Return dict with input_ids, attention_mask

    # Apply tokenization lazily if streaming
    print("Tokenizing datasets...")
    # Use batched=False for streaming map if issues occur, but True is generally faster
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    eval_tokenized = eval_dataset.map(tokenize_function, batched=True)

    # Shuffle the training data buffer if streaming
    if config.data.streaming:
        train_tokenized = train_tokenized.shuffle(
            seed=config.training.seed,
            buffer_size=config.data.shuffle_buffer_size
        )
        print(f"Streaming training dataset shuffled with buffer size: {config.data.shuffle_buffer_size}")
        # Note: Shuffling eval set is usually not necessary or desired


    # --- Batching and JAX Conversion ---
    def numpy_to_jax(batch):
        """Converts a batch of numpy arrays to JAX arrays."""
        return jax.tree_util.tree_map(jnp.array, batch)

    def collate_fn(batch_list):
        """Collates a list of examples into a batch dictionary of numpy arrays."""
        if not batch_list: # Handle empty list
             return {}
        # Ensure all examples are dicts and have keys before proceeding
        if not all(isinstance(ex, dict) and ex for ex in batch_list):
             print("Error during collation: Found non-dict or empty item in batch_list.")
             return {}

        keys = batch_list[0].keys()
        batch = {}
        try:
            for k in keys:
                 # Check if key exists in all examples before stacking
                 if not all(k in example for example in batch_list):
                      print(f"Error during collation: Missing key '{k}' in some examples.")
                      return {}
                 batch[k] = np.stack([example[k] for example in batch_list])
            return batch
        except KeyError as e:
             print(f"Error during collation: Missing key {e} in an example.")
             return {}
        except ValueError as e:
             print(f"Error during collation (likely inconsistent shapes): {e}")
             for k in keys:
                 try:
                     shapes = [example[k].shape for example in batch_list]
                     if len(set(shapes)) > 1:
                         print(f"Inconsistent shapes for key '{k}': {shapes}")
                 except KeyError: # Should be caught above, but safeguard
                     pass
             return {}


    def data_generator(dataset, batch_size, dataset_name="unknown", drop_last=False):
        """
        Generator yielding batches of JAX arrays.

        Args:
            dataset: The tokenized dataset (iterable).
            batch_size: The desired batch size for each yield.
            dataset_name: Name for logging purposes.
            drop_last: If True, drop the last partial batch. Important for pmap.
        """
        buffer = []
        count = 0
        print(f"Starting data generator for {dataset_name} with batch size {batch_size} (drop_last={drop_last})...")
        for example in dataset:
            count += 1
            # Filter out potentially empty examples after tokenization/mapping
            if example and 'input_ids' in example and example['input_ids'].size > 0:
                 buffer.append(example)
                 if len(buffer) == batch_size:
                     batch = collate_fn(buffer)
                     if batch: # Only yield if collation was successful
                         yield numpy_to_jax(batch)
                     else:
                         print(f"Skipping batch in {dataset_name} due to collation error.")
                     buffer = [] # Reset buffer regardless of collation success

        # Handle the last batch
        if buffer and not drop_last:
            print(f"Yielding final partial batch of size {len(buffer)} for {dataset_name}.")
            batch = collate_fn(buffer)
            if batch:
                yield numpy_to_jax(batch)
            else:
                print(f"Skipping final batch in {dataset_name} due to collation error.")
        elif buffer and drop_last:
             print(f"Dropping final partial batch of size {len(buffer)} for {dataset_name}.")

        print(f"Data generator for {dataset_name} finished after processing {count} examples.")


    # Calculate global batch size across all devices
    global_batch_size = config.data.batch_size_per_device * jax.device_count()
    print(f"Global batch size for generators: {global_batch_size}")

    # Use drop_last=True for the training loader if using pmap, as pmap typically requires
    # all batches to have the same size across all devices.
    # For evaluation, dropping the last batch might be acceptable or you might need padding.
    is_distributed = config.training.use_distributed and jax.device_count() > 1
    train_loader = data_generator(train_tokenized, global_batch_size, "train", drop_last=is_distributed)
    eval_loader = data_generator(eval_tokenized, global_batch_size, "eval", drop_last=is_distributed)

    return train_loader, eval_loader, tokenizer
