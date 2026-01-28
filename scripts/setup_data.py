from datasets import load_dataset

print("Loading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories")

print("Saving dataset...")
dataset.save_to_disk("data/tinystories")

print(f"Dataset saved! Train samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")