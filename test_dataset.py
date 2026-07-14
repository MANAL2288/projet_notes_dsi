from datasets import load_dataset

ds = load_dataset(
    "chainyo/rvl-cdip",
    split="train",
    streaming=True
)

it = iter(ds)
example = next(it)
print(example.keys())
print(example["label"])
print(ds.features["label"].names)