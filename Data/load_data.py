
from datasets import load_dataset, load_from_disk
datasets = load_dataset("code_x_glue_ct_code_to_text", 'python', split="train")
print(datasets.features)

keys = datasets.features.keys()

# Print the first row for each feature
for key in keys:
    print(f"{key}: {datasets[key][0]}")