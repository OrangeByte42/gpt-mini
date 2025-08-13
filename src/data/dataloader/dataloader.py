import os
from datasets import load_dataset

dataset_dir: str = os.path.join(".", "data", "datasets")
ds1 = load_dataset("izumi-lab/wikipedia-en-20230720", cache_dir=os.path.join(dataset_dir, "wikipedia-en-20230720"))
ds2 = load_dataset("common-pile/arxiv_papers", cache_dir=os.path.join(dataset_dir, "arxiv_papers"))
ds3 = load_dataset("erikanesse/great_books", cache_dir=os.path.join(dataset_dir, "great_books"))
ds4 = load_dataset("izumi-lab/open-text-books", cache_dir=os.path.join(dataset_dir, "open_text_books"))

print(type(ds1), type(ds2), type(ds3), type(ds4))
print(ds1.keys(), ds2.keys(), ds3.keys(), ds4.keys())
print(type(ds1["train"]), type(ds2["train"]), type(ds3["train"]), type(ds4["train"]))

