Thanks to:
- BPE: https://zhuanlan.zhihu.com/p/714899440


Datasets:
- Wikipedia-en: https://huggingface.co/datasets/izumi-lab/wikipedia-en-20230720
- Great-books: https://huggingface.co/datasets/erikanesse/great_books
- Open-text-books: https://huggingface.co/datasets/izumi-lab/open-text-books
- https://huggingface.co/datasets/datablations/c4-subsets


python -u -m src.data.prepare_datasets
python -u -m src.data.bpe_tokenizer


