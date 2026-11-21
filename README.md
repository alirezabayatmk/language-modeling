# Language Modeling

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24409642.svg)](https://doi.org/10.5281/zenodo.24409642)

Interactive simulations and short, runnable code on how language models work, by Alireza Bayat Makou.

**Website:** https://alirezabayatmk.github.io/language-modeling/

| Piece | What it is |
|---|---|
| [Training a Transformer](https://alirezabayatmk.github.io/language-modeling/training-a-transformer/) | A 93-slide interactive simulation of one training run of a tiny GPT: tokenizer, embeddings, attention, MLP, loss, backpropagation, AdamW updates and sampling. Every number comes from [`tiny_gpt.py`](training-a-transformer/tiny_gpt.py). |

## Run the code

```bash
pip install torch
python training-a-transformer/tiny_gpt.py   # CPU only, about 20 s on a laptop
```

The script reproduces the numbers shown in the simulation: the losses, the trained model's predictions and the generated text.

## License

- Content (text, simulations, figures and their interface): [CC BY-NC-SA 4.0](LICENSE).
- Code (`*.py`): [MIT](LICENSE-CODE).

Copyright, which license covers which files, and third-party attributions: see [NOTICE](NOTICE).

## Citation

See [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this repository" button), or:

```bibtex
@misc{bayatmakou2026trainingtransformer,
  author = {Bayat Makou, Alireza},
  title  = {Training a Transformer: An Interactive Simulation},
  year   = {2026},
  doi    = {10.5281/zenodo.24409642},
  url    = {https://alirezabayatmk.github.io/language-modeling/training-a-transformer/}
}
```

## Credits

- The model and the training loop build on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy (MIT License).
- The simulation's interface was built with the help of Claude (Anthropic).

## About this repository

The pages are built from private sources and committed here as self-contained HTML files; there is no build step.
GitHub Pages serves the repository as it is (see `.github/workflows/pages.yml`).
