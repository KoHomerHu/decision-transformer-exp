# decision-transformer-exp

Implementation of [decision transformer](https://arxiv.org/abs/2106.01345) in PyTorch.

## Transformer Class

The Transformer class in ```decision_transformer.py``` was originally adapted from [this tutorial](https://nlp.seas.harvard.edu/annotated-transformer/) and [this repo](https://github.com/hyunwoongko/transformer/tree/master). 

Since we only need the transformer to predict one step ahead, I found that it would be better to use a decoder-only (i.e. causal) transformer model similar to [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). Meantime, I have replaced the part that embeds tokens with an ANN, because we are not dealing with words.

## Decision-Transformer Class

To be implemented.