# decision-transformer-exp

Implementation of [decision transformer](https://arxiv.org/abs/2106.01345) in PyTorch.

## Transformer Class

The Transformer class in ```decision_transformer.py``` was originally adapted from [this tutorial](https://nlp.seas.harvard.edu/annotated-transformer/) and [this repo](https://github.com/hyunwoongko/transformer/tree/master). 


Modifications: 

    - Decision transformer requires a decoder-only (i.e. causal) transformer variant same with that of the [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) since we only need to predict the next action. 
    
    - I have replaced the part that embeds tokens with an ANN, since we are not dealing with words.

[Why next-token prediction is enough for AGI - Ilya Sutskever](https://www.youtube.com/watch?v=YEUclZdj_Sc): 

> "Because if you think about it, what does it mean to predict the next token well enough? It's actually a much deeper question than it seems. Predicting the next token well means that you understand the underlying reality that led to the creation of that token. It's not statistics. Like it is statistics but what is statistics? In order to understand those statistics to compress them, you need to understand what is it about the world that creates this set of statistics? And so then you say — Well, I have all those people. What is it about people that creates their behaviors? Well they have thoughts and their feelings, and they have ideas, and they do things in certain ways. All of those could be deduced from next-token prediction."

## DecisionTransformer Class

The decision transformer is implemented for discrete actions and without considering the timestep.

Label-smoothing and other techniques may also be considered.