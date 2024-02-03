# decision-transformer-exp

Implementation of [decision transformer](https://arxiv.org/abs/2106.01345) in PyTorch.

## Transformer Class

The Transformer class in ```decision_transformer.py``` was originally adapted from [this tutorial](https://nlp.seas.harvard.edu/annotated-transformer/) and [this repo](https://github.com/hyunwoongko/transformer/tree/master). 


Modifications: 

- [x] Decision transformer requires a decoder-only (i.e. causal) transformer variant same with that of the [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) since we only need to predict the next action. 

- [x] I have replaced the part that embeds tokens with an ANN, since we are not dealing with words.

[Why next-token prediction is enough for AGI - Ilya Sutskever](https://www.youtube.com/watch?v=YEUclZdj_Sc): 

> "Because if you think about it, what does it mean to predict the next token well enough? It's actually a much deeper question than it seems. Predicting the next token well means that you understand the underlying reality that led to the creation of that token. It's not statistics. Like it is statistics but what is statistics? In order to understand those statistics to compress them, you need to understand what is it about the world that creates this set of statistics? And so then you say — Well, I have all those people. What is it about people that creates their behaviors? Well they have thoughts and their feelings, and they have ideas, and they do things in certain ways. All of those could be deduced from next-token prediction."

## DecisionTransformer

The decision transformer is implemented for discrete actions and without considering the timestep.

The input is the latest state, reward-to-go, and memory; and outputs the predicted action and a new memory which concatenates the action.

The initial reward-to-go needs to be manually setup, but for path planning, maybe a distribution of the reward-to-go conditioned on the intial state may be learned, as suggest in Appendix A.3 in the paper of decision transformer.

**Remark.** The original DecisionTransformer class implemented is not used because for some reason the model never converges (probably due to wrongly copying tensors during training time). 

## To-do List

- [ ] Modify state representation and collect more data
- [ ] Implement and train a probabilistic neural network that outputs a Gaussian distribution of the predicted reward-to-go, apply some prior (e.g. Gamma) to indicate the preference of optimal paths.


