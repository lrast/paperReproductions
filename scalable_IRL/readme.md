# Imitating Language via Scalable Inverse Reinforcement Learning

by Wulfmeier et al, 2024.

This paper reframes LLM finetuning as an IRL problem, with the logits viewed as q-values. This means that the softmax logits become policy and the logsumexp normalizer corresponds to the value function.
