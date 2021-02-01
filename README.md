# HierarchicalAttention

This repository combines the use of Hierarchical Attention Network (HAN) with d3-based visualisations to give the user insight into what the network is detecting.

The HAN uses attention at both the sentence level and the word level and we can visualise the informative sentences and words in a sentence by extracting attention weights.

Below are the results for YahooAnswers dataset, using HuggingFace's DistilRoberta for tokenizing and frozen word embedding word and sentence attention layers added on. 

![HANYH](img/han.jpg?raw=true "Title")

Sources:

1. HAN Paper: [https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)
2. HAN Implementation I referred to to begin working (but have changed significantly since): [https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification)
3. D3 Visualisation was inspired by ecco: [https://jalammar.github.io/explaining-transformers/](https://jalammar.github.io/explaining-transformers/)