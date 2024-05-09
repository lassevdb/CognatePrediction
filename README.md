# Decoder-Only Cognate Prediction

## Introduction
This project introduces a decoder-only transformer based approach to cognate prediction, leveraging its capacity to handle sequential data and capture linguistic patterns. Cognates, words across different languages with shared origins, provide significant insights into language history and evolution. However, their prediction is challenging, primarily due to the subtle phonetic and semantic shifts over long periods of time. Our method employs a decoder-only architecture typically used in generative tasks, adapted to predict a cognate in one language, given its cognate in another related language. We train our model on a dataset of Romance language and Germanic language cognates. The results demonstrate non-trivial performance with interesting generalization patterns.

## This Repo contains
- **models**: Contains pre-trained models for Cognate Prediction
- **data**: Contains data and python helper files to process data
- **model**: Implementation of tokenizer and model
- **main.ipynb**: The Notebook from which to run models and analyze behavior

## Getting Started

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.7 or higher
- Most packages used are very common ('dependencies.txt')
- You might want to install the following:
- sacrebleu
- seaborn
- python-Levenshtein

### How to run

Simply open **main.ipynb** and run cells. There are explanations that guide through the notebooks functionality.



