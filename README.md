# Pay attention experiments

## Overview

This repository contains the code and data used for the experiments detailed in our research paper, which investigates the **GUIDE** framework—a novel method for modifying attention mechanisms within large language models (LLMs) to improve their performance across various tasks. Our primary innovation is the **Influence** metric, a heuristic designed to measure the impact of specific instruction tokens on the model's outputs. The repository includes scripts for running experiments in text summarization, information retrieval (Needle in a Haystack), and structured data generation (JSON format).

## Table of Contents
- [Background](#background)
- [Experiments](#experiments)
  - [1. Summarization in French](#1-summarization-in-french)
  - [2. Needle in a Haystack](#2-needle-in-a-haystack)
  - [3. JSON Generation](#3-json-generation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Background

The GUIDE framework introduces a novel approach to enhancing attention mechanisms within LLMs. By adjusting the attention strength using a parameter `∆attn`, we aim to improve the model’s focus on critical instruction tokens, thereby influencing the overall output quality. This repository includes code for calculating the **Influence** metric, which helps in quantifying the impact of these modifications.

### Key Concepts

- **Influence Metric**: Quantifies the impact of instruction tokens on the model's output.
- **Attention Rollout**: A baseline method that we compare against the Influence metric.
- **∆attn**: A tunable parameter that controls the strength of attention on specified tokens.

## Experiments

### 1. Summarization in French

This experiment evaluates the effect of modifying attention mechanisms on the model’s ability to generate text in French when given English instructions. We use the **OpenWebText** dataset, where the model generates summaries in French based on the prompt "Summarize in French."

- **Metrics**:
  - Language Accuracy: Whether the summary is in French.
  - BERTScore: Cosine similarity between generated and reference summaries.

### 2. Needle in a Haystack

In this experiment, we test the model's ability to retain and retrieve specific information embedded within a text. The model is queried about this information (the "needle") after processing a lengthy context (the "haystack").

- **Metrics**:
  - Retrieval Accuracy: The correctness of the model's response related to the needle information.
  - Influence Distribution: Analysis of how Influence scores vary with needle placement within the text.

### 3. JSON Generation

This experiment tests the model’s capability to generate outputs in a specific structured format (JSON) by following a predefined schema. We use historical book data from the **BL Books** dataset as input.

- **Metrics**:
  - Jaccard Index: Measures the overlap between generated JSON keys and the schema.
  - Output Format Accuracy: Whether the generated output matches the expected JSON format.

## Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/netop-team/pay_attention_experiment.git
cd pay_attention_experiment
pip install -r requirements.txt
```

## Usage

### Running Experiments

To run any of the experiments, use the following command:

```bash
python summarize.py --delta 1 --layers all --max_new_tokens 200 --model_name mistral --num_generations 10
```


## Results

Results from each experiment, including metrics and visualizations, are stored in the `results/` directory. The key findings from our experiments include:

- **Summarization in French**: Fine-tuning attention with `∆attn` improves the likelihood of generating correct French summaries, especially for longer contexts.
- **Needle in a Haystack**: Applying `∆attn` improves the model's ability to retrieve specific information, particularly when the information is located at the beginning or end of a text.
- **JSON Generation**: Optimal `∆attn` values significantly increase the accuracy of generating correct JSON outputs.

## Contributing

We welcome contributions to improve the GUIDE framework or the experimental setup. Please fork the repository and create a pull request with your enhancements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.