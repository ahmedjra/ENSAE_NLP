# BERT-based Named Entity Recognition for Biomedical Text

This repository contains implementations for nested NER on the GENIA biomedical dataset using BERT-based models and prompt-based methods.

## Project Structure

```
├── data/                      # Dataset files
│   └── dataset/               # GENIA dataset (train, dev, test splits)
├── src/                       # Source code
│   └── evaluate.py            # Evaluation metrics for nested NER              
│── span_bert_analysis.ipynb   # Analysis of BERT models results
|── dataset_analysis.ipynb     # Analysis of dataset
├── bert_test.py               # Main BERT NER pipeline implementation
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## Overview

This project implements and evaluates several approaches to nested named entity recognition on the GENIA biomedical dataset:

1. **BERT-based Models**:
   - Base models: BioBERT, PubMedBERT, SciBERT
   - Fine-tuned versions of these models

2. **Prompt-based Methods**:
   - Base prompting
   - Recursive prompting
   - Extraction-classification
   - Flat QA
   - Nested QA
   - Structure QA

## Dataset

The GENIA dataset is used for all experiments, with the following splits:
- train.json: 15,023 examples with 46,142 entities
- dev.json: 1,669 examples with 4,367 entities
- test.json: 1,854 examples with 5,506 entities

Each entry contains tokens, entity annotations with start/end positions and entity types (protein, DNA, RNA, cell_type, cell_line).

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Code

### Data Analysis

To analyze the GENIA dataset and generate visualizations:



use the Jupyter notebook for interactive analysis:

```bash
jupyter notebook notebooks/dataset_analysis.ipynb
```

### BERT NER Pipeline

To run the complete BERT NER pipeline (base models, fine-tuning, and evaluation):

```bash
python bert_test.py
```

### Simulated Results


To analyze the results:

```bash
jupyter notebook notebooks/span_bert_analysis.ipynb
```

## Evaluation Metrics

We use three different F1 score variants to evaluate nested NER performance:

1. **Flat F1**: Standard F1 score for NER, ignoring nesting
2. **Nested F1**: Considers the full nested structure and evaluates all entities
3. **Nesting F1**: Focuses specifically on correctly identifying nesting relationships

# ENSAE_NLP
