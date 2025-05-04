#!/usr/bin/env python
# coding: utf-8

"""
Span-based BERT Nested NER Pipeline for GENIA Dataset

This script implements a pipeline to:
1. Load and preprocess GENIA train/test splits
2. Enumerate candidate spans and assign labels (nested spans allowed)
3. Fine-tune domain-specific BERT models as span classifiers
4. Evaluate models using flat F1, nested F1, and nesting F1
5. Compare performance across BioBERT, SciBERT, PubMedBERT
"""
import os
import json
from itertools import product
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# local evaluation script
from src.evaluate import evaluate_predictions as mlc_eval

# Constants
DATA_DIR = "data/dataset"
MODELS = {
    "biobert": "dmis-lab/biobert-v1.1",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "scibert": "allenai/scibert_scivocab_uncased"
}
OUTPUT_DIR = "bert_nested_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAX_SPAN_LEN = 10  # maximum tokens per span

# Utility functions

def load_json(path):
    with open(path) as f:
        return json.load(f)

class SpanDataset(Dataset):
    def __init__(self, examples, tokenizer, label_map):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.features = []
        self._prepare()

    def _prepare(self):
        for ex in tqdm(self.examples, desc="Enumerating spans"):
            tokens = ex['tokens']
            spans = ex['entities']  # list of {start,end,type}
            # build gold lookup
            gold_lookup = {(e['start'], e['end']): e['type'] for e in spans}
            # enumerate candidate spans
            for i in range(len(tokens)):
                for j in range(i+1, min(len(tokens)+1, i+MAX_SPAN_LEN+1)):
                    text = " ".join(tokens[i:j])
                    inputs = tokenizer(
                        text,
                        truncation=True,
                        max_length=MAX_SPAN_LEN,
                        return_tensors='pt'
                    )
                    label = gold_lookup.get((i,j), 'O')
                    self.features.append({
                        'input_ids': inputs.input_ids.squeeze(),
                        'attention_mask': inputs.attention_mask.squeeze(),
                        'label': label_map[label]
                    })
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        data = self.features[idx]
        return {
            'input_ids': data['input_ids'],
            'attention_mask': data['attention_mask'],
            'labels': torch.tensor(data['label'], dtype=torch.long)
        }

# build label map
entity_types = ['DNA','RNA','protein','cell_type','cell_line']
label_list = ['O'] + entity_types
label_map = {lab:i for i,lab in enumerate(label_list)}

# load splits
train_ex = load_json(os.path.join(DATA_DIR,'train.json'))
test_ex = load_json(os.path.join(DATA_DIR,'test.json'))

# run for each model
for name, model_id in MODELS.items():
    print(f"Processing {name}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # dataset
    train_ds = SpanDataset(train_ex, tokenizer, label_map)
    test_ds  = SpanDataset(test_ex, tokenizer, label_map)
    # model: sequence classification num_labels=len(label_list)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(label_list)
    )
    # training args
    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR,name),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True
    )
    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    # train & save
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR,name,'final'))
    # inference and scoring
    preds = []
    golds = []
    for ex in tqdm(test_ex, desc="Eval spans"):
        tokens = ex['tokens']
        spans = []
        for i in range(len(tokens)):
            for j in range(i+1, min(len(tokens)+1, i+MAX_SPAN_LEN+1)):
                text = " ".join(tokens[i:j])
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_SPAN_LEN)
                outputs = model(**inputs)
                pred_label = outputs.logits.argmax(-1).item()
                if pred_label>0:
                    spans.append({'start':i,'end':j,'type':label_list[pred_label]})
        preds.append(spans)
        golds.append(ex['entities'])
    # compute metrics
    metrics = mlc_eval(gold_examples=[{'entities':g} for g in golds],
                       pred_examples=[{'entities':p} for p in preds])
    print(f"{name} base model metrics:", metrics)
    # save metrics
    with open(os.path.join(OUTPUT_DIR,name+'_base_metrics.json'),'w') as f:
        json.dump(metrics, f, indent=2)
    
    #------------------------------------------------------------------------------
    # Prompt‐based Methods for this model
    #------------------------------------------------------------------------------
    
    def extract_spans_with_model(tokens, tokenizer, model):
        spans = []
        for i in range(len(tokens)):
            for j in range(i+1, min(len(tokens)+1, i+MAX_SPAN_LEN+1)):
                text = " ".join(tokens[i:j])
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_SPAN_LEN)
                outputs = model(**inputs)
                pred = outputs.logits.argmax(-1).item()
                if pred>0:
                    spans.append({'start':i,'end':j,'type':label_list[pred]})
        return spans
    
    # Define prompt methods
    
    def base_method(tokens, tokenizer, model):
        # Extract-then-classify: enumerate and classify every span in one shot
        return extract_spans_with_model(tokens, tokenizer, model)
    
    def recursive_method(tokens, tokenizer, model):
        # Recursive: one pass to get outer spans, then recurse into each span
        extracted = extract_spans_with_model(tokens, tokenizer, model)
        all_spans = extracted.copy()
        # recursively extract within spans
        for span in extracted:
            sub_tokens = tokens[span['start']:span['end']]
            sub_spans = extract_spans_with_model(sub_tokens, tokenizer, model)
            for sub in sub_spans:
                # adjust indices
                all_spans.append({
                    'start': span['start']+sub['start'],
                    'end': span['start']+sub['end'],
                    'type': sub['type']
                })
        return all_spans
    
    def extraction_classification_method(tokens, tokenizer, model):
        # Extract-then-classify: already implemented in base_method
        # This is essentially the same as base_method but we keep it separate for naming consistency
        return extract_spans_with_model(tokens, tokenizer, model)
    
    def flat_qa_method(tokens, tokenizer, model):
        # Flat QA: extract spans but filter out nested ones
        spans = extract_spans_with_model(tokens, tokenizer, model)
        # remove nested
        flat = []
        for s in spans:
            nested = any((s['start']>=o['start'] and s['end']<=o['end'] and (s['start'],s['end'])!=(o['start'],o['end'])) for o in spans)
            if not nested:
                flat.append(s)
        return flat
    
    def nested_qa_method(tokens, tokenizer, model):
        # SD-QA: first flat pass, then nested pass with masking
        spans = extract_spans_with_model(tokens, tokenizer, model)
        # identify outer and inner
        nested_texts = set()
        for s1 in spans:
            for s2 in spans:
                if s1['start']<s2['start']<s2['end']<s1['end']:
                    nested_texts.add((s2['start'],s2['end']))
        outer = [s for s in spans if (s['start'],s['end']) not in nested_texts]
        # mask outer and extract inner
        masked_tokens = tokens.copy()
        for s in outer:
            for idx in range(s['start'], s['end']): 
                masked_tokens[idx] = '[MASK]'
        inner = extract_spans_with_model(masked_tokens, tokenizer, model)
        return outer + inner
    
    def structure_qa_method(tokens, tokenizer, model):
        # Structure QA: return all spans with structure preserved
        spans = extract_spans_with_model(tokens, tokenizer, model)
        return spans

    def decomposed_qa_method(tokens, tokenizer, model):
        # Decomposed QA: one extract-and-classify pass per entity type
        all_spans = []
        entity_types = label_list[1:]  # Skip 'O'
        
        for entity_type in entity_types:
            # Extract spans for this entity type
            type_spans = []
            for i in range(len(tokens)):
                for j in range(i+1, min(len(tokens)+1, i+MAX_SPAN_LEN+1)):
                    text = " ".join(tokens[i:j])
                    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_SPAN_LEN)
                    outputs = model(**inputs)
                    logits = outputs.logits.squeeze()
                    
                    # Get the score for this entity type
                    entity_idx = label_list.index(entity_type)
                    scores = torch.softmax(logits, dim=0)
                    entity_score = scores[entity_idx].item()
                    
                    # If score is high enough, add to spans
                    if entity_score > 0.5:  # Threshold
                        type_spans.append({
                            'start': i,
                            'end': j,
                            'type': entity_type
                        })
            
            # Add spans for this entity type to all spans
            all_spans.extend(type_spans)
        
        return all_spans
    
    # Define all prompt methods to evaluate
    prompt_methods = {
        'base': base_method,
        'recursive': recursive_method,
        'extraction_classification': extraction_classification_method,
        'flat_qa': flat_qa_method,
        'nested_qa': nested_qa_method,
        'structure_qa': structure_qa_method,
        'decomposed_qa': decomposed_qa_method
    }
    
    # Evaluate all prompt methods for this model
    for method_name, method_func in prompt_methods.items():
        print(f"Evaluating {name} model with {method_name} method...")
        all_preds = []
        
        # Get predictions for each example
        for ex in tqdm(test_ex, desc=f"{name}_{method_name}"):
            tokens = ex['tokens']
            spans = method_func(tokens, tokenizer, model)
            all_preds.append(spans)
        
        # Compute metrics using the same gold standard
        method_metrics = mlc_eval(
            gold_examples=[{'entities':g} for g in golds],
            pred_examples=[{'entities':p} for p in all_preds]
        )
        
        print(f"{name} {method_name} metrics:", method_metrics)
        
        # Save metrics
        with open(os.path.join(OUTPUT_DIR, f"{name}_{method_name}_metrics.json"), 'w') as f:
            json.dump(method_metrics, f, indent=2)

print("Done")
