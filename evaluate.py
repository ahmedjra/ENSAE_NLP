from typing import Dict, List, Tuple, Set, Any
import numpy as np
from collections import defaultdict

def find_token_indices(tokens: List[str], entity_text: str) -> List[Tuple[int, int]]:
    """
    Find all occurrences of an entity text in a tokenized sentence.
    
    Args:
        tokens: List of tokens in the sentence
        entity_text: The entity text to find
        
    Returns:
        List of (start_idx, end_idx) tuples representing token spans
    """
    entity_tokens = entity_text.split()
    if not entity_tokens:
        return []
    
    matches = []
    n_tokens = len(tokens)
    n_entity_tokens = len(entity_tokens)
    
    # Sliding window search
    for i in range(n_tokens - n_entity_tokens + 1):
        if ' '.join(tokens[i:i+n_entity_tokens]) == entity_text:
            # Return indices as [start, end) - end is exclusive
            matches.append((i, i+n_entity_tokens))
    
    return matches

def convert_predictions_to_token_indices(
    predicted_entities: Dict[str, str], 
    tokens: List[str]
) -> List[Dict[str, Any]]:
    """
    Convert predicted entities (text spans) to token indices format.
    
    Args:
        predicted_entities: Dictionary of {entity_text: entity_type}
        tokens: List of tokens in the sentence
        
    Returns:
        List of dictionaries with keys 'start', 'end', and 'type'
    """
    converted_entities = []
    sentence = ' '.join(tokens)
    
    for entity_text, entity_type in predicted_entities.items():
        # Skip if entity text not in sentence
        if entity_text not in sentence:
            continue
            
        # Find all occurrences of this entity text
        spans = find_token_indices(tokens, entity_text)
        
        # Add each occurrence as a separate entity
        for start, end in spans:
            converted_entities.append({
                'start': start,
                'end': end,  # end is exclusive
                'type': entity_type
            })
    
    return converted_entities

def calculate_flat_f1(
    predicted_entities: List[Dict[str, Any]], 
    gold_entities: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate standard F1 score for NER, ignoring nesting.
    
    Args:
        predicted_entities: List of predicted entities with 'start', 'end', 'type'
        gold_entities: List of gold entities with 'start', 'end', 'type'
        
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    # Convert to sets of (start, end, type) tuples for comparison
    pred_set = {(e['start'], e['end'], e['type']) for e in predicted_entities}
    gold_set = {(e['start'], e['end'], e['type']) for e in gold_entities}
    
    # Calculate true positives, false positives, false negatives
    true_positives = len(pred_set.intersection(gold_set))
    false_positives = len(pred_set - gold_set)
    false_negatives = len(gold_set - pred_set)
    
    # Calculate precision, recall, F1
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def calculate_nested_f1(
    predicted_entities: List[Dict[str, Any]], 
    gold_entities: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate F1 score considering the full nested structure.
    This is essentially the same as flat F1 but applied to all entities.
    
    Args:
        predicted_entities: List of predicted entities with 'start', 'end', 'type'
        gold_entities: List of gold entities with 'start', 'end', 'type'
        
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    # For nested F1, we use the same calculation as flat F1
    # The difference is in how we use it - nested F1 considers all entities
    return calculate_flat_f1(predicted_entities, gold_entities)

def identify_nesting_relations(entities: List[Dict[str, Any]]) -> Set[Tuple[int, int]]:
    """
    Identify nesting relations between entities.
    
    Args:
        entities: List of entities with 'start', 'end', 'type'
        
    Returns:
        Set of (container_idx, contained_idx) tuples representing nesting relations
    """
    nesting_relations = set()
    n = len(entities)
    
    # Compare each pair of entities
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
                
            # Entity i contains entity j if start_i <= start_j and end_i >= end_j
            if (entities[i]['start'] <= entities[j]['start'] and 
                entities[i]['end'] >= entities[j]['end']):
                nesting_relations.add((i, j))
    
    return nesting_relations

def calculate_nesting_f1(
    predicted_entities: List[Dict[str, Any]], 
    gold_entities: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate F1 score focusing on correctly identifying nesting relationships.
    
    Args:
        predicted_entities: List of predicted entities with 'start', 'end', 'type'
        gold_entities: List of gold entities with 'start', 'end', 'type'
        
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    # Identify nesting relations in predicted and gold entities
    pred_relations = identify_nesting_relations(predicted_entities)
    gold_relations = identify_nesting_relations(gold_entities)
    
    # Calculate true positives, false positives, false negatives
    true_positives = len(pred_relations.intersection(gold_relations))
    false_positives = len(pred_relations - gold_relations)
    false_negatives = len(gold_relations - pred_relations)
    
    # Calculate precision, recall, F1
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def evaluate_predictions(
    predicted_entities: Dict[str, str],
    gold_example: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate predicted entities against gold standard.
    
    Args:
        predicted_entities: Dictionary of {entity_text: entity_type}
        gold_example: Gold standard example with 'tokens' and 'entities'
        
    Returns:
        Dictionary with flat_f1, nested_f1, and nesting_f1 scores
    """
    # Convert predicted entities to token indices format
    tokens = gold_example['tokens']
    pred_entities = convert_predictions_to_token_indices(predicted_entities, tokens)
    gold_entities = gold_example['entities']
    
    # Calculate the three types of F1 scores
    flat_f1 = calculate_flat_f1(pred_entities, gold_entities)
    nested_f1 = calculate_nested_f1(pred_entities, gold_entities)
    nesting_f1 = calculate_nesting_f1(pred_entities, gold_entities)
    
    return {
        'flat_f1': flat_f1,
        'nested_f1': nested_f1,
        'nesting_f1': nesting_f1
    }

def calculate_average_scores(all_scores: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate average scores across multiple examples.
    
    Args:
        all_scores: List of score dictionaries from evaluate_predictions
        
    Returns:
        Dictionary with averaged scores
    """
    if not all_scores:
        return {}
        
    # Initialize with zeros
    avg_scores = {
        'flat_f1': {'precision': 0, 'recall': 0, 'f1': 0},
        'nested_f1': {'precision': 0, 'recall': 0, 'f1': 0},
        'nesting_f1': {'precision': 0, 'recall': 0, 'f1': 0}
    }
    
    # Sum up all scores
    for scores in all_scores:
        for metric in ['flat_f1', 'nested_f1', 'nesting_f1']:
            for key in ['precision', 'recall', 'f1']:
                avg_scores[metric][key] += scores[metric][key]
    
    # Divide by number of examples
    n = len(all_scores)
    for metric in avg_scores:
        for key in avg_scores[metric]:
            avg_scores[metric][key] /= n
    
    return avg_scores
