"""
Voting strategies for DeepThinkLLM.
"""

from collections import Counter, defaultdict
from typing import List, Optional, Dict, Any, Union, Tuple
import numpy as np
from .reliability import (
    compute_two_stage_thresholds,
    extract_stages_from_trace,
    aggregate_highest_k_entropy,
    compute_trace_entropy_weight
)

def simple_majority_vote(answers: List[str]) -> Optional[str]:
    """
    Perform simple majority voting on a list of answers.
    
    Args:
        answers: A list of answer strings.
        
    Returns:
        The most common answer string, or None if the list is empty.
    """
    if not answers:
        return None
        
    vote_counts = Counter(answers)
    return vote_counts.most_common(1)[0][0]


def weighted_majority_vote(answers: List[str], weights: List[float]) -> Optional[str]:
    """
    Perform weighted majority voting.
    
    Args:
        answers: List of answer strings.
        weights: Corresponding weights (confidences) for each answer.
        
    Returns:
        The answer with the highest accumulated weight.
    """
    if not answers or not weights:
        return None
    
    if len(answers) != len(weights):
        raise ValueError(f"Length mismatch: answers ({len(answers)}) vs weights ({len(weights)})")
    
    answer_weights: Dict[str, float] = defaultdict(float)
    
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] += float(weight)
    
    if not answer_weights:
        return None
    
    return max(answer_weights.keys(), key=lambda x: answer_weights[x])

def evaluate_voting_results(
    voting_results: Dict[str, Any], 
    ground_truth: str,
    equal_func_handler: callable
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate voting results against ground truth.
    
    Args:
        voting_results: Dictionary mapping method names to result dictionaries.
        ground_truth: The ground truth answer string.
        equal_func_handler: Function to compare answer with ground truth.
        
    Returns:
        Dictionary containing evaluation metrics for each method.
    """
    evaluation = {}
    gt_str = str(ground_truth).strip()
    
    for method, res in voting_results.items():
        if res and res.get('answer') is not None:
            ans = str(res['answer']).strip()
            evaluation[method] = {
                'answer': ans,
                'is_correct': equal_func_handler(ans, gt_str),
                'num_votes': res.get('num_votes', 0)
            }
        else:
            evaluation[method] = {
                'answer': None, 
                'is_correct': False, 
                'num_votes': 0
            }
    return evaluation


# ============= CONFIDENCE CALCULATION FUNCTIONS - REMOVED per user request =============

def reliable_filtering_and_voting(
    traces: List[Dict[str, Any]],
    filtering_ratio: float = 0.4,
    gamma: float = 1.0,
    thresholds: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Two-stage filtering with weighted majority voting using reliability (negative entropy).
    Uses precomputed token_entropies in traces.
    """
    if not traces:
        return {
            'answer': None,
            'num_votes': 0,
            'filtered_traces': [],
            'thresholds': (0.0, 0.0)
        }
    
    # Compute or use precomputed thresholds and optimal_k
    if thresholds is None:
        thinking_threshold, reasoning_threshold, optimal_k = compute_two_stage_thresholds(traces, filtering_ratio)
    else:
        thinking_threshold, reasoning_threshold = thresholds
        optimal_k = kwargs.get('optimal_k', 32)
    
    # Dual-level Filtering
    filtered_traces = []
    answers, weights = [], []
    for trace in traces:
        stages = extract_stages_from_trace(trace, trace.get('token_entropies', []))
        
        # Compute reliability (negative entropy) for each stage
        s1_entropy = aggregate_highest_k_entropy(stages['thinking_stage'], optimal_k)
        s2_entropy = aggregate_highest_k_entropy(stages['reasoning_stage'], optimal_k)
        
        s1_rel = -s1_entropy
        s2_rel = -s2_entropy
        
        has_s1 = len(stages['thinking_stage']) > 0
        has_s2 = len(stages['reasoning_stage']) > 0
        
        # Filter based on reliability thresholds
        passes_filter = False
        if has_s1 and has_s2:
            passes_filter = (s1_rel >= thinking_threshold) and (s2_rel >= reasoning_threshold)
        elif has_s1:
            passes_filter = s1_rel >= reasoning_threshold
        
        if passes_filter:
            # Use precomputed weight if available, otherwise compute
            if 'weight' in trace:
                weight = trace['weight']
            else:
                weight = compute_trace_entropy_weight(
                    s1_rel, s2_rel, thinking_threshold, reasoning_threshold, 
                    has_s1, has_s2, gamma
                )
            
            if trace.get('extracted_answer'):
                filtered_traces.append(trace)
                answers.append(trace['extracted_answer'])
                weights.append(weight)
    
    if answers and weights:
        voted_answer = weighted_majority_vote(answers, weights)
    else:
        all_answers = [t.get('extracted_answer') for t in traces]
        valid_answers = [a for a in all_answers if a] or all_answers
        voted_answer = simple_majority_vote(valid_answers)
    
    return {
        'answer': voted_answer,
        'num_votes': len(filtered_traces),
        'filtered_traces': filtered_traces,
        'thresholds': (thinking_threshold, reasoning_threshold),
        'optimal_k': optimal_k
    }


# ============= VOTING RESULT COMPUTATION =============

def compute_all_voting_results(traces: List[Dict[str, Any]], gamma: float = 1.0, filtering_ratio: float = 0.4) -> Dict[str, Any]:
    """Compute results for all voting methods"""
    if not traces:
        return {method: None for method in [
            'Self-Consistency', 'Reliable_TWI'
        ]}
    
    # Get answers for ALL traces
    answers = [t.get('extracted_answer') for t in traces]
    
    voting_results = {}
    
    # 1. Simple majority vote
    voting_results['Self-Consistency'] = {'answer': simple_majority_vote(answers), 'num_votes': len(traces)}
    
    # 2. Dual-stage filtering (offline mode)
    RTWI_result = reliable_filtering_and_voting(traces, filtering_ratio=filtering_ratio, gamma=gamma)
    voting_results['Reliable_TWI'] = {
        'answer': RTWI_result['answer'],
        'num_votes': RTWI_result['num_votes'],
        'thresholds': RTWI_result['thresholds'],
        'optimal_k': RTWI_result.get('optimal_k', 32)
    }
    
    return voting_results


def compute_all_voting_results_online(
    traces: List[Dict[str, Any]],
    thresholds: Optional[Tuple[float, float]] = None,
    optimal_k: Optional[int] = None,
    gamma: float = 1.0,
    filtering_ratio: float = 0.4
) -> Dict[str, Any]:
    """Online mode voting with precomputed thresholds."""
    voting_results = compute_all_voting_results(traces, gamma=gamma, filtering_ratio=filtering_ratio)
    
    if traces and thresholds is not None:
        RTWI_result = reliable_filtering_and_voting(
            traces, filtering_ratio=filtering_ratio, gamma=gamma, thresholds=thresholds, optimal_k=optimal_k
        )
        voting_results['Reliable_TWI'] = {
            'answer': RTWI_result['answer'],
            'num_votes': RTWI_result['num_votes'],
            'thresholds': thresholds,
            'optimal_k': optimal_k if optimal_k else RTWI_result.get('optimal_k', 32)
        }
    return voting_results