"""
Reliability analysis and filtering for DeepThinkLLM.
"""

import math
import heapq
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict

# ============= BASIC CONFIDENCE METRICS =============

def compute_confidence(logprobs: List[Dict]) -> List[float]:
    """Compute confidence score from logprobs and return only confidence values."""
    # Handle cases where logprobs might be None
    if not logprobs:
        return []
        
    return [round(-np.mean([lp.logprob for lp in t.values()]), 3) for t in logprobs if t]


def compute_least_grouped(confs: List[float], group_size: int) -> List[float]:
    """
    Compute sliding window mean confidence.
    Returns the mean confidence of each window.
    """
    if not confs:
        return [0.0]
    if len(confs) < group_size:
        return [sum(confs) / len(confs)]
    return [round(sum(confs[i:i + group_size]) / group_size, 3) 
            for i in range(len(confs) - group_size + 1)]


def calculate_mean_confidence(trace: Dict[str, Any]) -> float:
    """Calculate mean confidence from confs in a trace."""
    try:
        confs = trace.get('confs', [])
        if confs:
            return float(np.mean(confs))
        return 0.0
    except Exception:
        return 0.0


def calculate_tail_confidence(trace: Dict[str, Any], tail_tokens: int = 32) -> float:
    """Calculate mean confidence from the last N tokens."""
    try:
        confs = trace.get('confs', [])
        if confs:
            tail_confs = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
            return float(np.mean(tail_confs))
        return 0.0
    except Exception:
        return 0.0


def calculate_bottom_window_confidence(trace: Dict[str, Any], window_size: int = 32, bottom_percent: float = 0.1) -> float:
    """
    Calculate mean confidence from sliding windows, return average of bottom percentile.
    If bottom_percent is -1, returns the minimum window confidence.
    """
    try:
        confs = trace.get('confs', [])
        if not confs:
            return 0.0
            
        if len(confs) < window_size:
            return float(np.mean(confs))
        
        # Calculate window means
        window_means = []
        current_sum = sum(confs[:window_size])
        window_means.append(current_sum / window_size)
        
        for i in range(1, len(confs) - window_size + 1):
            current_sum = current_sum - confs[i-1] + confs[i + window_size - 1]
            window_means.append(current_sum / window_size)
        
        if not window_means:
            return 0.0
        
        if bottom_percent == -1:  # Min window
            return min(window_means)
        
        num_bottom = max(1, int(len(window_means) * bottom_percent))
        if num_bottom == 1:
            return min(window_means)
        else:
            # Use partition for O(N) selection instead of O(N log N) sort
            bottom_means = np.partition(window_means, num_bottom-1)[:num_bottom]
            return float(np.mean(bottom_means))
        
    except Exception:
        return 0.0


def filter_top_confidence(traces: List[Dict[str, Any]], confidence_type: str = 'tail', top_percent: float = 0.1) -> List[Dict[str, Any]]:
    """Filter traces to keep only the top ones based on a confidence metric."""
    if not traces: return []
    
    fmap = {
        'mean': calculate_mean_confidence, 
        'tail': calculate_tail_confidence, 
        'bottom_window': calculate_bottom_window_confidence, 
        'min_window': lambda t: calculate_bottom_window_confidence(t, bottom_percent=-1)
    }
    
    scoring_func = fmap.get(confidence_type, calculate_mean_confidence)
    confs = [scoring_func(t) for t in traces]
    
    # Calculate threshold (keep top X%)
    thresh = np.percentile(confs, (1 - top_percent) * 100)
    
    return [t for t, c in zip(traces, confs) if c >= thresh]


# ============= ENTROPY AND RELIABILITY =============

def calculate_token_entropies(logprobs_data: List[Any], k: int = 10) -> List[float]:
    """Compute token-level entropies: -sum(p * log(p)) after normalizing top-k."""
    entropies = []
    for entry in (logprobs_data or []):
        if not entry:
            entropies.append(0.0)
            continue
            
        vals = entry.values() if isinstance(entry, dict) else entry
        # Extract probabilities from logprobs
        probs = np.array([math.exp(getattr(v, 'logprob', v)) for v in vals][:k])
        probs = probs[probs > 0]
        
        if len(probs) > 0:
            probs /= probs.sum()  # Normalize
            entropy = -np.sum(probs * np.log(probs))
            entropies.append(round(float(entropy), 6))
        else:
            entropies.append(0.0)
    return entropies


def extract_stages_from_trace(trace: Dict[str, Any], token_entropies: List[float]) -> Dict[str, List[float]]:
    """
    Extract thinking_stage and reasoning_stage entropies based on turn_boundaries.
    Assumes standard DeepThink multi-turn structure (thinking -> reasoning).
    """
    turn_boundaries = trace.get('turn_boundaries', [])
    stages = {'thinking_stage': [], 'reasoning_stage': []}
    
    if not turn_boundaries:
        # If no boundaries, assume entire trace is thinking stage (or handle accordingly)
        stages['thinking_stage'] = token_entropies
        return stages
    
    if len(turn_boundaries) == 1:
        start, end = turn_boundaries[0]
        if start is not None and end is not None:
            stages['thinking_stage'] = token_entropies[start:end + 1]
        return stages
    
    # Multiple turns:
    # Everything up to the second-to-last turn is 'thinking'
    second_last_end = turn_boundaries[-2][1]
    stages['thinking_stage'] = token_entropies[:second_last_end + 1]
    
    # The last turn is 'reasoning'
    final_turn = turn_boundaries[-1]
    if final_turn[0] is not None and final_turn[1] is not None:
        stages['reasoning_stage'] = token_entropies[final_turn[0]:final_turn[1] + 1]
    
    return stages


def extract_all_stages(trace: Dict[str, Any], token_entropies: List[float]) -> List[List[float]]:
    """Extract all stages/segments based on turn boundaries."""
    turn_boundaries = trace.get('turn_boundaries', [])
    
    if not turn_boundaries:
        return [token_entropies]
    
    stages = []
    for start, end in turn_boundaries:
        if start is not None and end is not None:
            stages.append(token_entropies[start:end + 1])
    
    return stages if stages else [token_entropies]


class IncrementalTopKMean:
    """Efficiently maintain mean of top-k largest values incrementally."""
    def __init__(self, k: int):
        self.k = k
        self.heap = []  # Min-heap to store k largest values
        self.current_sum = 0.0
        
    def add(self, value: float):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, value)
            self.current_sum += value
        else:
            # If value is larger than the smallest of the top-k (heap root)
            if value > self.heap[0]:
                removed = heapq.heapreplace(self.heap, value)
                self.current_sum += (value - removed)
                
    def get_mean(self) -> float:
        if not self.heap:
            return 0.0
        return self.current_sum / len(self.heap)


def aggregate_highest_k_entropy(values: List[float], k: int) -> float:
    """Compute the mean of the top-k highest entropy values."""
    if not values:
        return 0.0
    num = min(k, len(values))
    if num == 0:
        return 0.0
    # Use partition for efficiency
    highest_tokens = np.partition(values, -num)[-num:]
    return float(np.mean(highest_tokens))


def compute_trace_entropy_weight(
    s1_rel: float, s2_rel: float, 
    thinking_threshold: float, reasoning_threshold: float, 
    has_s1: bool, has_s2: bool, gamma: float = 1.0
) -> float:
    """Calculate the weight for a trace based on its reliability scores."""
    epsilon = 1e-6
    
    if has_s1 and has_s2:
        # Two-stage case: weight based on stage2 vs stage1 reliability difference
        rel_gap = max(s2_rel - s1_rel, 0.0)
        trace_rel = (abs(s2_rel + s1_rel) + epsilon) * gamma
        return float(np.exp(rel_gap / trace_rel))
    
    elif has_s1:
        # Single-stage case: weight based on threshold difference
        rel_gap = max(reasoning_threshold - thinking_threshold, 0.0)
        trace_rel = (abs(2 * s1_rel) + epsilon) * gamma
        return float(np.exp(rel_gap / trace_rel))
    
    else:
        return 0.0


def find_optimal_k(two_stage_traces: List[Dict[str, Any]]) -> int:
    """Find optimal k for aggregation by maximizing thinking-reasoning entropy difference."""
    if not two_stage_traces:
        return 32
    
    try:
        trace_data = []
        lengths = []
        
        for trace in two_stage_traces:
            entropies = trace.get('token_entropies', [])
            stages = extract_stages_from_trace(trace, entropies)
            s1, s2 = stages['thinking_stage'], stages['reasoning_stage']
            
            if s1 and s2:
                trace_data.append((sorted(s1, reverse=True), sorted(s2, reverse=True)))
                lengths.append(len(s1) + len(s2))
        
        if not trace_data:
            return 32
            
        mean_len = np.mean(lengths)
        k_min = max(1, int(0.1 * mean_len))
        k_max = min(int(0.4 * mean_len), int(np.min(lengths))) if lengths else 100
        
        # Avoid empty range
        if k_max <= k_min:
             k_max = k_min + 1

        k_candidates = sorted(list(set(np.linspace(k_min, k_max, 10, dtype=int))))
        
        best_k, best_diff = 32, -np.inf
        for k in k_candidates:
            if k <= 0: continue
            diffs = []
            for s1_sorted, s2_sorted in trace_data:
                m1 = np.mean(s1_sorted[:min(k, len(s1_sorted))])
                m2 = np.mean(s2_sorted[:min(k, len(s2_sorted))])
                diffs.append(m1 - m2)
            
            avg_diff = np.mean(diffs)
            if avg_diff > best_diff:
                best_diff, best_k = avg_diff, k
        return int(best_k)
    except Exception:
        return 32


def compute_two_stage_thresholds(
    warmup_traces: List[Dict[str, Any]], 
    filtering_ratio: float = 0.5
) -> Tuple[float, float, int]:
    """Compute reliability thresholds and optimal_k from warmup traces."""
    if not warmup_traces:
        return 0.0, 0.0, 32
    
    two_stage_traces = [t for t in warmup_traces if len(t.get('turn_boundaries', [])) > 1]
    optimal_k = find_optimal_k(two_stage_traces) if two_stage_traces else 32
    
    s1_rel_vals, s2_rel_vals = [], []
    for t in warmup_traces:
        stages = extract_stages_from_trace(t, t.get('token_entropies', []))
        s1_entropy = aggregate_highest_k_entropy(stages['thinking_stage'], optimal_k)
        s2_entropy = aggregate_highest_k_entropy(stages['reasoning_stage'], optimal_k)
        
        if stages['reasoning_stage']:
            # Two-stage trace: both stages exist
            s1_rel_vals.append(-s1_entropy)
            s2_rel_vals.append(-s2_entropy)
        elif stages['thinking_stage']:
            # Single-stage trace
            s2_rel_vals.append(-s1_entropy)
    
    if not s1_rel_vals and not s2_rel_vals:
        return 0.0, 0.0, 32

    # Calculate thresholds from reliability percentiles
    thinking_threshold = float(np.percentile(s1_rel_vals, filtering_ratio * 100)) if s1_rel_vals else 0.0
    reasoning_threshold = float(np.percentile(s2_rel_vals, filtering_ratio * 100)) if s2_rel_vals else 0.0
    
    return thinking_threshold, reasoning_threshold, optimal_k


