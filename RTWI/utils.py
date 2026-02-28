"""
Utility functions for DeepThinkLLM
"""

import os
import json
import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple
from dynasor.core.evaluator import math_equal
from .prompts import prepare_initial_messages, get_selection_user_prompt, get_free_form_user_prompt
from .voting import evaluate_voting_results as _evaluate_voting_results
from .reliability import calculate_token_entropies, extract_stages_from_trace, aggregate_highest_k_entropy, compute_trace_entropy_weight

def prepare_messages(question: str, options: List[str], image_input, dataset_type: str = 'default') -> List[Dict]:
    """Prepare messages for benchmark question with image."""
    
    is_free = dataset_type in ('mathverse', 'logicvista', 'mathvision', 'visualprobe')
    user_prompt = get_free_form_user_prompt() if is_free else get_selection_user_prompt()
    opts = [] if is_free else options
    
    return prepare_initial_messages(question, opts, image_input, user_prompt)


def extract_answer(text: str) -> Optional[str]:
    """Extract boxed answer from text"""
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        return a.strip()
    
    return None

def extract_token_trace(token_ids: List[int], logprobs: List[Dict]) -> List[Dict[str, Any]]:
    """Extract token probabilities and decoded characters."""
    token_trace = []
    if token_ids and logprobs and len(token_ids) == len(logprobs):
        for tid, lp_dict in zip(token_ids, logprobs):
            if lp_dict is None:
                 token_trace.append({"id": tid, "prob": 0.0, "decode": ""})
                 continue
            
            val = lp_dict.get(tid)
            prob = 0.0
            decode = ""
            
            if val is not None:
                if hasattr(val, 'logprob'):
                    log_val = val.logprob
                    decode_val = getattr(val, 'decoded_token', None)
                elif isinstance(val, dict):
                    log_val = val.get('logprob', -float('inf'))
                    decode_val = val.get('decoded_token')
                else:
                     log_val = val
                     decode_val = None

                try:
                    prob = math.exp(log_val)
                except OverflowError:
                    prob = 0.0
                
                if decode_val:
                    decode = decode_val
            
            token_trace.append({
                "id": tid,
                "prob": prob,
                "decode": decode
            })
    return token_trace

def equal_func(answer: str, ground_truth: str) -> bool:
    """
    Check if answer equals ground truth using math_equal when available
    
    Args:
        answer: Model's answer
        ground_truth: Ground truth answer
    """
    if not answer or not ground_truth:
        return False
    
    answer = str(answer).strip()
    ground_truth = str(ground_truth).strip()
    
    # First try exact string match (case-insensitive for single letters)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.upper() == ground_truth.upper()
    
    # Use math_equal if available, otherwise fall back to string comparison
    if math_equal is not None:
        try:
            return math_equal(answer, ground_truth)
        except Exception as e:
            return answer == ground_truth
    else:
        return answer == ground_truth

def evaluate_voting_results(voting_results, ground_truth):
    """Evaluate voting results against ground truth (Wrapper for backward compat)"""
    return _evaluate_voting_results(voting_results, ground_truth, equal_func)


def print_evaluation_report(question, ground_truth, evaluation, result, eval_type="selection"):
    """Print detailed evaluation report"""
    print(f"\n--- Evaluation ({result.total_traces_count} traces, {result.total_tokens} tokens) ---")
    print(f"GT: {ground_truth}")
    
    # # Count individual trace accuracy
    all_traces = result.all_traces
    total = len(all_traces)
    correct = sum(1 for t in all_traces if t.get('extracted_answer') and equal_func(t['extracted_answer'], ground_truth)
    )
    if total > 0:
        print(f"Trace Acc: {correct}/{total} ({correct/total:.1%})")
    
    print(f"{'Method':<25} {'Answer':<20} {'Correct':<8} {'Votes':<6}")
    print("-" * 65)
    
    for method, ev in evaluation.items():
        ans_str = str(ev['answer'])
        ans = (ans_str[:17] + '..') if len(ans_str) > 19 else ans_str
        print(f"{method:<25} {ans:<20} {'✓' if ev['is_correct'] else '✗':<8} {ev['num_votes']:<6}")


# ============= OUTPUT PROCESSING =============

def process_output(output) -> Dict[str, Any]:
    """Process a single vLLM output - for online mode with entropy-based reliability"""
    text = output.text
    token_ids = output.token_ids
    logprobs = output.logprobs
    
    # Calculate entropy
    token_entropies = calculate_token_entropies(logprobs)
    token_trace = extract_token_trace(token_ids, logprobs)

    extracted_answer = extract_answer(text)
    
    # If generation was truncated due to length, invalidate the answer
    if output.finish_reason == "length":
        extracted_answer = None

    turn_boundaries = getattr(output, 'turn_boundaries', [])
    
    result = {
        "stop_reason": output.finish_reason,
        "text": text,
        "token_ids": token_ids,
        "token_trace": token_trace,
        "num_tokens": len(token_ids) if token_ids else 0,
        "token_entropies": token_entropies,
        "extracted_answer": extracted_answer,
        "turn_boundaries": turn_boundaries,
    }
    
    # Compute stage-level information for online filtering
    stages = extract_stages_from_trace(result, token_entropies)
    result['has_stage1'] = len(stages['thinking_stage']) > 0
    result['has_stage2'] = len(stages['reasoning_stage']) > 0
    result['stage1_entropies'] = stages['thinking_stage']
    result['stage2_entropies'] = stages['reasoning_stage']
    
    return result


def process_batch_results(batch_outputs) -> Dict[str, Any]:
    """Process batch results from vLLM for a single question"""
    question_outputs = []
    for output_list in batch_outputs:
        question_outputs += output_list.outputs
    
    # Process all traces for this question
    traces = []
    total_tokens = 0
    
    for output in question_outputs:
        trace_data = process_output(output)
        traces.append(trace_data)
        total_tokens += trace_data["num_tokens"]
    
    return {
        'traces': traces,
        'total_tokens': total_tokens,
        'num_traces': len(traces)
    }


def process_output_offline(output, dataset_name: str) -> Dict[str, Any]:
    """Process a single MultiTurnOutput for offline mode."""
    token_ids = output.token_ids
    logprobs = output.logprobs
    texts = output.texts  # List of texts
    turn_boundaries = output.turn_boundaries
    tool_bboxes = getattr(output, 'tool_bboxes', [])
    
    # Calculate entropy from logprobs
    token_entropies = calculate_token_entropies(logprobs)
    token_trace = extract_token_trace(token_ids, logprobs)
    
    # Extract answer only from the last turn's text
    last_turn_text = texts[-1] if texts else ""
    extracted_answer = extract_answer(last_turn_text)
    
    # If generation was truncated due to length, invalidate the answer
    if output.finish_reason == "length":
        extracted_answer = None

    return {
        "stop_reason": output.finish_reason, 
        "texts": texts,  
        "token_ids": token_ids,
        "token_trace": token_trace,
        "num_tokens": len(token_ids) if token_ids else 0,
        "token_entropies": token_entropies,
        "extracted_answer": extracted_answer,
        "turn_boundaries": turn_boundaries,
        "num_turns": len(texts),
        "tool_bboxes": tool_bboxes,
    }
    

def process_batch_results_offline(batch_outputs, dataset_name: str) -> Dict[str, Any]:
    """Process batch results from vLLM for offline mode."""
    question_outputs = []
    
    # Convert batch_outputs to flat list of outputs
    for output_item in batch_outputs:
        question_outputs.append(output_item)

    # Process all traces for this question
    traces = []
    total_tokens = 0
    
    for output in question_outputs:
        trace_data = process_output_offline(output, dataset_name)
        traces.append(trace_data)
        total_tokens += trace_data["num_tokens"]
    
    return {
        'traces': traces,
        'total_tokens': total_tokens,
        'num_traces': len(traces)
    }

def compute_trace_weight(
    trace: Dict[str, Any], 
    optimal_k: int, 
    thinking_thresh: float, 
    reason_thresh: float, 
    gamma: float
) -> float:
    """
    Compute entropy-based weight for a single trace.
    """
    
    token_entropies = trace.get('token_entropies', [])
    if not token_entropies:
        trace['stop_reason'] = 'no_entropy'
        return 0.0
    
    stages = extract_stages_from_trace(trace, token_entropies)
    s1_entropy = aggregate_highest_k_entropy(stages.get('thinking_stage', []), optimal_k)
    s2_entropy = aggregate_highest_k_entropy(stages.get('reasoning_stage', []), optimal_k)
    has_s1 = len(stages.get('thinking_stage', [])) > 0
    has_s2 = len(stages.get('reasoning_stage', [])) > 0
    
    return compute_trace_entropy_weight(
        -s1_entropy, -s2_entropy, thinking_thresh, reason_thresh, has_s1, has_s2, gamma
    )


def print_summary(all_results, args):
    """Print evaluation summary and append a JSON record to results/."""
    valid = [r for r in all_results if 'error' not in r]
    if not valid:
        print("No valid results to summarize.")
        return

    method_stats = {}
    for r in valid:
        if ev := r.get('evaluation'):
            for m, res in ev.items():
                s = method_stats.setdefault(m, {'c': 0, 't': 0})
                s['t'] += 1
                if res.get('is_correct'):
                    s['c'] += 1

    print(f"\n{'='*60}\nEVALUATION SUMMARY: {args.dataset.upper()} ({args.subset})\n{'='*60}")
    print(f"Total: {len(all_results)} | Valid: {len(valid)}")
    print(f"{'Method':<25} {'Accuracy':<15}")
    print("-" * 45)
    for m, s in sorted(method_stats.items()):
        acc = s['c'] / s['t']
        print(f"{m:<25} {s['c']}/{s['t']} ({acc:.1%})")
    print("=" * 60 + "\n")

    # Collect token and trace statistics
    total_tokens = 0
    total_traces = 0
    num_samples = 0
    for r in valid:
        token_stats = r.get('token_stats', {})
        if token_stats:
            total_tokens += token_stats.get('total_tokens', 0)
            num_samples += 1
        traces = r.get('all_traces', [])
        if traces:
            total_traces += len(traces)

    if num_samples > 0 and total_tokens > 0:
        print(f"Token Statistics:")
        print(f"  Total Tokens: {total_tokens}")
        print(f"  Avg Tokens per Sample: {total_tokens / num_samples:.2f}")
        print("=" * 60 + "\n")
    if num_samples > 0 and total_traces > 0:
        print(f"Trace Statistics:")
        print(f"  Total Traces Generated: {total_traces}")
        print(f"  Avg Traces per Sample: {total_traces / num_samples:.2f}")
        print("=" * 60 + "\n")

    # Build result record
    # online mode: only save Reliable_TWI; offline: save all methods
    if args.mode == 'online':
        stats_to_save = {
            m: {'correct': s['c'], 'total': s['t'], 'acc': round(s['c'] / s['t'], 2)}
            for m, s in method_stats.items() if m == 'Reliable_TWI'
        }
    else:
        stats_to_save = {
            m: {'correct': s['c'], 'total': s['t'], 'acc': round(s['c'] / s['t'], 2)}
            for m, s in method_stats.items()
        }

    import datetime
    summary_data: Dict[str, Any] = {
        'dataset': args.dataset,
        'subset': args.subset,
        'model': args.model,
        'mode': args.mode,
        'timestamp': datetime.datetime.now().isoformat(),
        'stats': stats_to_save,
    }

    # Merge token + trace stats into one trace_stats block
    if num_samples > 0 and (total_tokens > 0 or total_traces > 0):
        trace_stats: Dict[str, Any] = {'num_samples': num_samples}
        if total_tokens > 0:
            trace_stats['total_tokens'] = total_tokens
            trace_stats['avg_tokens_per_sample'] = round(total_tokens / num_samples, 2)
        if total_traces > 0:
            trace_stats['total_traces'] = total_traces
            trace_stats['avg_traces_per_sample'] = round(total_traces / num_samples, 2)
        summary_data['trace_stats'] = trace_stats

    results_dir = os.path.join("results", args.model, args.subset)
    os.makedirs(results_dir, exist_ok=True)
    summary_file = os.path.join(results_dir, f"{args.dataset}_{args.mode}_summary.jsonl")
    with open(summary_file, 'a') as f:
        json.dump(summary_data, f, ensure_ascii=False)
        f.write('\n')
    print(f"✓ Summary appended: {summary_file}")


def save_trace_details(all_results, args=None):
    """Save trace information to JSONL file"""
    if not args: return
    
    results_dir = os.path.join("results", args.model, args.subset)
    os.makedirs(results_dir, exist_ok=True)
    
    base_name = f"{args.dataset}_{args.subset}_{args.mode}_{args.budget}"
    output_file = os.path.join(results_dir, f"{base_name}_0.jsonl")
    
    idx = 0
    while os.path.exists(output_file):
        idx += 1
        output_file = os.path.join(results_dir, f"{base_name}_{idx}.jsonl")
    
    with open(output_file, 'w') as f:
        for res in all_results:
            if 'error' in res: continue
            for i, t in enumerate(res.get('all_traces', [])):
                f.write(json.dumps({
                    'qid': res.get('qid'), 'question': res.get('question'), 'ground_truth': res.get('ground_truth'),
                    'idx': i, 'pred_answer': t.get('extracted_answer'), 'turn_boundaries': t.get('turn_boundaries'),
                    'entropy': [round(e, 6) for e in t.get('token_entropies', [])]
                }, ensure_ascii=False) + '\n')
    print(f"✓ Traces saved: {output_file}")





