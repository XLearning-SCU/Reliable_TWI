import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from RTWI.voting import reliable_filtering_and_voting, weighted_majority_vote
from RTWI.utils import equal_func
from RTWI.reliability import (
    compute_two_stage_thresholds,
    aggregate_highest_k_entropy,
    compute_trace_entropy_weight,
    extract_stages_from_trace,
    IncrementalTopKMean
)


def load_traces(filepath: str) -> list:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    traces = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not (line := line.strip()): continue
            t = json.loads(line)
            if 'token_prob_vectors' in t:
                t['token_entropies'] = t['token_prob_vectors']
            else:
                t['token_entropies'] = t.get('entropy', [])
            t['extracted_answer'] = t.get('pred_answer', t.get('ans', t.get('trace_answer', '')))
            t['ground_truth'] = t.get('ground_truth', t.get('gt', ''))
            t['turn_boundaries'] = t.get('turn_boundaries', t.get('turns', []))
            traces.append(t)
    return traces


def simulate_token_stop(trace: Dict, thresh: float, k: int) -> Tuple[bool, int]:
    """Token-level early stopping. Returns (passed, tokens_consumed)."""
    entropies = trace.get('token_entropies', [])
    if not entropies:
        return True, 0

    boundaries = trace.get('turn_boundaries', [])
    segments_info = []
    if not boundaries:
        segments_info.append((0, entropies))
    else:
        for start, end in boundaries:
            if start is not None and end is not None:
                segments_info.append((start, entropies[start:end + 1]))
        if not segments_info:
            segments_info.append((0, entropies))

    for start_idx, seg in segments_info:
        if not seg: continue
        tracker = IncrementalTopKMean(k)
        for i, val in enumerate(seg):
            tracker.add(val)
            token_idx = i + 1
            if token_idx >= k and tracker.get_mean() > thresh:
                return False, start_idx + token_idx

    return True, len(entropies)


def evaluate_question_online(
    traces: List[Dict],
    warmup: int = 8,
    budget: int = 32,
    ratio: float = 0.4,
    gamma: float = 0.1,
    consensus: float = 0.9
) -> Dict:
    working = traces[:budget]
    n_warmup = min(warmup, len(working))
    warmup_traces = working[:n_warmup]

    t_think, t_reason, k = compute_two_stage_thresholds(warmup_traces, filtering_ratio=ratio)
    entropy_thresh = -t_think if t_think != 0.0 else -t_reason

    voted_ans, voted_weights = [], []
    used_tokens = 0
    final_pool = []

    def add_to_pool(t):
        ent = t['token_entropies']
        stages = extract_stages_from_trace(t, ent)
        s1 = aggregate_highest_k_entropy(stages.get('thinking_stage', []), k)
        s2 = aggregate_highest_k_entropy(stages.get('reasoning_stage', []), k)
        w = compute_trace_entropy_weight(
            -s1, -s2, t_think, t_reason,
            bool(stages['thinking_stage']), bool(stages['reasoning_stage']), gamma
        )
        final_pool.append(t)
        ans = t.get('extracted_answer')
        if ans:
            voted_ans.append(str(ans))
            voted_weights.append(w)

    for t in warmup_traces:
        used_tokens += len(t['token_entropies'])
        add_to_pool(t)

    for t in working[n_warmup:]:
        if voted_ans:
            winner = weighted_majority_vote(voted_ans, voted_weights)
            if winner and (sum(w for a, w in zip(voted_ans, voted_weights) if a == winner) / sum(voted_weights)) >= consensus:
                break
        passed, stopped_at = simulate_token_stop(t, entropy_thresh, k)
        used_tokens += stopped_at
        if passed: add_to_pool(t)

    result = reliable_filtering_and_voting(final_pool, filtering_ratio=ratio, gamma=gamma, thresholds=(t_think, t_reason), optimal_k=k)
    return {
        'ans': result['answer'],
        'used': used_tokens,
        'potential': sum(len(t['token_entropies']) for t in working),
    }


def run_online_eval(filepath: str, warmup=8, budget=32, ratio=0.4, gamma=0.1, consensus=0.9):
    print(f"\n{'='*70}\n[RTWI Online Eval] {Path(filepath).name}\n{'='*70}")
    print(f"warmup={warmup}  budget={budget}  ratio={ratio}  gamma={gamma}  consensus={consensus}\n")

    traces = load_traces(filepath)
    q_groups = defaultdict(list)
    for t in traces: q_groups[t['qid']].append(t)

    metrics = {'total': 0, 'orig': 0, 'online': 0, 'used': 0, 'pot': 0}

    for qid in sorted(q_groups.keys()):
        group = q_groups[qid]
        gt = group[0]['ground_truth']

        # Baseline: majority vote
        baseline_answers = [t['extracted_answer'] for t in group[:budget]]
        baseline_ans = Counter(baseline_answers).most_common(1)[0][0] if baseline_answers else None
        if equal_func(str(baseline_ans), gt): metrics['orig'] += 1

        # Online simulation
        res = evaluate_question_online(group, warmup, budget, ratio, gamma, consensus)
        if equal_func(str(res['ans']), gt): metrics['online'] += 1

        metrics['total'] += 1
        metrics['used'] += res['used']
        metrics['pot'] += res['potential']


    total = metrics['total']
    if total > 0:
        orig, online = metrics['orig'], metrics['online']
        print(f"{'Metric':<28} | {'Value'}")
        print(f"{'-'*45}")
        print(f"{'Total Questions':<28} | {total}")
        print(f"{'Self-Consistency':<28} | {orig/total:.2%} ({orig}/{total})")
        print(f"{'RTWI (Online)':<28} | {online/total:.2%} ({online}/{total})")

        pot, used = metrics['pot'], metrics['used']
        saved = pot - used
        saving_ratio = 1 - (used / pot) if pot > 0 else 0.0
        print(f"\n{'Token Efficiency':}")
        print(f"  {'Total Tokens':<24} {pot:,}")
        print(f"  {'Used Tokens':<24} {used:,}")
        print(f"  {'Saved Tokens':<24} {saved:,}")
        print(f"  {'Token Saving Ratio':<24} {saving_ratio:.2%}")


    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--budget', type=int, default=32)
    parser.add_argument('--filtering_ratio', type=float, default=0.4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--consensus', type=float, default=0.9)
    args = parser.parse_args()
    run_online_eval(args.filepath, args.warmup, args.budget, args.filtering_ratio, args.gamma, args.consensus)
