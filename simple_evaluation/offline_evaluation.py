import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from RTWI.voting import reliable_filtering_and_voting
from RTWI.utils import equal_func


def load_traces(filepath: str) -> list:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    traces = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not (line := line.strip()): continue
            t = json.loads(line)
            t['token_entropies'] = t.get('entropy', [])
            t['extracted_answer'] = t.get('pred_answer', t.get('ans', t.get('trace_answer', '')))
            t['ground_truth'] = t.get('ground_truth', t.get('gt', ''))
            traces.append(t)
    return traces


def run_evaluation(filepath: str, total_budget: int = 32, filtering_ratio: float = 0.4, gamma: float = 0.1):
    print(f"\n{'='*70}\n[RTWI Offline Eval] {Path(filepath).name}\n{'='*70}")
    print(f"budget={total_budget}  ratio={filtering_ratio}  gamma={gamma}\n")

    try:
        raw_traces = load_traces(filepath)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    questions = defaultdict(list)
    for t in raw_traces:
        questions[t['qid']].append(t)

    metrics = {'total': 0, 'orig': 0, 'rtwi': 0, 'votes': 0, 'tokens': 0}

    for qid in sorted(questions.keys()):
        q_traces = questions[qid][:total_budget]
        gt = q_traces[0].get('ground_truth', '')

        # Baseline: majority vote
        answers = [t['extracted_answer'] for t in q_traces]
        orig_ans = Counter(answers).most_common(1)[0][0] if answers else None

        # RTWI: two-stage filtering + weighted voting
        result = reliable_filtering_and_voting(q_traces, filtering_ratio=filtering_ratio, gamma=gamma)

        metrics['total'] += 1
        if equal_func(str(orig_ans), gt): metrics['orig'] += 1
        if equal_func(str(result['answer']), gt): metrics['rtwi'] += 1
        metrics['votes'] += result['num_votes']
        metrics['tokens'] += sum(len(t['token_entropies']) for t in q_traces)

    total = metrics['total']
    if total > 0:
        orig, rtwi = metrics['orig'], metrics['rtwi']
        print(f"{'Metric':<28} | {'Value'}")
        print(f"{'-'*45}")
        print(f"{'Total Questions':<28} | {total}")
        print(f"{'Self-Consistency':<28} | {orig/total:.2%} ({orig}/{total})")
        print(f"{'RTWI (Offline)':<28} | {rtwi/total:.2%} ({rtwi}/{total})")

        total_tokens = metrics['tokens']
        print(f"\n{'Token Statistics':}")
        print(f"  {'Total Tokens':<24} {total_tokens:,}")
        print(f"  {'Avg Tokens per Sample':<24} {total_tokens/total:.2f}")

        print(f"\n{'Avg Filtered Traces':<28} | {metrics['votes']/total:.1f}")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RTWI offline on trace JSONL files.")
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--total_budget', type=int, default=32)
    parser.add_argument('--filtering_ratio', type=float, default=0.4)
    parser.add_argument('--gamma', type=float, default=0.1)
    
    args = parser.parse_args()
    run_evaluation(args.filepath, args.total_budget, args.filtering_ratio, args.gamma)
