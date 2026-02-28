"""
Reliable Thinking with Image - Unified Benchmark Evaluation
"""

import os
import argparse
from RTWI.wrapper import Reliable_TWI
from RTWI.utils import save_trace_details, print_summary, evaluate_voting_results, print_evaluation_report
from RTWI.dataload import (
    get_subset_info, load_and_process_dataset, SUBSET_REGISTRY
)
from RTWI.dataset_processor import load_sample, cleanup_sample, build_deepthink_kwargs
from RTWI.config import get_sampling_params_from_config


def process_dataset(data_source, Reliable_mllm, args, actual_subset, dataset_type):
    all_results = []
    total_items = len(data_source)
    if args.max_questions and args.max_questions > 0:
        total_items = min(total_items, args.max_questions)

    for idx in range(total_items):
        item = data_source[idx] if dataset_type == 'vstar' else idx

        # Load dataset-specific data
        sample = load_sample(idx, total_items, dataset_type, data_source, item, args, actual_subset)

        # Build kwargs and run inference
        dkwargs = build_deepthink_kwargs(sample['messages'], args, sample['img_path'], actual_subset, sample['sample_name'])
        result = Reliable_mllm.reliable_think(**dkwargs)

        # Evaluate
        ground_truth = sample['ground_truth']
        evaluation = (
            evaluate_voting_results(result.voting_results, ground_truth)
            if ground_truth and result.voting_results else None
        )
        if evaluation:
            print_evaluation_report(sample['question'], ground_truth, evaluation, result)

        # Cleanup temp image
        cleanup_sample(dataset_type, sample['img_path'])

        all_results.append({
            **result.to_dict(),
            'question': sample['question'],
            'ground_truth': ground_truth,
            'qid': idx,
            'mode': args.mode,
            'evaluation': evaluation,
            'dataset': dataset_type,
            'subset': args.subset,
            'options': sample['options'],
        })

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Reliable_TWI Offline Mode - Unified Benchmark Evaluation')
    
    # Model and dataset arguments
    parser.add_argument('--model', type=str, default="Qwen3-VL-8B-Thinking", help='Model name (e.g., Qwen3-VL-8B-Thinking)')
    parser.add_argument('--model_dir', type=str, default="./model", help='Model base directory')
    parser.add_argument('--dataset', type=str, default='vstar',
                       choices=['vstar', 'hrbench'],
                       help='Dataset to evaluate')
    parser.add_argument('--subset', type=str, default='Attr',
                       choices=list(SUBSET_REGISTRY.keys()),
                       help='Subset of the dataset to evaluate')
    parser.add_argument('--dataset_path', type=str, default='./dataset/vstar_bench', help='Path to the dataset (root directory or file path)')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['online', 'offline'], default='offline',
                       help='Generation mode: online (with warmup + adaptive sampling) or offline (all traces at once)')
    
    # Trace generation parameters
    parser.add_argument('--budget', type=int, default=32,
                       help='Number of traces to generate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for parallel trace generation')
    parser.add_argument('--warmup_traces', type=int, default=8,
                       help='Number of warmup traces (online mode, default: 8)')
    
    # Sampling parameters
    parser.add_argument('--max_tokens', type=int, default=4096,
                       help='Maximum tokens per generation')
    parser.add_argument('--reasoning_effort', type=str, default=None,
                       help='Reasoning effort for GPT models (if None, auto-select based on model type)')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Sampling temperature (if None, auto-select based on model type)')
    parser.add_argument('--top_p', type=float, default=None,
                       help='Top-p sampling parameter (if None, auto-select based on model type)')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Top-k sampling parameter (if None, auto-select based on model type)')
    
    # Two-stage filtering parameters
    parser.add_argument('--filtering_ratio', type=float, default=0.4,
                       help='Ratio for two-stage filtering')
    parser.add_argument('--gamma', type=float, default=None,
                       help='Temperature parameter for weighted voting (if None, auto-select based on model type)')
    
    # Output and other options
    parser.add_argument('--no_multiple_voting', type=bool, default=False,
                       help='Disable multiple voting analysis')
    parser.add_argument('--max_questions', type=int, default=3,
                       help='Number of questions to process (default: 3 for testing)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.7,
                       help='GPU memory utilization ratio (0.0-1.0)')
    parser.add_argument('--max_model_len', type=int, default=51200,
                       help='Maximum model length for vLLM')
    parser.add_argument('--save_zoom_images', type=bool, default=False, help='Save zoom-in images')
    
    args = parser.parse_args()
    
    # Auto-configure sampling parameters from model config
    config = get_sampling_params_from_config(args.model)
    for key in ['temperature', 'top_p', 'top_k', 'gamma', 'reasoning_effort']:
        if getattr(args, key) is None:
            setattr(args, key, config.get(key))
    
    args.model_path = os.path.join(args.model_dir, args.model)
    actual_subset_path, dataset_type, category_filter = get_subset_info(args.subset)
    
    print("\n" + "="*80)
    print(f"Reliable_TWI Unified Evaluation\nModel: {args.model}\nDataset: {args.dataset.upper()} ({args.subset})\nMode: {args.mode.upper()} (Budget: {args.budget})")
    print("="*80)
    
    # Load dataset using dataset_type and actual_subset_path
    data_source, data_path = load_and_process_dataset(
        dataset_type, args, actual_subset_path, category_filter
    )

    # Initialize Reliable_TWI
    Reliable_mllm = Reliable_TWI(
        model=args.model_path,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    # Process dataset
    all_results = process_dataset(
        data_source, Reliable_mllm, args, actual_subset_path, dataset_type
    )
    
    # Print summary
    if all_results:
        print_summary(all_results, args)
        save_trace_details(all_results, args)


if __name__ == "__main__":
    main()