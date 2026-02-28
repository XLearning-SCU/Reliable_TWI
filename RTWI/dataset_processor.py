"""
Dataset processing utilities for unified benchmark evaluation
"""

from typing import Dict, List, Tuple, Optional, Any
import os
from .utils import prepare_messages
from .dataload import load_and_extract_sample


def build_deepthink_kwargs(messages, args, img_path: str, actual_subset: str, sample_name: str) -> Dict[str, Any]:
    """
    Build kwargs for deepthink() call based on mode and args.
    
    Args:
        messages: Prepared messages list
        args: Arguments object
        img_path: Path to image file
        actual_subset: Actual subset name
        sample_name: Sample name for saving
        
    Returns:
        Dictionary of kwargs for deepthink()
    """
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_tokens=args.max_tokens, logprobs=10
    )
    
    kwargs = {
        "messages": messages,
        "mode": args.mode,
        "sampling_params": sampling_params,
        "compute_multiple_voting": not args.no_multiple_voting,
        "img_path": img_path,
        "dataset_name": actual_subset,
        "sample_name": sample_name,
        "reasoning_effort": args.reasoning_effort,
        "save_zoom_images": args.save_zoom_images,
        "gamma": args.gamma,
        "dual_stage_filtering_ratio": args.filtering_ratio,
    }
    
    if args.mode == "online":
        kwargs.update({
            "warmup_traces": args.warmup_traces,
            "budget": args.budget
        })
    else:
        kwargs.update({
            "budget": args.budget,
            "batch_size": args.batch_size
        })
    
    return kwargs


def prepare_sample_metadata(qid: int, dataset_type: str, data_source, item, args, actual_subset: str) -> Dict[str, str]:
    """
    Prepare metadata (display_name, sample_name, vstar_subset) for a sample.
    
    Args:
        qid: Question ID
        dataset_type: Type of dataset ('vstar', 'hrbench', etc.)
        data_source: Data source list
        item: Current item from data_source
        args: Arguments object with .subset attribute
        actual_subset: Actual subset path/name
        
    Returns:
        Dictionary with 'display_name', 'sample_name', 'vstar_subset'
    """
    if dataset_type == 'vstar':
        img_file = item[0] if isinstance(item, tuple) else item
        vstar_subset = item[1] if isinstance(item, tuple) else actual_subset
        display_name = f"{img_file} ({vstar_subset})" if isinstance(item, tuple) else img_file
        sample_name = f"{args.subset}/{qid}_{os.path.splitext(img_file)[0]}"
    else:
        vstar_subset = actual_subset
        display_name = f"Question {qid+1}"
        sample_name = f"{args.subset}/{qid}"
    
    return {
        'display_name': display_name,
        'sample_name': sample_name,
        'vstar_subset': vstar_subset
    }


def load_sample(qid: int, total_items: int, dataset_type: str, data_source, item,
                args, actual_subset: str) -> Dict[str, Any]:
    """
    Load and prepare a single sample: metadata, question/options/ground_truth,
    image path, and formatted messages. Does NOT run inference.

    Returns:
        Dict with keys: display_name, sample_name, vstar_subset,
                        question, options, ground_truth, img_path, messages.
    """

    meta = prepare_sample_metadata(qid, dataset_type, data_source, item, args, actual_subset)
    print(f"\n{'='*80}\n{meta['display_name']} ({qid+1}/{total_items})\n{'='*80}")

    question, options, ground_truth, img_path = load_and_extract_sample(
        dataset_type, data_source, qid, args, meta['vstar_subset']
    )
    messages = prepare_messages(question, options, img_path, dataset_type=dataset_type)

    return {
        **meta,
        'question': question,
        'options': options,
        'ground_truth': ground_truth,
        'img_path': img_path,
        'messages': messages,
    }


def cleanup_sample(dataset_type: str, img_path: Optional[str]) -> None:
    """Remove temporary image file for non-vstar datasets."""
    from .dataload import cleanup_temp_image
    if dataset_type != 'vstar' and img_path:
        cleanup_temp_image(img_path)
