"""
Data loading utilities for V* benchmark and HR-Bench datasets
"""

import os
import json
import re
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import tempfile
from typing import List, Tuple, Dict, Optional, Union


def load_vstar_dataset(vstar_path: str, subset: str, max_questions: Optional[int] = None) -> Tuple[List[str], str]:
    """
    Load V* benchmark dataset
    
    Args:
        vstar_path: Path to V* benchmark dataset root
        subset: Dataset subset ('direct_attributes' or 'relative_position')
        max_questions: Maximum number of questions to load (None = all)
        
    Returns:
        tuple: (image_files list, full dataset path)
    """
    test_path = os.path.join(vstar_path, subset)
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"V* dataset path {test_path} does not exist")
    
    # Get all image files (excluding json files)
    image_files = [f for f in os.listdir(test_path) if not f.endswith('.json')]
    image_files.sort()  # Sort for consistent ordering
    
    if max_questions:
        image_files = image_files[:max_questions]
    
    return image_files, test_path


def load_hrbench_dataset(
    hrbench_path: str,
    subset: str,
    category_filter: Optional[str] = None,
    max_questions: Optional[int] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Load HR-Bench dataset
    
    Args:
        hrbench_path: Path to HR-Bench dataset root
        subset: Dataset subset name (e.g., 'hr_bench_4k')
        category_filter: Optional category filter ('single' or 'cross')
        max_questions: Maximum number of questions to load (None = all)
        
    Returns:
        tuple: (DataFrame, full data path)
    """

    data_path = os.path.join(hrbench_path, subset + '.tsv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"HR-Bench dataset file {data_path} does not exist")
    
    df = pd.read_csv(data_path, sep='\t')
    
    # Sort by index to ensure consistent ordering across runs
    if 'index' in df.columns:
        df = df.sort_values('index').reset_index(drop=True)
    
    # Filter by category if specified
    if category_filter:
        original_len = len(df)
        df = df[df['category'].str.strip('"') == category_filter]
        filtered_len = len(df)
        print(f"Filtering by category='{category_filter}': {original_len} -> {filtered_len} samples")
    
    if max_questions:
        df = df.iloc[:max_questions]
    
    return df, data_path


def load_vstar_question(vstar_path: str, subset: str, img_file: str) -> Dict:
    """
    Load a single V* benchmark question with annotation
    
    Args:
        vstar_path: Path to V* dataset root
        subset: Dataset subset
        img_file: Image filename
        
    Returns:
        dict: Contains 'question', 'options', 'img_path', 'ground_truth'
    """
    # 🔥 Use absolute paths to avoid path resolution issues
    img_path = os.path.abspath(os.path.join(vstar_path, subset, img_file))
    anno_path = os.path.abspath(os.path.join(vstar_path, subset, img_file.replace('.jpg', '.json')))
    
    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"Annotation file {anno_path} does not exist")
    
    with open(anno_path, 'r') as f:
        anno = json.load(f)
    
    return {
        'question': anno['question'],
        'options': anno['options'],
        'img_path': img_path,
        'ground_truth': 'A'  # V* always has answer as 'A'
    }


def load_hrbench_question(df: pd.DataFrame, qid: int) -> Dict:
    """
    Load a single HR-Bench question with image
    
    Args:
        df: DataFrame containing HR-Bench data
        qid: Question index
        
    Returns:
        dict: Contains 'question', 'options', 'img_path', 'ground_truth'
              Note: img_path is a temporary file that should be cleaned up
    """
    row = df.iloc[qid]
    
    question = row['question']
    options = [
        f"{row['A']}",
        f"{row['B']}",
        f"{row['C']}",
        f"{row['D']}"
    ]
    ground_truth = row['answer']
    
    # Decode and save image to temporary file
    image_data = base64.b64decode(row['image'])
    image = Image.open(BytesIO(image_data))
    tmp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    image.save(tmp_file.name)
    img_path = tmp_file.name
    tmp_file.close()
    
    return {
        'question': question,
        'options': options,
        'img_path': img_path,
        'ground_truth': ground_truth
    }


def cleanup_temp_image(img_path: str) -> None:
    """
    Clean up temporary image file
    
    Args:
        img_path: Path to temporary image file
    """
    if img_path and os.path.exists(img_path):
        try:
            os.remove(img_path)
        except Exception as e:
            print(f"Warning: Failed to delete temp file {img_path}: {e}")


# Registry of supported subsets and their metadata
# Key: subset_name -> (actual_path, dataset_type, category_filter)
SUBSET_REGISTRY = {
    # V* Benchmark
    'Attr':            ('direct_attributes', 'vstar',     None),
    'Spatial':         ('relative_position', 'vstar',     None),
    
    # HR-Bench
    'HR4K-FSP':          ('hr_bench_4k',       'hrbench',   'single'),
    'HR4K-FCP':          ('hr_bench_4k',       'hrbench',   'cross'),
    'HR8K-FSP':          ('hr_bench_8k',       'hrbench',   'single'),
    'HR8K-FCP':          ('hr_bench_8k',       'hrbench',   'cross'),
}


def get_display_name(subset: str) -> str:
    """Get descriptive display name for file saving."""
    return subset


def get_subset_info(subset: str) -> Tuple[str, str, Optional[str]]:
    """Map subset name to dataset type and loading parameters."""
    if subset not in SUBSET_REGISTRY:
        raise ValueError(f"Unknown subset: {subset}. Choices: {list(SUBSET_REGISTRY.keys())}")
    return SUBSET_REGISTRY[subset]


def load_and_process_dataset(dataset_type: str, args, actual_subset: str, category_filter: Optional[str]) -> Tuple[any, str]:
    """
    Load dataset and print loading information
    
    Args:
        dataset_type: 'vstar' or 'hrbench'
        args: Command line arguments
        actual_subset: Actual subset name
        category_filter: Category filter for HR-Bench
        
    Returns:
        tuple: (data_source, data_path) where data_source is either image_files list or DataFrame or list of tuples
    """
    dataset_path = args.dataset_path
    
    if dataset_type == 'vstar':
        try:
            if actual_subset == 'all':
                # Load both Attr and Spatial, keep them separate with source info
                attr_files, attr_path = load_vstar_dataset(
                    dataset_path, 'direct_attributes', args.max_questions
                )
                spatial_files, spatial_path = load_vstar_dataset(
                    dataset_path, 'relative_position', args.max_questions
                )
                # Store tuples of (filename, subset_name) for proper loading
                data_source = [(f, 'direct_attributes') for f in attr_files] + [(f, 'relative_position') for f in spatial_files]
                data_path = f"{dataset_path}/all"
                print(f"Loaded V* Attr subset: {len(attr_files)} images")
                print(f"Loaded V* Spatial subset: {len(spatial_files)} images")
                # Store vstar_path and actual_subset in args for later use
                args.vstar_combined = True
            else:
                # actual_subset is now the actual directory name (e.g., 'direct_attributes')
                vstar_files, data_path = load_vstar_dataset(
                    dataset_path, actual_subset, args.max_questions
                )
                data_source = vstar_files
                args.vstar_combined = False
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
    elif dataset_type == 'hrbench':
        try:
            data_source, data_path = load_hrbench_dataset(
                dataset_path, actual_subset, category_filter, args.max_questions
            )
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    print(f"Data path: {data_path}")
    print(f"Total questions: {len(data_source)}")
    print("="*80 + "\n")
    
    return data_source, data_path


def load_and_extract_sample(dataset_type: str, data_source, qid: int, args, actual_subset: str) -> Tuple[str, List[str], str, str]:
    """
    Load and extract sample data for a single question
    
    Args:
        dataset_type: 'vstar' or 'hrbench'
        data_source: image_files list (vstar) or DataFrame (hrbench)
        qid: Question ID
        args: Command line arguments
        actual_subset: Actual subset name
        
    Returns:
        tuple: (question, options, ground_truth, img_path)
    """
    dataset_path = args.dataset_path
    
    if dataset_type == 'vstar':
        # Check if this is a combined V* dataset (Vstar-All)
        if hasattr(args, 'vstar_combined') and args.vstar_combined:
            img_file, subset_name = data_source[qid]
        else:
            img_file = data_source[qid]
            subset_name = actual_subset
        sample_data = load_vstar_question(dataset_path, subset_name, img_file)
    
    elif dataset_type == 'hrbench':
        sample_data = load_hrbench_question(data_source, qid)
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return (
        sample_data['question'],
        sample_data['options'],
        sample_data['ground_truth'],
        sample_data['img_path']
    )

