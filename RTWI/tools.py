"""
Tool execution and management module.
"""

import os
import re
import math
import json
import tempfile
import traceback
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image

# Qwen function calling format markers
FN_NAME = '✿FUNCTION✿'
FN_ARGS = '✿ARGS✿'
FN_RESULT = '✿RESULT✿'
FN_EXIT = '✿RETURN✿'
FN_STOP = '✿STOP✿'  # sometimes used

# ============= IMAGE PROCESSING UTILS =============

def round_by_factor(number: int, factor: int) -> int:
    """Round number to nearest multiple of factor."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Ceil number to nearest multiple of factor."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Floor number to nearest multiple of factor."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int = 32, 
                 min_pixels: int = 256 * 32 * 32, 
                 max_pixels: int = 12845056) -> Tuple[int, int]:
    """
    Calculate new dimensions preserving aspect ratio within pixel constraints.
    Ensures dimensions are multiples of 'factor'.
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    
    return h_bar, w_bar


def crop_and_resize_image(
    pil_image: Image.Image,
    bbox: Tuple[int, int, int, int],
    dataset_name: Optional[str] = None,
    sample_name: Optional[str] = None,
    turn_count: int = 0,
    model_name: Optional[str] = None,
    run_id: Optional[str] = None,
    trace_idx: Optional[int] = None,
    save_image: bool = False
) -> Tuple[Image.Image, str]:
    """
    Crop image by bbox and resize to appropriate dimensions.
    """
    img_width, img_height = pil_image.size
    
    # ===== Step 1: Convert relative coordinates [0-1000] to absolute pixels =====
    rel_x1, rel_y1, rel_x2, rel_y2 = bbox
    left = max(0, rel_x1 / 1000.0 * img_width)
    top = max(0, rel_y1 / 1000.0 * img_height)
    right = min(img_width, rel_x2 / 1000.0 * img_width)
    bottom = min(img_height, rel_y2 / 1000.0 * img_height)
    
    # ===== Step 1.5: Validate and fix bbox coordinates =====
    if left >= right:
        left, right = min(left, right), max(left, right)
        if left == right:
            center_x = left
            left = max(0, center_x - 16)
            right = min(img_width, center_x + 16)
    
    if top >= bottom:
        top, bottom = min(top, bottom), max(top, bottom)
        if top == bottom:
            center_y = top
            top = max(0, center_y - 16)
            bottom = min(img_height, center_y + 16)
    
    # ===== Step 2: Expand bbox if too small (< 32x32) =====
    height = bottom - top
    width = right - left
    
    if height < 32 or width < 32:
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        aspect_ratio = max(height, width) / max(min(height, width), 1e-6)
        
        # Calculate expansion size
        if aspect_ratio > 10:
            if height > width:
                target_h, target_w = max(32, height), max(32, height / 10)
            else:
                target_h, target_w = max(32, width / 10), max(32, width)
            half_h, half_w = math.ceil(target_h * 0.5), math.ceil(target_w * 0.5)
        else:
            ratio = 32 / min(height, width)
            half_h = math.ceil(height * ratio * 0.5)
            half_w = math.ceil(width * ratio * 0.5)
        
        new_left = max(0, math.floor(center_x - half_w))
        new_right = min(img_width, math.ceil(center_x + half_w))
        new_top = max(0, math.floor(center_y - half_h))
        new_bottom = min(img_height, math.ceil(center_y + half_h))
        
        new_h = new_bottom - new_top
        new_w = new_right - new_left
        new_aspect = max(new_h, new_w) / max(min(new_h, new_w), 1e-6)
        
        if new_h >= 32 and new_w >= 32 and new_aspect < 150:
            left, top, right, bottom = new_left, new_top, new_right, new_bottom
    
    # ===== Step 3: Crop and resize image =====
    cropped_image = pil_image.crop((left, top, right, bottom))
    new_w, new_h = smart_resize(right - left, bottom - top, factor=32, 
                                min_pixels=256 * 32 * 32)
    cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
    
    # ===== Step 4: Determine save path =====
    if save_image and dataset_name and sample_name:
        # Permanent save with meaningful directory structure
        base_paths = [os.getcwd(), "zoom_in_img"]
        if model_name:
            base_paths.extend([model_name, dataset_name])
        else:
            base_paths.append(dataset_name)
        base_paths.append(sample_name)
        
        zoom_folder = os.path.join(*base_paths)
        os.makedirs(zoom_folder, exist_ok=True)
        
        filename = f"trace{trace_idx}_turn{turn_count}.png" if trace_idx is not None \
                   else f"{turn_count}.png"
        cropped_path = os.path.join(zoom_folder, filename)
    else:
        # Temporary save
        suffix = f"_trace{trace_idx}_turn{turn_count}.png" if trace_idx is not None \
                 else f"_turn{turn_count}.png"
        cropped_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
    
    # ===== Step 5: Save image =====
    try:
        cropped_image.save(cropped_path)
    except Exception as e:
        print(f"[ERROR] Failed to save image: {e}")
        raise
    
    return cropped_image, cropped_path


def cleanup_zoom_image(image_path: str):
    """Delete a zoom-in image file."""
    if image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
        except Exception:
            pass


# ============= TOOL PARSING =============

def has_tool_call(text: str) -> bool:
    """Check if text contains a tool call marker."""
    return FN_NAME in text or '[TOOL_CALL]' in text or 'function_call' in text or '<tool_call>' in text


def detect_tool_call(text: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Detect if text contains a tool call and extract details.
    Supports ✿FUNCTION✿, [TOOL_CALL], and <tool_call> formats.
    """
    if not has_tool_call(text):
        return False, None, None
    
    # 1. Try ✿FUNCTION✿ format
    fn_pattern = f'{FN_NAME}:\s*(\w+)'
    fn_match = re.search(fn_pattern, text)
    
    if fn_match:
        tool_name = fn_match.group(1).strip()
        args_pattern = f'{FN_ARGS}:\s*(.+?)(?={FN_EXIT}|{FN_STOP}|$)'
        args_match = re.search(args_pattern, text, re.DOTALL)
        
        if args_match:
            args_str = args_match.group(1).strip()
            try:
                tool_args = json.loads(args_str)
            except:
                tool_args = {"raw_args": args_str}
        else:
            tool_args = {}
        return True, tool_name, tool_args
    
    # 2. Try [TOOL_CALL] format
    tool_call_pattern = r'\[TOOL_CALL\]\s*(\w+)\s*\n\s*(\{[^}]+\})'
    tool_match = re.search(tool_call_pattern, text, re.DOTALL)
    
    if tool_match:
        tool_name = tool_match.group(1).strip()
        args_str = tool_match.group(2).strip()
        try:
            tool_args = json.loads(args_str)
        except:
            tool_args = {"raw_args": args_str}
        return True, tool_name, tool_args
        
    # 3. Try <tool_call> format
    # Handled separately usually due to multi-call support, but supporting simple case here
    start_tag = '<tool_call>'
    end_tag = '</tool_call>'
    s_idx = text.find(start_tag)
    e_idx = text.find(end_tag)
    
    if s_idx != -1 and e_idx != -1:
        json_str = text[s_idx + len(start_tag):e_idx].strip()
        try:
            data = json.loads(json_str)
            return True, data.get('name'), data.get('arguments', {})
        except:
            pass
            
    return False, None, None


def parse_tool_calls_from_response(response_text: str) -> List[Tuple[str, str]]:
    """
    Parse ALL tool calls from model response text.
    Returns list of (tool_name, tool_args_json_str) tuples.
    Primarily targeting <tool_call> JSON format.
    """
    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.finditer(tool_call_pattern, response_text, re.DOTALL)
    
    tool_calls = []
    for match in matches:
        try:
            tool_call_json = match.group(1)
            tool_call_dict = json.loads(tool_call_json)
            tool_name = tool_call_dict.get('name')
            tool_args = tool_call_dict.get('arguments', {})
            tool_args_str = json.dumps(tool_args, ensure_ascii=False)
            
            if tool_name:
                tool_calls.append((tool_name, tool_args_str))
        except json.JSONDecodeError:
            continue
    
    # If no <tool_call> found, try legacy single call detection
    if not tool_calls:
        has_call, name, args = detect_tool_call(response_text)
        if has_call and name:
            tool_calls.append((name, json.dumps(args, ensure_ascii=False)))
            
    return tool_calls


def format_tool_result_message(tool_name: str, result: str) -> str:
    """Format tool result in Qwen agent format."""
    return f'{FN_RESULT} {tool_name}\n{result}\n{FN_EXIT}'


# ============= MAIN EXECUTION HANDLER =============

def execute_tool_call(
    tool_name: str,
    tool_args_json: str,
    all_image_paths: list,
    model_short_name: str,
    dataset_name: Optional[str] = None,
    sample_name: Optional[str] = None,
    turn_count: int = 0,
    run_id: Optional[str] = None,
    trace_idx: Optional[int] = None,
    save_image: bool = False
) -> Tuple[bool, Optional[str], Optional[Image.Image], str, Optional[list]]:
    """
    Execute a single tool call.
    Currently supports 'image_zoom_in_tool'.
    """
    if tool_name != "image_zoom_in_tool" or not all_image_paths:
        return False, None, None, "", None
    
    try:
        # Parse tool arguments
        tool_args = json.loads(tool_args_json)
        bbox = tool_args.get("bbox_2d")
        label = tool_args.get("label", "region")
        img_idx = tool_args.get("img_idx", 0)  # Extract img_idx parameter
        
        if not bbox or len(bbox) != 4:
            print(f"Invalid bbox in tool args: {bbox}")
            return False, None, None, "", None
        
        # Select image based on img_idx
        if img_idx < 0 or img_idx >= len(all_image_paths):
            print(f"Invalid img_idx {img_idx}, valid range: 0-{len(all_image_paths)-1}")
            return False, None, None, "", None
        
        selected_image_path = all_image_paths[img_idx]
        
        # Load the selected image
        if selected_image_path.startswith('file://'):
            selected_image_path = selected_image_path[len('file://'):]
        
        if not os.path.exists(selected_image_path):
            print(f"Image file not found: {selected_image_path}")
            return False, None, None, "", None
        
        source_pil_image = Image.open(selected_image_path)
        
        # Execute crop and resize
        cropped_image, cropped_path = crop_and_resize_image(
            source_pil_image, tuple(bbox), dataset_name, sample_name, turn_count, 
            model_short_name, run_id, trace_idx, save_image
        )
        
        # Save original image if this is the first zoom action and save_image is enabled
        if save_image and dataset_name and sample_name and all_image_paths:
            try:
                base_paths = [os.getcwd(), "zoom_in_img"]
                if model_short_name:
                    base_paths.extend([model_short_name, dataset_name])
                else:
                    base_paths.append(dataset_name)
                base_paths.append(sample_name)
                
                zoom_folder = os.path.join(*base_paths)
                os.makedirs(zoom_folder, exist_ok=True)
                
                ori_path = os.path.join(zoom_folder, "ori.png")
                
                if not os.path.exists(ori_path):
                    original_img_path = all_image_paths[0]
                    if original_img_path.startswith('file://'):
                        original_img_path = original_img_path[len('file://'):]
                    
                    if os.path.exists(original_img_path):
                        Image.open(original_img_path).save(ori_path)
            except Exception as e:
                print(f"Warning: Failed to save original image: {e}")
        
        # Return result
        result_text = f"Zoomed in on: {label} (from image {img_idx})"
        return True, cropped_path, cropped_image, result_text, bbox
        
    except Exception as e:
        print(f"Error executing tool {tool_name}: {e}")
        traceback.print_exc()
        return False, None, None, "", None
