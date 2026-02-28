"""
Qwen Tool Calling Format Prompts and Tool Descriptions
"""

import json
from typing import Tuple, List, Dict

def get_selection_user_prompt() -> str:
    """Get the user prompt for selection-based benchmark questions (V* and HR-Bench)"""
    return """Your role is that of a research assistant specializing in visual information. Answer questions about images by looking at them closely and providing detailed analysis. Please follow this structured thinking process and show your work.

    Start an iterative loop for each question:

    - **First, look closely:** Begin with a detailed description of the image, paying attention to the user's question. List what you can tell just by looking, and what you'll need to look up.
    - **Next, find information:** Use a tool to research the things you need to find out.
    - **Then, review the findings:** Carefully analyze what the tool tells you and decide on your next action.

    Continue this loop until your research is complete.

    To finish, put your final answer within \\boxed{}. Answer with the option's letter from the given choices directly, e.g. \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D} etc. Note that YOU MUST choose One Answer from the Options."""

def get_free_form_user_prompt() -> str:
    """Get the user prompt for selection-based benchmark questions (V* and HR-Bench)"""
    return """Your role is that of a research assistant specializing in visual information. Answer questions about images by looking at them closely. Please follow this structured thinking process and show your work.

    Start an iterative loop for each question:

    - **First, look closely:** Begin with a detailed description of the image, paying attention to the user's question. List what you can tell just by looking, and what you'll need to look up.
    - **Next, find information:** Use a tool to research the things you need to find out.
    - **Then, review the findings:** Carefully analyze what the tool tells you and decide on your next action.

    Continue this loop until your research is complete.

    To finish, You MUST PUT your FINAL ANSWER WITHIN \\boxed{}, and make sure it contains only the answer itself without extra words or symbols."""


def build_system_prompt() -> str:
    """
    Build system prompt with tool descriptions following OpenAI function calling format
    
    Returns:
        System prompt string
    """
    
    # Build functions list in OpenAI format
    functions = [
        {
            "type": "function",
            "function": {
                "name": "image_zoom_in_tool",
                "description": "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                            "description": "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner"
                        },
                        "label": {
                            "type": "string",
                            "description": "The name or label of the object in the specified bounding box"
                        },
                        "img_idx": {
                            "type": "number",
                            "description": "The index of the zoomed-in image (starting from 0)"
                        }
                    },
                    "required": ["bbox_2d", "label", "img_idx"]
                }
            }
        }
    ]
    
    tools_json = json.dumps(functions, ensure_ascii=False)
    
    template_info = f"""

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_json}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""
    
    return template_info


def prepare_initial_messages(question: str, options: List[str], image_path: str,
                             user_prompt: str, parallel_calls: bool = False) -> List[Dict]:
    """Prepare initial messages for question with image using Qwen tool calling format"""
    prompt_text = f"Question: {question}\n"
    if options:
        option_str = "".join([f"{chr(65+i)}. {opt}\n" for i, opt in enumerate(options)])
        prompt_text += f"Options:\n{option_str}\n"
    
    return [
        {"role": "system", "content": [{"text": user_prompt}, {"text": build_system_prompt()}]},
        {"role": "user", "content": [{"image": image_path}, {"text": prompt_text}]}
    ]

