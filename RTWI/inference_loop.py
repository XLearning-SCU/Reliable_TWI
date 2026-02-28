"""
Inference loop for DeepThinkLLM handling multi-turn generation and tool execution.
"""
import copy
import re
import json
import os
import tempfile
import traceback
from typing import List, Optional, Callable, Dict, Any, Tuple
from vllm import SamplingParams
from qwen_vl_utils import process_vision_info
from PIL import Image

def get_image_path(pil_image: Image.Image) -> str:
    """Get or create file path for a PIL image."""
    if hasattr(pil_image, 'filename') and pil_image.filename:
        return pil_image.filename
    else:
        # Save to temp file if no filename
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            pil_image.save(tmp.name)
            return tmp.name


class TraceState:
    """Manages the state of a single trace during inference."""
    def __init__(
        self, 
        trace_id: int, 
        initial_messages: List[Dict], 
        sampling_params: SamplingParams,
        image_paths: List[str]
    ):
        self.trace_id = trace_id
        self.messages = copy.deepcopy(initial_messages)
        self.sampling_params = sampling_params
        
        # History tracking
        self.image_paths = copy.deepcopy(image_paths)
        self.vllm_outputs = []       # Raw vllm outputs
        self.turn_texts = []         # Text response for each turn
        self.tool_bboxes = []        # BBoxes from tool calls
        
        # Status
        self.is_finished = False
        self.error = None

class BatchInferenceLoop:
    """
    Handles the main inference loop for a batch of traces.
    Supports multi-turn generation, tool parsing, and execution.
    """
    def __init__(
        self, 
        llm, 
        processor, 
        tool_executor_func: Callable,
        tool_parser_func: Callable
    ):
        self.llm = llm
        self.processor = processor
        self.tool_executor = tool_executor_func
        self.tool_parser = tool_parser_func

    def run(
        self, 
        trace_states: List[TraceState], 
        max_turns: int = 10,
        reasoning_effort: str = "low",
        dataset_name: Optional[str] = None,
        sample_name: Optional[str] = None,
        run_id: Optional[str] = None,
        batch_start_idx: int = 0,
        save_zoom_images: bool = False
    ) -> List[TraceState]:
        """
        Run the inference loop for the provided trace states.
        """
        
        for turn in range(1, max_turns + 1):
            # 1. Identify active traces
            active_states = [s for s in trace_states if not s.is_finished]
            if not active_states:
                break
                
            if turn > 1:
                print(f"  Turn {turn}: continuing {len(active_states)} active traces...")

            # 2. Prepare inputs for this turn
            prompts, valid_states = self._prepare_inputs(active_states, reasoning_effort)
            if not prompts:
                break
                
            # 3. Generate (Batch Inference)
            # Collect sampling params corresponding to valid states
            sampling_params_list = [s.sampling_params for s in valid_states]
            outputs = self.llm.generate(prompts, sampling_params_list)
            
            # 4. Process outputs & Execute tools
            for i, state in enumerate(valid_states):
                vllm_output = outputs[i]
                self._process_single_output(
                    state, 
                    vllm_output, 
                    dataset_name, 
                    sample_name, 
                    run_id, 
                    turn,
                    save_zoom_images
                )
        
        return trace_states

    def _prepare_inputs(self, active_states: List[TraceState], reasoning_effort: str):
        """Prepare prompts and multi-modal data for vLLM generation."""
        prompts = []
        valid_states = []
        
        for state in active_states:
            if not state.messages:
                continue
                
            msg = state.messages
            
            # Apply chat template
            prompt_text = self.processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=reasoning_effort
            )
            
            # Process vision info
            image_inputs, video_inputs = process_vision_info(msg)
            current_mm_data = {}
            if image_inputs is not None:
                current_mm_data['image'] = image_inputs
            if video_inputs is not None:
                current_mm_data['video'] = video_inputs
            
            prompts.append({
                'prompt': prompt_text,
                'multi_modal_data': current_mm_data
            })
            valid_states.append(state)
            
        return prompts, valid_states

    def _process_single_output(
        self, 
        state: TraceState, 
        vllm_output,
        dataset_name,
        sample_name,
        run_id,
        turn_count,
        save_zoom_images
    ):
        """Processes a single generation output, handles tool calls & updates state."""
        response_text = vllm_output.outputs[0].text
        
        # Update history
        state.vllm_outputs.append(vllm_output)
        state.turn_texts.append(response_text)
        
        # Check for tool calls
        all_tool_calls = self.tool_parser(response_text)
        
        if not all_tool_calls:
            # No tool calls -> End of trace
            state.messages.append({"role": "assistant", "content": response_text})
            state.is_finished = True
            return

        # Handle tool calls
        self._handle_tool_execution(
            state, 
            response_text, 
            all_tool_calls, 
            dataset_name, 
            sample_name, 
            run_id,
            turn_count,
            save_zoom_images
        )

    def _handle_tool_execution(
        self,
        state: TraceState,
        response_text: str,
        all_tool_calls: list,
        dataset_name: str,
        sample_name: str,
        run_id: str,
        turn_count: int,
        save_zoom_images: bool
    ):
        """Parses tool calls from text, executes them, and updates messages."""
        
        # 1. Parse thinking and tool parts for message history
        first_tool_call_pos = response_text.find('<tool_call>')
        thinking_part = response_text[:first_tool_call_pos].strip() if first_tool_call_pos > 0 else ""
        last_tool_call_end = response_text.rfind('</tool_call>')
        tool_calls_part = (
            response_text[first_tool_call_pos:last_tool_call_end + len('</tool_call>')] 
            if last_tool_call_end >= 0 else response_text[first_tool_call_pos:]
        )
        
        content_items = []
        if thinking_part:
            content_items.append({"text": thinking_part})
        content_items.append({"text": ""}) # spacing
        content_items.append({"text": tool_calls_part})
        
        state.messages.append({
            "role": "assistant",
            "content": content_items
        })
        
        # 2. Execute tools
        tool_responses = []
        tool_call_failed = False
        
        for tool_call_idx, (tool_name, tool_args) in enumerate(all_tool_calls):
            # Calculate a unique ID for this tool call (for saving images etc.)
            # We use turn_count * 100 + idx
            call_id_suffix = turn_count * 100 + tool_call_idx
            
            # Call the executor callback with keyword arguments
            result = self.tool_executor(
                tool_name=tool_name, 
                tool_args_json=tool_args, 
                all_image_paths=state.image_paths,
                dataset_name=dataset_name, 
                sample_name=sample_name, 
                turn_count=call_id_suffix, 
                run_id=run_id, 
                trace_idx=state.trace_id,
                save_image=save_zoom_images
            )
            
            # Unpack result: used_tool, tool_result_path, cropped_pil_image, tool_desc, bbox
            used_tool, tool_result_path, _, tool_desc, bbox = result
            
            if used_tool and tool_result_path:
                state.image_paths.append(tool_result_path)
                tool_responses.append((tool_name, tool_result_path, tool_desc))
                if bbox:
                    state.tool_bboxes.append(bbox)
            else:
                print(f"  ⚠️  Tool call failed for trace {state.trace_id}, ending trace.")
                tool_call_failed = True
                break
        
        # 3. Update User Messages with Tool Responses
        if not tool_call_failed and len(tool_responses) == len(all_tool_calls):
            user_content = []
            for _, result_path, _ in tool_responses:
                user_content.append({"text": f"<tool_response>\n"})
                user_content.append({"image": result_path})
                user_content.append({"text": f"\n</tool_response>"})
            
            state.messages.append({
                "role": "user",
                "content": user_content
            })
        else:
            # excessive failure handling -> stop trace
            state.is_finished = True
