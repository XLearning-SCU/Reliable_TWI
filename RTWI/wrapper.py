"""
Reliable_TWI implementation with online and offline mode support
"""

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import time
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import os
import copy
import json
from .outputs import DeepThinkOutput
from .voting import weighted_majority_vote, compute_all_voting_results, compute_all_voting_results_online
from .utils import (
    process_batch_results, 
    process_batch_results_offline,
    compute_trace_weight
)
from .reliability import (
    compute_two_stage_thresholds, aggregate_highest_k_entropy, 
    compute_trace_entropy_weight, extract_stages_from_trace,
    extract_all_stages, IncrementalTopKMean
)
from .processors import WrappedPerReqLogitsProcessor
from .tools import (
    cleanup_zoom_image, 
    execute_tool_call,
    parse_tool_calls_from_response
)
from functools import partial
import math
from PIL import Image
from qwen_vl_utils import process_vision_info
import random
from .inference_loop import (
    BatchInferenceLoop, TraceState, 
    get_image_path
)


class MultiTurnOutput:
    def __init__(self, token_ids, logprobs, texts, boundaries, tool_bboxes=None):
        self.token_ids = token_ids
        self.logprobs = logprobs
        self.texts = texts  # List of texts, one per turn
        self.turn_boundaries = boundaries  # List of (start, end) tuples
        self.tool_bboxes = tool_bboxes or []  # List of bbox coordinates from tool calls
        self.num_turns = len(texts)
        self.finish_reason = "stop"

class Reliable_TWI:
    """Enhanced LLM wrapper with deep thinking capabilities"""
    
    def __init__(self, model: str, **vllm_kwargs):
        """
        Initialize Reliable_TWI
        
        Args:
            model: Model path or name
            **vllm_kwargs: Additional arguments for vLLM initialization
        """
        self.model_name = model
        self.model_short_name = os.path.basename(model.rstrip('/'))
        self.run_id = f"{int(time.time() * 1e6)}_{random.randint(1,9)}"
        self.vllm_kwargs = vllm_kwargs
        
        # Initialize vLLM
        default_kwargs = {
            "tensor_parallel_size": len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
            "enable_prefix_caching": True,
            "trust_remote_code": True,
            "disable_log_stats": True,
        }
        
        # Auto-detect FP8 models and add required settings
        model_name_lower = model.lower()
        if 'fp8' in model_name_lower or 'int8' in model_name_lower:
            default_kwargs.update({
                "quantization": "fp8",
                "kv_cache_dtype": "fp8",
            })
        
        default_kwargs.update(vllm_kwargs)
        
        print("Initializing vLLM engine...")
        llm_init_start = time.time()
        self.llm = LLM(model=model, logits_processors=[WrappedPerReqLogitsProcessor], **default_kwargs)
        llm_init_time = time.time() - llm_init_start
        print(f"vLLM engine initialized in {llm_init_time:.2f} seconds")
        
        # Initialize tokenizer
        print("Initializing tokenizer...")
        tokenizer_init_start = time.time()
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        tokenizer_init_time = time.time() - tokenizer_init_start
        print(f"Tokenizer initialized in {tokenizer_init_time:.2f} seconds")
        
        # Store initialization times
        self.init_times = {
            'llm_init_time': llm_init_time,
            'tokenizer_init_time': tokenizer_init_time
        }
    
    def generate(self, *args, **kwargs):
        """Simple wrapper around vLLM's generate method"""
        return self.llm.generate(*args, **kwargs)
    
    def reliable_think(
        self,
        messages: Optional[list] = None,
        mode: str = "offline",
        warmup_traces: int = 8,
        budget: int = 32,
        sampling_params: Optional[SamplingParams] = None,
        compute_multiple_voting: bool = True,
        img_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        sample_name: Optional[str] = None,
        reasoning_effort: str = "low",
        batch_size: int = 32,
        save_zoom_images: bool = False,
        **kwargs
    ) -> DeepThinkOutput:
       
        total_start_time = time.time()
        
        # Create output object
        output = DeepThinkOutput()
        output.mode = mode
        output.llm_init_time = self.init_times['llm_init_time']
        output.tokenizer_init_time = self.init_times['tokenizer_init_time']
        
        if img_path and sample_name is None:
            sample_name = os.path.splitext(os.path.basename(img_path))[0]
            
        # Set configuration
        output.config = {
            "model": self.model_name,
            "mode": mode,
            "compute_multiple_voting": compute_multiple_voting,
        }
        
        # Prepare mode-specific parameters
        gamma = kwargs.get('gamma', 1.0)
        filtering_ratio = kwargs.get('dual_stage_filtering_ratio', 0.4)
        consensus_threshold = kwargs.get('consensus_threshold', 0.9)
        adaptive_step_size = kwargs.get('adaptive_step_size', 1)

        if mode == "online":
            output.config.update({
                "warmup_traces": warmup_traces,
                "budget": budget,
                "gamma": gamma,
                "dual_stage_filtering_ratio": filtering_ratio,
                "consensus_threshold": consensus_threshold,
                "adaptive_step_size": adaptive_step_size,
            })
            result = self._deepthink_online(messages, output, 
                warmup_traces, budget,
                sampling_params, reasoning_effort,
                img_path=img_path, dataset_name=dataset_name, 
                sample_name=sample_name, save_zoom_images=save_zoom_images,
                gamma=gamma, filtering_ratio=filtering_ratio,
                consensus_threshold=consensus_threshold, adaptive_step_size=adaptive_step_size
            )
        else:
            output.config.update({
                "budget": budget,
                "batch_size": batch_size,
                "gamma": gamma,
                "dual_stage_filtering_ratio": filtering_ratio,
            })
            # Extract sample_name from img_path if not provided
            result = self._deepthink_offline(messages, output,
                budget, sampling_params, img_path,
                dataset_name, sample_name, reasoning_effort, batch_size, save_zoom_images
            )

        # Perform multiple voting analysis if requested
        if compute_multiple_voting and output.all_traces:
            print("Computing multiple voting results...")
            voting_start = time.time()
            if output.mode == "online":
                # Use pre-computed thresholds from warmup for two-stage filtering
                thresholds = getattr(output, 'two_stage_thresholds', None)
                optimal_k = getattr(output, 'two_stage_optimal_k', None)
                output.voting_results = compute_all_voting_results_online(
                    output.all_traces, thresholds=thresholds, optimal_k=optimal_k, gamma=gamma
                )
            else:   
                output.voting_results = compute_all_voting_results(
                    output.all_traces, gamma=gamma, filtering_ratio=filtering_ratio
                )
            
            voting_time = time.time() - voting_start
            print(f"Multiple voting computed in {voting_time:.2f} seconds")
        
        output.total_time = time.time() - total_start_time
        output.print_summary()
        
        return output
    
    def _convert_states_to_outputs(self, states: List[TraceState]) -> List[MultiTurnOutput]:
        """Helper to convert TraceStates to MultiTurnOutputs"""
        all_outputs = []
        for state in states:
            combined_ids = []
            combined_logprobs = []
            boundaries = []
            current_pos = 0
            
            for vllm_out in state.vllm_outputs:
                out = vllm_out.outputs[0]
                t_ids = out.token_ids or []
                t_logprobs = out.logprobs or []
                
                start = current_pos
                end = current_pos + len(t_ids) - 1
                boundaries.append((start, end))
                
                combined_ids.extend(t_ids)
                combined_logprobs.extend(t_logprobs)
                current_pos += len(t_ids)
                
            final_output_wrapper = MultiTurnOutput(
                token_ids=combined_ids,
                logprobs=combined_logprobs,
                texts=state.turn_texts,
                boundaries=boundaries,
                tool_bboxes=state.tool_bboxes
            )
            all_outputs.append(final_output_wrapper)
        return all_outputs

    def _deepthink_online(
        self,
        messages: List[Dict],
        output: DeepThinkOutput,
        warmup_traces: int,
        budget: int,
        sampling_params: Optional[SamplingParams],
        reasoning_effort: Optional[str] = None,
        img_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        sample_name: Optional[str] = None,
        save_zoom_images: bool = False,
        gamma: float = 1.0,
        filtering_ratio: float = 0.4,
        consensus_threshold: float = 0.9,
        adaptive_step_size: int = 1
    ) -> DeepThinkOutput:
        """Online deep thinking with reliability-based selection and tool support"""
        
        processing_start = time.time()
        
        # Prepare inference loop
        executor = partial(execute_tool_call, model_short_name=self.model_short_name)
        inference_loop = BatchInferenceLoop(
            llm=self.llm,
            processor=self.processor,
            tool_executor_func=executor,
            tool_parser_func=parse_tool_calls_from_response
        )
        
        # Prepare base image path
        pil_image = Image.open(img_path) if img_path else None
        base_image_path = [get_image_path(pil_image)] if pil_image else []
        
        # Warmup phase
        print(f"Starting warmup phase...")
        warmup_gen_start = time.time()
        
        warmup_states = []
        base_seed = time.time_ns()
        
        for param_id in range(warmup_traces):
            params = copy.deepcopy(sampling_params) 
            params.logprobs = 10
            params.seed = base_seed + param_id
            params.n = 1
            
            state = TraceState(
                trace_id=param_id,
                initial_messages=messages,
                sampling_params=params,
                image_paths=base_image_path
            )
            warmup_states.append(state)
        
        # Output warmup traces
        inference_loop.run(
            trace_states=warmup_states,
            max_turns=10, 
            reasoning_effort=reasoning_effort,
            dataset_name=dataset_name,
            sample_name=sample_name,
            run_id=self.run_id,
            batch_start_idx=0,
            save_zoom_images=save_zoom_images
        )
        output.generation_time += time.time() - warmup_gen_start
        
        # Process warmup results using offline processor (handles multi-turn)
        warmup_process_start = time.time()
        warmup_outputs = self._convert_states_to_outputs(warmup_states)
        warmup_result = process_batch_results_offline(warmup_outputs, dataset_name)
        output.processing_time += time.time() - warmup_process_start
        
        output.warmup_traces = warmup_result['traces']
        output.total_tokens = warmup_result['total_tokens']
        
        if not save_zoom_images:
            for state in warmup_states:
                for path in state.image_paths[1:]:
                    cleanup_zoom_image(path)
        
        # Compute two-stage thresholds from warmup traces
        thinking_thresh, reason_thresh, optimal_k = compute_two_stage_thresholds(
            output.warmup_traces, filtering_ratio=filtering_ratio
        )
        output.two_stage_thresholds = (thinking_thresh, reason_thresh)
        output.two_stage_optimal_k = optimal_k
        final_gen_start = time.time()
        
        output.final_traces = []
        
        # Compute weights for warmup traces
        for trace in output.warmup_traces:
            trace['weight'] = compute_trace_weight(trace, optimal_k, thinking_thresh, reason_thresh, gamma)
        
        # Pool for consensus checking
        current_traces = output.warmup_traces[:]
        remaining_budget = budget - warmup_traces
        
        # Adaptive Sampling Loop
        while remaining_budget > 0:
            # Consensus Check
            valid_answers = []
            weights = []
            
            for trace in current_traces:
                if trace.get('extracted_answer'):
                    valid_answers.append(trace['extracted_answer'])
                    weights.append(trace.get('weight', 1.0))

            if valid_answers:
                winner = weighted_majority_vote(valid_answers, weights)
                if winner:
                    total_weight = sum(weights)
                    winner_weight = sum(w for a, w in zip(valid_answers, weights) if a == winner)
                    beta = winner_weight / total_weight if total_weight > 0 else 0.0
                    if beta >= consensus_threshold:
                        print(f"Consensus reached (beta={beta:.3f}). Stopping early.")
                        break

            step = min(adaptive_step_size, remaining_budget)
            current_states = []
            
            for i in range(step):
                current_seed = base_seed + (budget - remaining_budget) + i
                p = copy.deepcopy(sampling_params)
                p.logprobs = 10
                p.seed = current_seed
                p.n = 1
                # Online reliability settings
                p.extra_args = {
                    "rel_thresh": thinking_thresh if thinking_thresh != 0.0 else reason_thresh,
                    "eos_token_id": self.processor.tokenizer.eos_token_id,
                    "optimal_k": optimal_k,
                    "rel_topk": 10,
                }
                
                state = TraceState(
                    trace_id=budget - remaining_budget + i,
                    initial_messages=messages,
                    sampling_params=p,
                    image_paths=base_image_path
                )
                current_states.append(state)
            
            # Generate
            inference_loop.run(
                trace_states=current_states,
                max_turns=10,
                reasoning_effort=reasoning_effort,
                dataset_name=dataset_name,
                sample_name=sample_name,
                run_id=self.run_id,
                batch_start_idx=0,
                save_zoom_images=save_zoom_images
            )
            
            # Process
            batch_outputs = self._convert_states_to_outputs(current_states)
            batch_res = process_batch_results_offline(batch_outputs, dataset_name)
            
            output.final_traces.extend(batch_res['traces'])
            output.total_tokens += batch_res['total_tokens']
            
            if not save_zoom_images:
                for state in current_states:
                    for path in state.image_paths[1:]:
                        cleanup_zoom_image(path)
            
            # Update trace pool with weights
            for trace in batch_res['traces']:
                trace['weight'] = compute_trace_weight(trace, optimal_k, thinking_thresh, reason_thresh, gamma)
                current_traces.append(trace)
                
            remaining_budget -= step

        output.generation_time += time.time() - final_gen_start
        
        # Combine results
        output.all_traces = output.warmup_traces + output.final_traces
        output.total_traces_count = len(output.all_traces)
        output.avg_tokens_per_trace = output.total_tokens / output.total_traces_count if output.total_traces_count > 0 else 0
        
        output.processing_time = time.time() - processing_start
        return output
    
    def _deepthink_offline(
        self,
        messages: Optional[list],
        output: DeepThinkOutput,
        budget: int,
        sampling_params: Optional[SamplingParams],
        img_path: str,
        dataset_name: Optional[str] = None,
        sample_name: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        batch_size: int = 32,
        save_zoom_images: bool = False
    ) -> DeepThinkOutput:
        """Offline deep thinking - generate all traces at once with multi-turn tool calling support"""
        
        initial_messages = copy.deepcopy(messages)
        pil_image = Image.open(img_path) if img_path else None
        
        # Initialize inference loop
        executor = partial(execute_tool_call, model_short_name=self.model_short_name)
        inference_loop = BatchInferenceLoop(
            llm=self.llm,
            processor=self.processor,
            tool_executor_func=executor,
            tool_parser_func=parse_tool_calls_from_response
        )
        
        base_seed = time.time_ns()
        
        # Prepare initial image path list
        base_image_path = []
        if pil_image:
            original_path = get_image_path(pil_image)
            base_image_path = [original_path]
            
        all_trace_states = []
        generation_start = time.time()
        
        # Batch processing loop
        for batch_start in range(0, budget, batch_size):
            batch_end = min(batch_start + batch_size, budget)
            batch_size_actual = batch_end - batch_start
            
            # 1. Initialize states for this batch
            batch_states = []
            for offset in range(batch_size_actual):
                global_trace_idx = batch_start + offset
                
                # Prepare sampling params for this trace
                params = copy.deepcopy(sampling_params)
                params.n = 1
                params.logprobs = 10
                params.seed = base_seed + global_trace_idx
                
                # Create state
                state = TraceState(
                    trace_id=global_trace_idx,
                    initial_messages=initial_messages,
                    sampling_params=params,
                    image_paths=base_image_path # Copy inside __init__
                )
                batch_states.append(state)
            
            # 2. Run inference loop
            inference_loop.run(
                trace_states=batch_states,
                max_turns=10,
                reasoning_effort=reasoning_effort,
                dataset_name=dataset_name,
                sample_name=sample_name,
                run_id=self.run_id,
                batch_start_idx=batch_start,
                save_zoom_images=save_zoom_images
            )
            
            # 3. Collect completed states
            all_trace_states.extend(batch_states)
            
            # 4. Clean up zoom images if needed
            if not save_zoom_images:
                for state in batch_states:
                    for path in state.image_paths[1:]:
                        cleanup_zoom_image(path)

        output.generation_time = time.time() - generation_start
        
        # Convert TraceStates to MultiTurnOutputs for processing
        all_outputs = self._convert_states_to_outputs(all_trace_states)
        
        # Process results
        processing_start = time.time()
        
        processed_results = process_batch_results_offline(all_outputs, dataset_name)
        
        output.all_traces = processed_results['traces']
        output.total_tokens = processed_results['total_tokens']
        output.total_traces_count = len(output.all_traces)
        output.avg_tokens_per_trace = output.total_tokens / output.total_traces_count if output.total_traces_count > 0 else 0
        
        output.processing_time = time.time() - processing_start
        return output
    

