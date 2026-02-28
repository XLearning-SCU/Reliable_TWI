import torch
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)
from vllm import SamplingParams
from vllm.config import VllmConfig
from typing import Optional
import heapq


class RelPerReqLogitsProcessor:
    """Token-level reliability-based early stopping using highest-k entropy.
    
    During generation, tracks top-k highest entropy values incrementally.
    When token count >= optimal_k, checks if mean of highest-k entropies exceeds threshold.
    If entropy > threshold (i.e., reliability too low), forces EOS token.
    """

    def __init__(self, rel_thresh: float, eos_token_id: int, optimal_k: int, rel_topk: int = 10) -> None:
        """
        Args:
            rel_thresh: Reliability threshold (negative entropy). Stop if rel < rel_thresh.
            eos_token_id: Token ID for EOS.
            optimal_k: Number of highest entropy values to track.
            rel_topk: Number of top probabilities to use for entropy calculation.
        """
        self.rel_thresh = rel_thresh  # This is negative (rel = -entropy)
        self.entropy_thresh = -rel_thresh  # Convert to entropy threshold
        self.eos_token_id = eos_token_id
        self.optimal_k = optimal_k
        self.rel_topk = rel_topk
        
        # Min-heap to track k highest entropy values (for IncrementalTopKMean)
        self.heap = []
        self.current_sum = 0.0
        self.token_count = 0

    def compute_entropy(self, logits: torch.Tensor) -> float:
        """Compute token-level entropy from logits using top-k probabilities."""
        probabilities = torch.softmax(logits, dim=-1)
        top_probs, _ = torch.topk(probabilities, self.rel_topk, dim=-1)
        # Normalize top-k probs
        top_probs = top_probs / top_probs.sum()
        # Compute entropy: -sum(p * log(p))
        log_probs = torch.log(top_probs + 1e-10)
        return -(top_probs * log_probs).sum().item()

    def _add_to_topk(self, value: float):
        """Add value to incremental top-k tracker (tracks k highest values)."""
        if len(self.heap) < self.optimal_k:
            heapq.heappush(self.heap, value)
            self.current_sum += value
        else:
            # If value is larger than the smallest of the top-k (heap root)
            if value > self.heap[0]:
                removed = heapq.heapreplace(self.heap, value)
                self.current_sum += (value - removed)

    def _get_topk_mean(self) -> float:
        """Get mean of current top-k highest values."""
        if not self.heap:
            return 0.0
        return self.current_sum / len(self.heap)

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        # Compute entropy for current token
        token_entropy = self.compute_entropy(logits)
        
        # Add to top-k tracker
        self._add_to_topk(token_entropy)
        self.token_count += 1
        
        # Check if should early stop (only after we have enough tokens)
        if self.token_count >= self.optimal_k:
            chunk_entropy = self._get_topk_mean()
            # If entropy > threshold, reliability is too low, force EOS
            if chunk_entropy > self.entropy_thresh:
                val_to_keep = logits[self.eos_token_id].item()
                logits[:] = float("-inf")
                logits[self.eos_token_id] = val_to_keep
        
        return logits


class WrappedPerReqLogitsProcessor(AdapterLogitsProcessor):
    """Adapter for per-request reliability-based logits processor."""

    def __init__(
        self, vllm_config, device: torch.device, is_pin_memory: bool
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        self.is_cuda = device.type == "cuda"

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional:
        """Create request-level logits processor if required args are present."""
        if not self.is_cuda or not params.extra_args:
            return None
        
        rel_thresh = params.extra_args.get("rel_thresh")
        eos_token_id = params.extra_args.get("eos_token_id")
        optimal_k = params.extra_args.get("optimal_k")
        rel_topk = params.extra_args.get("rel_topk", 10)
        
        if rel_thresh is None or eos_token_id is None or optimal_k is None:
            return None
        
        return RelPerReqLogitsProcessor(rel_thresh, eos_token_id, optimal_k, rel_topk)
