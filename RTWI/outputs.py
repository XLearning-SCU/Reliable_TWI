"""
Output classes for DeepThinkLLM
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple


@dataclass
class DeepThinkOutput:
    """Output container for deep thinking results"""
    
    # Multiple voting results
    voting_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Traces
    warmup_traces: List[Dict[str, Any]] = field(default_factory=list)
    final_traces: List[Dict[str, Any]] = field(default_factory=list)
    all_traces: List[Dict[str, Any]] = field(default_factory=list)
    
    # Confidence information (for online mode)
    two_stage_thresholds: Optional[Tuple[float, float]] = None
    two_stage_optimal_k: Optional[int] = None
    
    # Statistics
    total_traces_count: int = 0
    
    # Token statistics
    total_tokens: int = 0
    avg_tokens_per_trace: float = 0.0
    
    # Timing information
    tokenizer_init_time: float = 0.0
    llm_init_time: float = 0.0
    generation_time: float = 0.0
    processing_time: float = 0.0
    total_time: float = 0.0
    
    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    mode: str = "offline"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            # Multiple voting results
            "voting_results": self.voting_results,
            
            # Traces
            "all_traces": self.all_traces,
            
            # Statistics
            "total_traces_count": self.total_traces_count,
            
            # Token statistics
            "token_stats": {
                "total_tokens": self.total_tokens,
                "avg_tokens_per_trace": self.avg_tokens_per_trace,
            },
            
            # Timing information
            "timing_stats": {
                "tokenizer_init_time": self.tokenizer_init_time,
                "llm_init_time": self.llm_init_time,
                "generation_time": self.generation_time,
                "processing_time": self.processing_time,
                "total_time": self.total_time,
            },
            
            # Configuration and metadata
            "config": self.config,
            "mode": self.mode,
            "timestamp": self.timestamp,
        }
    
    def print_summary(self):
        """Print a formatted summary of the results"""
        print(f"\n=== Reliable Thinking Summary ===")
        print(f"Mode: {self.mode}")
        
        if self.mode == "online":
            print(f"Total traces: {len(self.all_traces)}")
        else:
            print(f"Generated traces: {self.total_traces_count}")
        
        print(f"Total tokens: {self.total_tokens}")
        
        if self.generation_time > 0:
            print(f"Generation time: {self.generation_time:.2f}s")
            print(f"Generation throughput: {self.total_tokens / self.generation_time:.1f} tokens/second")
        
        print(f"Total time: {self.total_time:.2f}s")
        
        # Print voting results summary
        if self.voting_results:
            print(f"\n=== Voting Results Summary ===")
            for method, result in self.voting_results.items():
                if result and result.get('answer'):
                    num_votes = result.get('num_votes', 0)
                    print(f"  {method}: {result['answer']} [{num_votes} votes]")
    
    def print_detailed_voting_results(self):
        """Print detailed voting results"""
        if not self.voting_results:
            print("No voting results available.")
            return
        
        print(f"\n=== Detailed Voting Results ===")
        print("-" * 55)
        print(f"{'Method':<25} {'Answer':<20} {'Votes':<6}")
        print("-" * 55)
        
        for method, result in self.voting_results.items():
            if result:
                answer = result.get('answer', 'None')[:18] + '...' if len(str(result.get('answer', 'None'))) > 20 else str(result.get('answer', 'None'))
                num_votes = result.get('num_votes', 0)
                
                print(f"{method:<25} {answer:<20} {num_votes:<6}")
    
    @property
    def overall_throughput(self) -> float:
        """Overall token generation throughput"""
        if self.generation_time > 0:
            return self.total_tokens / self.generation_time
        return 0.0
    
    def get_voting_method_names(self) -> List[str]:
        """Get list of available voting method names"""
        return list(self.voting_results.keys())
    
    def get_voting_answers(self) -> Dict[str, str]:
        """Get answers from all voting methods"""
        return {method: result.get('answer') for method, result in self.voting_results.items() 
                if result and result.get('answer')}