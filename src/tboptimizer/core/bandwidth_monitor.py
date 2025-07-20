"""Real-time measurement of collaborative reasoning efficiency.

This module implements precise bandwidth monitoring following the mathematical
relationships discovered in the research (R² > 0.99), enabling measurement
of collaborative reasoning patterns and degradation detection.
"""

import time
import statistics
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from scipy import stats


@dataclass
class CollaborationMetrics:
    """Comprehensive metrics for collaborative reasoning performance."""
    
    turns_per_second: float
    total_turns: int
    total_time: float
    efficiency_ratio: float
    response_times: List[float]
    turn_gaps: List[float]
    bandwidth_degradation: float
    statistical_significance: float
    confidence_interval: tuple
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for export."""
        return {
            "turns_per_second": self.turns_per_second,
            "total_turns": self.total_turns, 
            "total_time": self.total_time,
            "efficiency_ratio": self.efficiency_ratio,
            "response_times": self.response_times,
            "turn_gaps": self.turn_gaps,
            "bandwidth_degradation": self.bandwidth_degradation,
            "statistical_significance": self.statistical_significance,
            "confidence_interval": self.confidence_interval,
        }


@dataclass 
class InteractionRecord:
    """Single interaction record for precise measurement."""
    
    timestamp: float
    response_time: float
    turn_number: int
    context_length: int = 0
    task_complexity: int = 1
    degradation_detected: bool = False


class BandwidthMonitor:
    """Real-time measurement of collaborative reasoning efficiency.
    
    Implements the mathematical framework from the research paper:
    E = N_turns / t_T (efficiency)
    γ = E_condition / E_baseline (bandwidth ratio)
    
    Provides circuit breaker functionality at 2-second threshold and
    statistical significance testing for degradation detection.
    """
    
    def __init__(self, 
                 window_size: int = 50,
                 degradation_threshold: float = 2.0,
                 baseline_efficiency: Optional[float] = None):
        """Initialize bandwidth monitor.
        
        Args:
            window_size: Rolling window size for efficiency calculations
            degradation_threshold: Circuit breaker threshold in seconds
            baseline_efficiency: Reference efficiency for comparison
        """
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.baseline_efficiency = baseline_efficiency or 0.125  # From research
        
        self.interactions: deque = deque(maxlen=window_size)
        self.session_start = time.time()
        self.total_interactions = 0
        self.circuit_breaker_triggered = False
        
        # Statistical tracking
        self.efficiency_history: List[float] = []
        self.response_time_history: List[float] = []
    
    def track_interaction(self, 
                         response_time: float, 
                         turn_count: int,
                         context_length: int = 0,
                         task_complexity: int = 1) -> Dict[str, Any]:
        """Track a single collaborative interaction.
        
        Args:
            response_time: Time for AI response in seconds
            turn_count: Current turn number in conversation
            context_length: Length of conversation context
            task_complexity: Task complexity rating (1-5)
            
        Returns:
            Dictionary with immediate metrics and degradation status
        """
        current_time = time.time()
        
        # Create interaction record
        record = InteractionRecord(
            timestamp=current_time,
            response_time=response_time,
            turn_number=turn_count,
            context_length=context_length,
            task_complexity=task_complexity
        )
        
        # Check for degradation
        degradation_detected = self.detect_degradation_threshold(response_time)
        record.degradation_detected = degradation_detected
        
        # Store interaction
        self.interactions.append(record)
        self.total_interactions += 1
        self.response_time_history.append(response_time)
        
        # Calculate current efficiency
        current_efficiency = self.calculate_efficiency_ratio()
        self.efficiency_history.append(current_efficiency)
        
        return {
            "efficiency": current_efficiency,
            "response_time": response_time,
            "degradation_detected": degradation_detected,
            "circuit_breaker": self.circuit_breaker_triggered,
            "turn_number": turn_count,
            "bandwidth_ratio": current_efficiency / self.baseline_efficiency
        }
    
    def calculate_efficiency_ratio(self) -> float:
        """Calculate current collaborative efficiency (turns per second).
        
        Returns:
            Current efficiency ratio following E = N_turns / t_T formula
        """
        if not self.interactions:
            return 0.0
            
        # Use rolling window for real-time calculation
        window_interactions = list(self.interactions)
        if len(window_interactions) < 2:
            return 0.0
            
        time_span = window_interactions[-1].timestamp - window_interactions[0].timestamp
        if time_span <= 0:
            return 0.0
            
        return len(window_interactions) / time_span
    
    def detect_degradation_threshold(self, response_time: float) -> bool:
        """Detect if response time exceeds degradation threshold.
        
        Implements circuit breaker at 2-second threshold based on research
        finding that this creates 21% bandwidth degradation.
        
        Args:
            response_time: Current response time in seconds
            
        Returns:
            True if degradation threshold exceeded
        """
        if response_time >= self.degradation_threshold:
            self.circuit_breaker_triggered = True
            return True
        return False
    
    def calculate_statistical_significance(self) -> tuple:
        """Calculate statistical significance of efficiency degradation.
        
        Returns:
            Tuple of (t_statistic, p_value) for current vs baseline efficiency
        """
        if len(self.efficiency_history) < 5:
            return (0.0, 1.0)
            
        # Compare recent efficiency with baseline
        recent_efficiency = self.efficiency_history[-min(10, len(self.efficiency_history)):]
        baseline_array = [self.baseline_efficiency] * len(recent_efficiency)
        
        try:
            t_stat, p_value = stats.ttest_ind(recent_efficiency, baseline_array)
            return (float(t_stat), float(p_value))
        except:
            return (0.0, 1.0)
    
    def calculate_confidence_interval(self, confidence: float = 0.95) -> tuple:
        """Calculate confidence interval for current efficiency.
        
        Args:
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(self.efficiency_history) < 3:
            return (0.0, 0.0)
            
        recent_efficiency = self.efficiency_history[-min(20, len(self.efficiency_history)):]
        mean_eff = statistics.mean(recent_efficiency)
        std_eff = statistics.stdev(recent_efficiency) if len(recent_efficiency) > 1 else 0.0
        
        # Calculate confidence interval
        alpha = 1 - confidence
        n = len(recent_efficiency)
        t_critical = stats.t.ppf(1 - alpha/2, n-1)
        margin_error = t_critical * (std_eff / np.sqrt(n))
        
        return (mean_eff - margin_error, mean_eff + margin_error)
    
    def generate_performance_report(self) -> CollaborationMetrics:
        """Generate comprehensive performance report.
        
        Returns:
            CollaborationMetrics object with complete analysis
        """
        if not self.interactions:
            return CollaborationMetrics(
                turns_per_second=0.0,
                total_turns=0,
                total_time=0.0,
                efficiency_ratio=0.0,
                response_times=[],
                turn_gaps=[],
                bandwidth_degradation=0.0,
                statistical_significance=1.0,
                confidence_interval=(0.0, 0.0)
            )
        
        # Calculate metrics
        current_efficiency = self.calculate_efficiency_ratio()
        total_time = time.time() - self.session_start
        response_times = [r.response_time for r in self.interactions]
        
        # Calculate turn gaps (time between interactions)
        turn_gaps = []
        for i in range(1, len(self.interactions)):
            gap = self.interactions[i].timestamp - self.interactions[i-1].timestamp
            turn_gaps.append(gap)
        
        # Statistical analysis
        t_stat, p_value = self.calculate_statistical_significance()
        confidence_interval = self.calculate_confidence_interval()
        
        # Bandwidth degradation calculation
        bandwidth_degradation = max(0.0, 
            (self.baseline_efficiency - current_efficiency) / self.baseline_efficiency)
        
        return CollaborationMetrics(
            turns_per_second=current_efficiency,
            total_turns=len(self.interactions),
            total_time=total_time,
            efficiency_ratio=current_efficiency / self.baseline_efficiency,
            response_times=response_times,
            turn_gaps=turn_gaps,
            bandwidth_degradation=bandwidth_degradation,
            statistical_significance=p_value,
            confidence_interval=confidence_interval
        )
    
    def reset_session(self):
        """Reset monitoring session for new measurement."""
        self.interactions.clear()
        self.session_start = time.time()
        self.total_interactions = 0
        self.circuit_breaker_triggered = False
        self.efficiency_history.clear()
        self.response_time_history.clear()
    
    def export_data(self, format: str = "dict") -> Any:
        """Export monitoring data for research analysis.
        
        Args:
            format: Export format ("dict", "pandas", "numpy")
            
        Returns:
            Data in requested format
        """
        if format == "dict":
            return {
                "interactions": [
                    {
                        "timestamp": r.timestamp,
                        "response_time": r.response_time,
                        "turn_number": r.turn_number,
                        "context_length": r.context_length,
                        "task_complexity": r.task_complexity,
                        "degradation_detected": r.degradation_detected
                    }
                    for r in self.interactions
                ],
                "efficiency_history": self.efficiency_history,
                "response_time_history": self.response_time_history,
                "session_metrics": self.generate_performance_report().to_dict()
            }
        elif format == "pandas":
            try:
                import pandas as pd
                return pd.DataFrame([
                    {
                        "timestamp": r.timestamp,
                        "response_time": r.response_time, 
                        "turn_number": r.turn_number,
                        "context_length": r.context_length,
                        "task_complexity": r.task_complexity,
                        "degradation_detected": r.degradation_detected
                    }
                    for r in self.interactions
                ])
            except ImportError:
                raise ImportError("pandas required for DataFrame export")
        elif format == "numpy":
            return {
                "response_times": np.array(self.response_time_history),
                "efficiency_history": np.array(self.efficiency_history),
                "timestamps": np.array([r.timestamp for r in self.interactions])
            }
        else:
            raise ValueError(f"Unsupported format: {format}")