"""Temporal pattern analysis for collaborative reasoning systems.

This module provides advanced temporal analysis capabilities for understanding
and predicting collaborative bandwidth patterns based on the research findings.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import signal, fft
from collections import deque
import statistics


@dataclass
class TemporalPattern:
    """Represents a detected temporal pattern in collaboration."""
    
    pattern_type: str  # "degradation", "adaptation", "rhythm", "baseline"
    frequency: float  # Pattern frequency in Hz
    amplitude: float  # Pattern strength
    phase: float  # Pattern phase offset
    confidence: float  # Detection confidence (0-1)
    duration: float  # Pattern duration in seconds
    
    
@dataclass
class RhythmAnalysis:
    """Analysis of collaborative rhythm patterns."""
    
    dominant_frequency: float
    rhythm_strength: float
    variability: float
    synchronization_score: float
    disruption_events: List[float]
    natural_pace: float


class TemporalAnalyzer:
    """Advanced temporal analysis for collaborative reasoning systems.
    
    Analyzes temporal patterns in human-AI collaboration to detect:
    - Natural collaborative rhythms
    - Degradation patterns under latency stress
    - Adaptation strategies
    - Bandwidth constraint violations
    """
    
    def __init__(self, sampling_rate: float = 1.0, analysis_window: int = 100):
        """Initialize temporal analyzer.
        
        Args:
            sampling_rate: Analysis sampling rate in Hz
            analysis_window: Size of analysis window for pattern detection
        """
        self.sampling_rate = sampling_rate
        self.analysis_window = analysis_window
        
        # Temporal data buffers
        self.timestamps: deque = deque(maxlen=analysis_window)
        self.response_times: deque = deque(maxlen=analysis_window) 
        self.efficiency_values: deque = deque(maxlen=analysis_window)
        self.turn_intervals: deque = deque(maxlen=analysis_window)
        
        # Pattern detection state
        self.detected_patterns: List[TemporalPattern] = []
        self.baseline_rhythm: Optional[RhythmAnalysis] = None
        
    def add_measurement(self, 
                       timestamp: float,
                       response_time: float, 
                       efficiency: float,
                       turn_interval: Optional[float] = None):
        """Add new temporal measurement for analysis.
        
        Args:
            timestamp: Measurement timestamp
            response_time: AI response time in seconds
            efficiency: Current collaborative efficiency
            turn_interval: Time since last turn (optional)
        """
        self.timestamps.append(timestamp)
        self.response_times.append(response_time)
        self.efficiency_values.append(efficiency)
        
        if turn_interval is not None:
            self.turn_intervals.append(turn_interval)
        elif len(self.timestamps) > 1:
            interval = timestamp - self.timestamps[-2]
            self.turn_intervals.append(interval)
    
    def detect_collaborative_rhythm(self) -> RhythmAnalysis:
        """Detect natural collaborative rhythm patterns.
        
        Analyzes turn-taking patterns to identify natural collaborative
        frequencies and rhythm stability.
        
        Returns:
            RhythmAnalysis with detected rhythm characteristics
        """
        if len(self.turn_intervals) < 10:
            return RhythmAnalysis(
                dominant_frequency=0.0,
                rhythm_strength=0.0,
                variability=1.0,
                synchronization_score=0.0,
                disruption_events=[],
                natural_pace=0.0
            )
        
        intervals = np.array(list(self.turn_intervals))
        
        # Calculate dominant frequency using FFT
        fft_result = fft.fft(intervals)
        frequencies = fft.fftfreq(len(intervals), d=1/self.sampling_rate)
        
        # Find dominant frequency (excluding DC component)
        power_spectrum = np.abs(fft_result[1:len(fft_result)//2])
        freq_bins = frequencies[1:len(frequencies)//2]
        
        if len(power_spectrum) > 0:
            dominant_idx = np.argmax(power_spectrum)
            dominant_frequency = abs(freq_bins[dominant_idx])
            rhythm_strength = power_spectrum[dominant_idx] / np.sum(power_spectrum)
        else:
            dominant_frequency = 0.0
            rhythm_strength = 0.0
        
        # Calculate rhythm variability (coefficient of variation)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        variability = std_interval / mean_interval if mean_interval > 0 else 1.0
        
        # Detect disruption events (intervals > 2 standard deviations)
        threshold = mean_interval + 2 * std_interval
        disruption_events = []
        for i, interval in enumerate(intervals):
            if interval > threshold:
                disruption_events.append(float(self.timestamps[i]))
        
        # Calculate synchronization score (1 - normalized variance)
        normalized_variance = variability ** 2
        synchronization_score = max(0.0, 1.0 - normalized_variance)
        
        # Natural pace (inverse of mean interval)
        natural_pace = 1.0 / mean_interval if mean_interval > 0 else 0.0
        
        return RhythmAnalysis(
            dominant_frequency=dominant_frequency,
            rhythm_strength=rhythm_strength,
            variability=variability,
            synchronization_score=synchronization_score,
            disruption_events=disruption_events,
            natural_pace=natural_pace
        )
    
    def detect_degradation_patterns(self) -> List[TemporalPattern]:
        """Detect temporal degradation patterns in efficiency data.
        
        Identifies patterns consistent with the research findings:
        - Linear degradation under constant latency
        - Step-function degradation at threshold crossings
        - Exponential recovery patterns
        
        Returns:
            List of detected temporal patterns
        """
        if len(self.efficiency_values) < 20:
            return []
        
        patterns = []
        efficiency_array = np.array(list(self.efficiency_values))
        time_array = np.array(list(self.timestamps))
        
        # Detect linear degradation trends
        if len(efficiency_array) > 5:
            slope, intercept, r_value, p_value, std_err = \
                signal.linregress(range(len(efficiency_array)), efficiency_array)
            
            if slope < -0.001 and abs(r_value) > 0.7:  # Significant negative trend
                patterns.append(TemporalPattern(
                    pattern_type="degradation",
                    frequency=0.0,  # Trend, not oscillatory
                    amplitude=abs(slope),
                    phase=0.0,
                    confidence=abs(r_value),
                    duration=time_array[-1] - time_array[0]
                ))
        
        # Detect step-function changes (threshold crossings)
        step_indices = self._detect_step_changes(efficiency_array)
        for step_idx in step_indices:
            if step_idx > 0 and step_idx < len(efficiency_array) - 1:
                before_mean = np.mean(efficiency_array[max(0, step_idx-5):step_idx])
                after_mean = np.mean(efficiency_array[step_idx:min(len(efficiency_array), step_idx+5)])
                
                if before_mean - after_mean > 0.01:  # Significant drop
                    patterns.append(TemporalPattern(
                        pattern_type="threshold_crossing",
                        frequency=0.0,
                        amplitude=before_mean - after_mean,
                        phase=0.0,
                        confidence=0.8,
                        duration=0.0  # Instantaneous
                    ))
        
        # Detect oscillatory patterns in response times
        response_array = np.array(list(self.response_times))
        if len(response_array) > 10:
            oscillation_patterns = self._detect_oscillations(response_array)
            patterns.extend(oscillation_patterns)
        
        return patterns
    
    def _detect_step_changes(self, signal_data: np.ndarray) -> List[int]:
        """Detect step changes in signal using change point detection."""
        if len(signal_data) < 10:
            return []
        
        # Simple change point detection using sliding window variance
        window_size = 5
        step_indices = []
        
        for i in range(window_size, len(signal_data) - window_size):
            before_window = signal_data[i-window_size:i]
            after_window = signal_data[i:i+window_size]
            
            before_mean = np.mean(before_window)
            after_mean = np.mean(after_window)
            
            # Check for significant mean difference
            if abs(before_mean - after_mean) > 0.02:  # Threshold for significance
                step_indices.append(i)
        
        return step_indices
    
    def _detect_oscillations(self, signal_data: np.ndarray) -> List[TemporalPattern]:
        """Detect oscillatory patterns in signal data."""
        patterns = []
        
        # Apply FFT to detect dominant frequencies
        fft_result = fft.fft(signal_data)
        frequencies = fft.fftfreq(len(signal_data), d=1/self.sampling_rate)
        power_spectrum = np.abs(fft_result)
        
        # Find peaks in power spectrum
        peaks, properties = signal.find_peaks(
            power_spectrum[:len(power_spectrum)//2], 
            height=np.max(power_spectrum) * 0.1  # 10% of max power
        )
        
        for peak_idx in peaks:
            freq = abs(frequencies[peak_idx])
            amplitude = power_spectrum[peak_idx] / len(signal_data)
            
            # Calculate phase
            phase = np.angle(fft_result[peak_idx])
            
            # Calculate confidence based on peak prominence
            confidence = properties['peak_heights'][list(peaks).index(peak_idx)] / np.max(power_spectrum)
            
            if freq > 0.01:  # Exclude very low frequencies (trends)
                patterns.append(TemporalPattern(
                    pattern_type="oscillation",
                    frequency=freq,
                    amplitude=amplitude,
                    phase=phase,
                    confidence=confidence,
                    duration=len(signal_data) / self.sampling_rate
                ))
        
        return patterns
    
    def analyze_latency_impact(self, latency_values: List[float]) -> Dict[str, Any]:
        """Analyze impact of latency on collaborative patterns.
        
        Args:
            latency_values: List of latency measurements
            
        Returns:
            Dictionary with latency impact analysis
        """
        if len(latency_values) != len(self.efficiency_values):
            raise ValueError("Latency and efficiency arrays must have same length")
        
        latency_array = np.array(latency_values)
        efficiency_array = np.array(list(self.efficiency_values))
        
        # Calculate correlation between latency and efficiency
        correlation = np.corrcoef(latency_array, efficiency_array)[0, 1]
        
        # Fit mathematical model from research: E = baseline - (latency * degradation_rate)
        slope, intercept, r_value, p_value, std_err = \
            signal.linregress(latency_array, efficiency_array)
        
        # Predict efficiency at different latency levels
        test_latencies = [0.5, 1.0, 2.0, 5.0, 10.0]
        predicted_efficiencies = [intercept + slope * lat for lat in test_latencies]
        
        # Calculate bandwidth degradation percentages
        baseline_efficiency = intercept  # Efficiency at zero latency
        degradation_percentages = [
            max(0.0, (baseline_efficiency - eff) / baseline_efficiency * 100)
            for eff in predicted_efficiencies
        ]
        
        return {
            "correlation": correlation,
            "mathematical_model": {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value ** 2,
                "p_value": p_value,
                "standard_error": std_err
            },
            "latency_predictions": {
                "latencies": test_latencies,
                "predicted_efficiencies": predicted_efficiencies,
                "degradation_percentages": degradation_percentages
            },
            "baseline_efficiency": baseline_efficiency,
            "model_quality": "excellent" if r_value ** 2 > 0.95 else 
                           "good" if r_value ** 2 > 0.80 else "poor"
        }
    
    def predict_collaboration_breakdown(self) -> Dict[str, Any]:
        """Predict when collaboration will break down based on current trends.
        
        Returns:
            Dictionary with breakdown predictions and early warning indicators
        """
        if len(self.efficiency_values) < 10:
            return {"status": "insufficient_data"}
        
        efficiency_array = np.array(list(self.efficiency_values))
        time_array = np.array(list(self.timestamps))
        
        # Calculate trend
        slope, intercept, r_value, p_value, std_err = \
            signal.linregress(range(len(efficiency_array)), efficiency_array)
        
        # Critical efficiency threshold (below which collaboration becomes ineffective)
        critical_threshold = 0.05  # 5% of baseline efficiency
        
        predictions = {}
        
        if slope < 0:  # Declining efficiency
            current_efficiency = efficiency_array[-1]
            steps_to_breakdown = (current_efficiency - critical_threshold) / abs(slope)
            
            if steps_to_breakdown > 0:
                predictions["breakdown_in_steps"] = int(steps_to_breakdown)
                predictions["estimated_time_to_breakdown"] = steps_to_breakdown / self.sampling_rate
                predictions["confidence"] = abs(r_value)
            else:
                predictions["status"] = "already_below_threshold"
        else:
            predictions["status"] = "stable_or_improving"
        
        # Early warning indicators
        recent_efficiency = efficiency_array[-5:] if len(efficiency_array) >= 5 else efficiency_array
        efficiency_variance = np.var(recent_efficiency)
        
        warnings = []
        if efficiency_variance > 0.001:
            warnings.append("high_variability")
        if len(self.detected_patterns) > 0:
            degradation_patterns = [p for p in self.detected_patterns if p.pattern_type == "degradation"]
            if degradation_patterns:
                warnings.append("degradation_pattern_detected")
        
        predictions["early_warnings"] = warnings
        predictions["current_efficiency"] = float(efficiency_array[-1])
        predictions["efficiency_trend"] = "declining" if slope < 0 else "stable_or_improving"
        
        return predictions
    
    def export_temporal_analysis(self) -> Dict[str, Any]:
        """Export complete temporal analysis for research purposes."""
        rhythm_analysis = self.detect_collaborative_rhythm()
        degradation_patterns = self.detect_degradation_patterns()
        
        return {
            "rhythm_analysis": {
                "dominant_frequency": rhythm_analysis.dominant_frequency,
                "rhythm_strength": rhythm_analysis.rhythm_strength,
                "variability": rhythm_analysis.variability,
                "synchronization_score": rhythm_analysis.synchronization_score,
                "disruption_events": rhythm_analysis.disruption_events,
                "natural_pace": rhythm_analysis.natural_pace
            },
            "detected_patterns": [
                {
                    "pattern_type": p.pattern_type,
                    "frequency": p.frequency,
                    "amplitude": p.amplitude,
                    "phase": p.phase,
                    "confidence": p.confidence,
                    "duration": p.duration
                }
                for p in degradation_patterns
            ],
            "temporal_data": {
                "timestamps": list(self.timestamps),
                "response_times": list(self.response_times),
                "efficiency_values": list(self.efficiency_values),
                "turn_intervals": list(self.turn_intervals)
            },
            "analysis_metadata": {
                "sampling_rate": self.sampling_rate,
                "analysis_window": self.analysis_window,
                "data_points": len(self.timestamps)
            }
        }