"""Temporal Bandwidth Optimizer - Research infrastructure for AI collaboration efficiency.

This package provides tools for measuring and optimizing collaborative reasoning
efficiency in autoregressive transformer deployments, addressing measured 
performance constraints in existing Claude/GPT systems.
"""

from .core.bandwidth_monitor import BandwidthMonitor, CollaborationMetrics
from .core.optimization_engine import OptimizationEngine, RoutingDecision, CacheStrategy
from .core.temporal_analyzer import TemporalAnalyzer
from .integrations.claude_adapter import ClaudeOptimizedClient
from .integrations.openai_adapter import OpenAIOptimizedClient
from .integrations.generic_llm import GenericLLMClient

__version__ = "0.1.0"
__author__ = "Claude AI Research"
__email__ = "research@anthropic.com"

__all__ = [
    "BandwidthMonitor",
    "CollaborationMetrics", 
    "OptimizationEngine",
    "RoutingDecision",
    "CacheStrategy",
    "TemporalAnalyzer",
    "ClaudeOptimizedClient",
    "OpenAIOptimizedClient",
    "GenericLLMClient",
]