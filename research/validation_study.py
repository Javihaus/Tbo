"""Research validation study replication.

Replicates the 200-task, 4-condition experimental design from the research
paper to validate bandwidth optimization effectiveness with statistical analysis.
"""

import asyncio
import time
import random
import json
import csv
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats
import pandas as pd

from ..src.tboptimizer import (
    BandwidthMonitor, 
    ClaudeOptimizedClient, 
    CollaborationContext,
    OptimizationLevel
)


@dataclass
class ExperimentalCondition:
    """Defines an experimental condition for the study."""
    
    name: str
    artificial_delay: float  # Additional delay in seconds
    description: str
    expected_degradation: float  # Expected bandwidth degradation %


@dataclass
class TaskDefinition:
    """Defines a collaborative task for the experiment."""
    
    id: str
    domain: str  # e.g., "creative_writing", "technical_analysis"
    complexity: int  # 1-5 scale
    description: str
    initial_prompt: str
    expected_turns: int  # Expected number of conversation turns
    

@dataclass
class ExperimentResult:
    """Results from a single experimental trial."""
    
    task_id: str
    condition: str
    total_time: float
    total_turns: int
    efficiency: float  # turns per second
    response_times: List[float]
    bandwidth_ratio: float
    degradation_percentage: float
    

class ValidationStudy:
    """Replicates the research validation study with systematic measurement."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 enable_detailed_logging: bool = True):
        """Initialize validation study.
        
        Args:
            api_key: API key for LLM provider
            enable_detailed_logging: Enable detailed experimental logging
        """
        self.api_key = api_key
        self.enable_detailed_logging = enable_detailed_logging
        
        # Define experimental conditions (replicating research)
        self.conditions = [
            ExperimentalCondition(
                name="baseline",
                artificial_delay=0.0,
                description="Normal API response times without additional delay",
                expected_degradation=0.0
            ),
            ExperimentalCondition(
                name="low_delay", 
                artificial_delay=2.0,
                description="2-second artificial delays",
                expected_degradation=21.0
            ),
            ExperimentalCondition(
                name="medium_delay",
                artificial_delay=5.0, 
                description="5-second artificial delays",
                expected_degradation=41.0
            ),
            ExperimentalCondition(
                name="high_delay",
                artificial_delay=10.0,
                description="10-second artificial delays", 
                expected_degradation=56.0
            )
        ]
        
        # Initialize task library
        self.tasks = self._generate_task_library()
        
        # Results storage
        self.results: List[ExperimentResult] = []
        self.baseline_efficiency: Optional[float] = None
        
    def _generate_task_library(self) -> List[TaskDefinition]:
        """Generate 200 diverse collaborative tasks across domains."""
        
        domains = [
            "creative_writing", "risk_assessment", "business_strategy",
            "optimization", "planning", "research_synthesis", 
            "design_thinking", "technical_analysis", "data_interpretation",
            "problem_solving"
        ]
        
        tasks = []
        
        for i in range(200):
            domain = domains[i % len(domains)]
            task_id = f"{domain}_{i:03d}"
            complexity = random.randint(1, 5)
            
            # Generate domain-specific prompts
            prompts = self._generate_domain_prompts(domain, complexity)
            
            task = TaskDefinition(
                id=task_id,
                domain=domain,
                complexity=complexity,
                description=prompts["description"],
                initial_prompt=prompts["initial_prompt"],
                expected_turns=random.randint(3, 6)
            )
            
            tasks.append(task)
        
        return tasks
    
    def _generate_domain_prompts(self, domain: str, complexity: int) -> Dict[str, str]:
        """Generate domain-specific prompts based on research methodology."""
        
        prompt_templates = {
            "creative_writing": {
                "description": f"Creative writing task (complexity {complexity})",
                "initial_prompt": "Help me develop a compelling story concept. I need to create characters, plot structure, and narrative themes. Let's work through this step by step."
            },
            "risk_assessment": {
                "description": f"Risk assessment task (complexity {complexity})",
                "initial_prompt": "I need to conduct a comprehensive risk assessment for a new project. Help me identify potential risks, assess their impact, and develop mitigation strategies."
            },
            "business_strategy": {
                "description": f"Business strategy task (complexity {complexity})",
                "initial_prompt": "Let's develop a business strategy for entering a new market. I need help analyzing competition, identifying opportunities, and creating an action plan."
            },
            "technical_analysis": {
                "description": f"Technical analysis task (complexity {complexity})",
                "initial_prompt": "I need to perform a technical analysis of system architecture. Help me evaluate performance bottlenecks, scalability concerns, and optimization opportunities."
            },
            "problem_solving": {
                "description": f"Problem solving task (complexity {complexity})",
                "initial_prompt": "I'm facing a complex problem that requires systematic analysis. Help me break it down, explore solutions, and evaluate trade-offs."
            }
        }
        
        # Add complexity-based variations
        if complexity >= 4:
            base_prompt = prompt_templates.get(domain, prompt_templates["problem_solving"])
            base_prompt["initial_prompt"] += " This is a high-complexity scenario requiring deep analysis and multiple iterations."
        
        return prompt_templates.get(domain, prompt_templates["problem_solving"])
    
    async def run_full_study(self, 
                           tasks_per_condition: int = 50,
                           randomize_order: bool = True) -> Dict[str, Any]:
        """Run the complete validation study.
        
        Args:
            tasks_per_condition: Number of tasks per experimental condition
            randomize_order: Randomize task and condition order
            
        Returns:
            Complete study results with statistical analysis
        """
        print("Starting validation study replication...")
        print(f"Total tasks: {tasks_per_condition * len(self.conditions)}")
        
        # Prepare experimental schedule
        experimental_schedule = []
        for condition in self.conditions:
            condition_tasks = random.sample(self.tasks, tasks_per_condition) if randomize_order else self.tasks[:tasks_per_condition]
            for task in condition_tasks:
                experimental_schedule.append((task, condition))
        
        if randomize_order:
            random.shuffle(experimental_schedule)
        
        # Run experiments
        total_experiments = len(experimental_schedule)
        for i, (task, condition) in enumerate(experimental_schedule):
            print(f"Running experiment {i+1}/{total_experiments}: {task.domain} under {condition.name}")
            
            try:
                result = await self._run_single_experiment(task, condition)
                self.results.append(result)
                
                # Store baseline efficiency for comparison
                if condition.name == "baseline" and self.baseline_efficiency is None:
                    self.baseline_efficiency = result.efficiency
                
                # Add delay between experiments to avoid rate limiting
                await asyncio.sleep(random.uniform(1.0, 3.0))
                
            except Exception as e:
                print(f"Error in experiment {i+1}: {e}")
                continue
        
        # Analyze results
        analysis_results = self._analyze_results()
        
        # Save results
        if self.enable_detailed_logging:
            self._save_results(analysis_results)
        
        return analysis_results
    
    async def _run_single_experiment(self, 
                                   task: TaskDefinition,
                                   condition: ExperimentalCondition) -> ExperimentResult:
        """Run a single experimental trial."""
        
        # Initialize optimized client for this experiment
        client = ClaudeOptimizedClient(
            api_key=self.api_key,
            optimization_level=OptimizationLevel.RESEARCH,
            enable_monitoring=True
        )
        
        # Create collaboration context
        context = CollaborationContext(
            session_id=f"{task.id}_{condition.name}_{int(time.time())}",
            task_type=task.domain,
            complexity=task.complexity,
            metadata={"experiment": True, "condition": condition.name}
        )
        
        # Track experiment timing
        experiment_start = time.time()
        response_times = []
        turn_count = 0
        
        # Initial message
        messages = [{"role": "user", "content": task.initial_prompt}]
        
        # Simulate conversation turns
        for turn in range(task.expected_turns):
            turn_start = time.time()
            
            # Apply artificial delay for non-baseline conditions
            if condition.artificial_delay > 0:
                await asyncio.sleep(condition.artificial_delay)
            
            # Get AI response
            response = await client.collaborate(
                messages=messages,
                context=context,
                max_tokens=300
            )
            
            turn_time = time.time() - turn_start
            response_times.append(turn_time)
            turn_count += 1
            
            # Add response to conversation
            messages.append({"role": "assistant", "content": response.content})
            
            # Generate next user message (simulated)
            if turn < task.expected_turns - 1:
                next_prompt = self._generate_follow_up_prompt(task.domain, turn, response.content)
                messages.append({"role": "user", "content": next_prompt})
        
        # Calculate metrics
        total_time = time.time() - experiment_start
        efficiency = turn_count / total_time if total_time > 0 else 0.0
        
        # Calculate bandwidth ratio and degradation
        baseline_eff = self.baseline_efficiency or 0.125  # Research baseline
        bandwidth_ratio = efficiency / baseline_eff if baseline_eff > 0 else 0.0
        degradation_percentage = max(0.0, (baseline_eff - efficiency) / baseline_eff * 100)
        
        return ExperimentResult(
            task_id=task.id,
            condition=condition.name,
            total_time=total_time,
            total_turns=turn_count,
            efficiency=efficiency,
            response_times=response_times,
            bandwidth_ratio=bandwidth_ratio,
            degradation_percentage=degradation_percentage
        )
    
    def _generate_follow_up_prompt(self, domain: str, turn: int, previous_response: str) -> str:
        """Generate realistic follow-up prompts for conversation simulation."""
        
        follow_up_templates = {
            0: ["Can you elaborate on that?", "What are the next steps?", "How should we proceed?"],
            1: ["That's helpful. Can you provide more details?", "What else should we consider?", "Are there any alternatives?"],
            2: ["Let's refine this approach.", "Can you give me a specific example?", "How would you implement this?"],
            3: ["This looks good. Any final recommendations?", "What are the potential risks?", "How do we measure success?"]
        }
        
        turn_prompts = follow_up_templates.get(turn, follow_up_templates[3])
        return random.choice(turn_prompts)
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze experimental results with statistical testing."""
        
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Group results by condition
        condition_results = {}
        for condition in self.conditions:
            condition_results[condition.name] = [
                r for r in self.results if r.condition == condition.name
            ]
        
        # Calculate summary statistics for each condition
        condition_stats = {}
        for condition_name, results in condition_results.items():
            if not results:
                continue
                
            efficiencies = [r.efficiency for r in results]
            times = [r.total_time for r in results]
            turns = [r.total_turns for r in results]
            degradations = [r.degradation_percentage for r in results]
            
            condition_stats[condition_name] = {
                "n": len(results),
                "efficiency_mean": np.mean(efficiencies),
                "efficiency_std": np.std(efficiencies),
                "time_mean": np.mean(times),
                "time_std": np.std(times),
                "turns_mean": np.mean(turns),
                "turns_std": np.std(turns),
                "degradation_mean": np.mean(degradations),
                "degradation_std": np.std(degradations)
            }
        
        # Statistical significance testing
        baseline_efficiencies = [r.efficiency for r in condition_results.get("baseline", [])]
        significance_tests = {}
        
        for condition_name, results in condition_results.items():
            if condition_name == "baseline" or not baseline_efficiencies:
                continue
                
            condition_efficiencies = [r.efficiency for r in results]
            
            if len(condition_efficiencies) > 0 and len(baseline_efficiencies) > 0:
                t_stat, p_value = stats.ttest_ind(baseline_efficiencies, condition_efficiencies)
                effect_size = (np.mean(baseline_efficiencies) - np.mean(condition_efficiencies)) / np.sqrt(
                    (np.var(baseline_efficiencies) + np.var(condition_efficiencies)) / 2
                )
                
                significance_tests[condition_name] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "effect_size_cohens_d": float(effect_size),
                    "significant": p_value < 0.001  # Research used p < 0.001
                }
        
        # Mathematical model validation (additive delay model)
        model_validation = self._validate_additive_model()
        
        # Research comparison
        research_comparison = self._compare_to_research_findings()
        
        return {
            "experimental_summary": {
                "total_experiments": len(self.results),
                "conditions_tested": len(self.conditions),
                "baseline_efficiency": self.baseline_efficiency
            },
            "condition_statistics": condition_stats,
            "significance_tests": significance_tests,
            "model_validation": model_validation,
            "research_comparison": research_comparison,
            "raw_results": [asdict(r) for r in self.results]
        }
    
    def _validate_additive_model(self) -> Dict[str, Any]:
        """Validate the additive delay model from research."""
        
        # Test if observed times match predicted times from additive model
        model_accuracy = {}
        
        for condition in self.conditions:
            if condition.name == "baseline":
                continue
                
            condition_results = [r for r in self.results if r.condition == condition.name]
            if not condition_results:
                continue
            
            # Calculate predictions using additive model
            baseline_time = 33.90  # Research baseline
            predicted_times = []
            observed_times = []
            
            for result in condition_results:
                predicted_time = baseline_time + (result.total_turns * condition.artificial_delay)
                predicted_times.append(predicted_time)
                observed_times.append(result.total_time)
            
            # Calculate correlation and accuracy
            if len(predicted_times) > 1:
                correlation = np.corrcoef(predicted_times, observed_times)[0, 1]
                mean_absolute_error = np.mean(np.abs(np.array(predicted_times) - np.array(observed_times)))
                
                model_accuracy[condition.name] = {
                    "correlation": float(correlation),
                    "mean_absolute_error": float(mean_absolute_error),
                    "model_fit": "excellent" if correlation > 0.95 else "good" if correlation > 0.80 else "poor"
                }
        
        return model_accuracy
    
    def _compare_to_research_findings(self) -> Dict[str, Any]:
        """Compare results to published research findings."""
        
        # Research findings for comparison
        research_benchmarks = {
            "low_delay": {"expected_degradation": 21.0, "expected_bandwidth_ratio": 0.79},
            "medium_delay": {"expected_degradation": 41.0, "expected_bandwidth_ratio": 0.59}, 
            "high_delay": {"expected_degradation": 56.0, "expected_bandwidth_ratio": 0.44}
        }
        
        comparison_results = {}
        
        for condition_name, benchmark in research_benchmarks.items():
            condition_results = [r for r in self.results if r.condition == condition_name]
            
            if not condition_results:
                continue
            
            observed_degradation = np.mean([r.degradation_percentage for r in condition_results])
            observed_bandwidth_ratio = np.mean([r.bandwidth_ratio for r in condition_results])
            
            comparison_results[condition_name] = {
                "expected_degradation": benchmark["expected_degradation"],
                "observed_degradation": float(observed_degradation),
                "degradation_difference": float(abs(benchmark["expected_degradation"] - observed_degradation)),
                "expected_bandwidth_ratio": benchmark["expected_bandwidth_ratio"],
                "observed_bandwidth_ratio": float(observed_bandwidth_ratio),
                "bandwidth_ratio_difference": float(abs(benchmark["expected_bandwidth_ratio"] - observed_bandwidth_ratio)),
                "replication_quality": "excellent" if abs(benchmark["expected_degradation"] - observed_degradation) < 5.0 else "good" if abs(benchmark["expected_degradation"] - observed_degradation) < 10.0 else "poor"
            }
        
        return comparison_results
    
    def _save_results(self, analysis_results: Dict[str, Any]):
        """Save experimental results in research formats."""
        
        timestamp = int(time.time())
        
        # Save as JSON for programmatic access
        with open(f"validation_study_results_{timestamp}.json", "w") as f:
            json.dump(analysis_results, f, indent=2)
        
        # Save as CSV for statistical analysis
        results_df = pd.DataFrame([asdict(r) for r in self.results])
        results_df.to_csv(f"validation_study_data_{timestamp}.csv", index=False)
        
        # Save summary table in academic format
        self._save_academic_table(analysis_results, timestamp)
        
        print(f"Results saved with timestamp {timestamp}")
    
    def _save_academic_table(self, analysis_results: Dict[str, Any], timestamp: int):
        """Save results in academic paper table format."""
        
        condition_stats = analysis_results.get("condition_statistics", {})
        significance_tests = analysis_results.get("significance_tests", {})
        
        # Create academic format table
        table_data = []
        for condition in self.conditions:
            stats = condition_stats.get(condition.name, {})
            sig_test = significance_tests.get(condition.name, {})
            
            row = {
                "Condition": condition.name.replace("_", " ").title(),
                "Time (s)": f"{stats.get('time_mean', 0):.2f} ± {stats.get('time_std', 0):.2f}",
                "Efficiency (E)": f"{stats.get('efficiency_mean', 0):.3f} ± {stats.get('efficiency_std', 0):.3f}",
                "Turns": f"{stats.get('turns_mean', 0):.1f} ± {stats.get('turns_std', 0):.1f}",
                "Degradation (%)": f"{stats.get('degradation_mean', 0):.1f}%",
                "t-statistic": f"{sig_test.get('t_statistic', 0):.3f}" if sig_test else "—",
                "p-value": f"< 0.001" if sig_test.get('p_value', 1) < 0.001 else f"{sig_test.get('p_value', 1):.3f}" if sig_test else "—",
                "Cohen's d": f"{sig_test.get('effect_size_cohens_d', 0):.3f}" if sig_test else "—"
            }
            table_data.append(row)
        
        # Save as CSV table
        table_df = pd.DataFrame(table_data)
        table_df.to_csv(f"academic_results_table_{timestamp}.csv", index=False)


# Convenience function for running the study
async def run_validation_study(api_key: Optional[str] = None, 
                             tasks_per_condition: int = 50) -> Dict[str, Any]:
    """Convenience function to run the complete validation study.
    
    Args:
        api_key: API key for LLM provider
        tasks_per_condition: Number of tasks per condition (default 50, full study uses 50)
        
    Returns:
        Complete study results
    """
    study = ValidationStudy(api_key=api_key, enable_detailed_logging=True)
    return await study.run_full_study(tasks_per_condition=tasks_per_condition)


if __name__ == "__main__":
    # Example usage
    import os
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: ANTHROPIC_API_KEY not found. Using placeholder.")
    
    # Run a smaller validation study for testing
    results = asyncio.run(run_validation_study(api_key=api_key, tasks_per_condition=5))
    
    print("\nValidation Study Results:")
    print(f"Total experiments: {results['experimental_summary']['total_experiments']}")
    print(f"Baseline efficiency: {results['experimental_summary']['baseline_efficiency']:.3f}")
    
    if "research_comparison" in results:
        print("\nComparison to Research Findings:")
        for condition, comparison in results["research_comparison"].items():
            print(f"{condition}: {comparison['replication_quality']} replication")
            print(f"  Expected degradation: {comparison['expected_degradation']}%")
            print(f"  Observed degradation: {comparison['observed_degradation']:.1f}%")