"""Benchmark datasets for temporal bandwidth optimization research.

Provides standardized datasets for evaluating collaborative AI performance
across different domains and complexity levels.
"""

import json
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class TaskDomain(Enum):
    """Cognitive domains for collaborative tasks."""
    CREATIVE_WRITING = "creative_writing"
    RISK_ASSESSMENT = "risk_assessment" 
    BUSINESS_STRATEGY = "business_strategy"
    OPTIMIZATION = "optimization"
    PLANNING = "planning"
    RESEARCH_SYNTHESIS = "research_synthesis"
    DESIGN_THINKING = "design_thinking"
    TECHNICAL_ANALYSIS = "technical_analysis"
    DATA_INTERPRETATION = "data_interpretation"
    PROBLEM_SOLVING = "problem_solving"


@dataclass
class BenchmarkTask:
    """Standardized benchmark task for evaluation."""
    
    id: str
    domain: TaskDomain
    complexity: int  # 1-5 scale
    title: str
    description: str
    initial_prompt: str
    expected_turns: int
    evaluation_criteria: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['domain'] = self.domain.value
        return data


class BenchmarkDatasets:
    """Curated benchmark datasets for research validation."""
    
    def __init__(self):
        self.tasks = self._generate_comprehensive_dataset()
    
    def _generate_comprehensive_dataset(self) -> List[BenchmarkTask]:
        """Generate comprehensive benchmark dataset."""
        
        tasks = []
        
        # Generate tasks for each domain
        for domain in TaskDomain:
            domain_tasks = self._generate_domain_tasks(domain, tasks_per_domain=20)
            tasks.extend(domain_tasks)
        
        return tasks
    
    def _generate_domain_tasks(self, domain: TaskDomain, tasks_per_domain: int) -> List[BenchmarkTask]:
        """Generate tasks for specific domain."""
        
        task_generators = {
            TaskDomain.CREATIVE_WRITING: self._generate_creative_writing_tasks,
            TaskDomain.RISK_ASSESSMENT: self._generate_risk_assessment_tasks,
            TaskDomain.BUSINESS_STRATEGY: self._generate_business_strategy_tasks,
            TaskDomain.OPTIMIZATION: self._generate_optimization_tasks,
            TaskDomain.PLANNING: self._generate_planning_tasks,
            TaskDomain.RESEARCH_SYNTHESIS: self._generate_research_synthesis_tasks,
            TaskDomain.DESIGN_THINKING: self._generate_design_thinking_tasks,
            TaskDomain.TECHNICAL_ANALYSIS: self._generate_technical_analysis_tasks,
            TaskDomain.DATA_INTERPRETATION: self._generate_data_interpretation_tasks,
            TaskDomain.PROBLEM_SOLVING: self._generate_problem_solving_tasks
        }
        
        generator = task_generators.get(domain, self._generate_generic_tasks)
        return generator(tasks_per_domain)
    
    def _generate_creative_writing_tasks(self, count: int) -> List[BenchmarkTask]:
        """Generate creative writing benchmark tasks."""
        
        templates = [
            {
                "title": "Character Development Workshop",
                "description": "Collaborative character creation and development",
                "prompt": "I want to create a compelling character for my story. Help me develop their background, motivations, and character arc. Let's start with the basic concept and build from there.",
                "complexity": 3,
                "turns": 4
            },
            {
                "title": "Plot Structure Analysis", 
                "description": "Analyze and improve story plot structure",
                "prompt": "I have a story idea but the plot feels weak. Help me analyze the structure and identify ways to create more compelling tension and resolution.",
                "complexity": 4,
                "turns": 5
            },
            {
                "title": "Dialogue Enhancement",
                "description": "Improve dialogue authenticity and impact",
                "prompt": "The dialogue in my script feels flat. Help me make it more natural and impactful. Let's work through some specific scenes together.",
                "complexity": 2,
                "turns": 3
            }
        ]
        
        tasks = []
        for i in range(count):
            template = templates[i % len(templates)]
            
            task = BenchmarkTask(
                id=f"creative_writing_{i:03d}",
                domain=TaskDomain.CREATIVE_WRITING,
                complexity=template["complexity"] + random.randint(-1, 1),
                title=template["title"],
                description=template["description"],
                initial_prompt=template["prompt"],
                expected_turns=template["turns"] + random.randint(-1, 1),
                evaluation_criteria=["creativity", "coherence", "engagement", "collaborative_flow"],
                metadata={"template_id": i % len(templates)}
            )
            
            # Clamp complexity to valid range
            task.complexity = max(1, min(5, task.complexity))
            task.expected_turns = max(2, min(7, task.expected_turns))
            
            tasks.append(task)
        
        return tasks
    
    def _generate_risk_assessment_tasks(self, count: int) -> List[BenchmarkTask]:
        """Generate risk assessment benchmark tasks."""
        
        templates = [
            {
                "title": "Project Risk Analysis",
                "description": "Comprehensive project risk evaluation",
                "prompt": "I'm launching a new product and need to identify potential risks. Help me systematically assess technical, market, and operational risks and develop mitigation strategies.",
                "complexity": 4,
                "turns": 5
            },
            {
                "title": "Investment Risk Evaluation",
                "description": "Financial investment risk assessment",
                "prompt": "I'm considering a significant investment opportunity. Help me evaluate the risks involved and create a framework for making this decision.",
                "complexity": 5,
                "turns": 6
            },
            {
                "title": "Operational Risk Review",
                "description": "Ongoing operational risk management",
                "prompt": "Our organization needs to review operational risks. Help me identify vulnerabilities in our current processes and prioritize risk mitigation efforts.",
                "complexity": 3,
                "turns": 4
            }
        ]
        
        return self._build_tasks_from_templates(TaskDomain.RISK_ASSESSMENT, "risk_assessment", templates, count, 
                                              ["thoroughness", "risk_identification", "mitigation_quality", "decision_support"])
    
    def _generate_business_strategy_tasks(self, count: int) -> List[BenchmarkTask]:
        """Generate business strategy benchmark tasks."""
        
        templates = [
            {
                "title": "Market Entry Strategy",
                "description": "Develop strategy for new market entry",
                "prompt": "We want to expand into a new geographic market. Help me develop a comprehensive market entry strategy including competitive analysis and go-to-market approach.",
                "complexity": 5,
                "turns": 6
            },
            {
                "title": "Competitive Positioning",
                "description": "Strategic competitive positioning analysis",
                "prompt": "Our company needs to better position itself against competitors. Help me analyze our competitive landscape and develop a differentiation strategy.",
                "complexity": 4,
                "turns": 5
            },
            {
                "title": "Digital Transformation Planning",
                "description": "Strategic digital transformation roadmap",
                "prompt": "We need to modernize our business operations. Help me create a digital transformation strategy that aligns with our business goals.",
                "complexity": 4,
                "turns": 5
            }
        ]
        
        return self._build_tasks_from_templates(TaskDomain.BUSINESS_STRATEGY, "business_strategy", templates, count,
                                              ["strategic_thinking", "market_analysis", "feasibility", "actionability"])
    
    def _generate_optimization_tasks(self, count: int) -> List[BenchmarkTask]:
        """Generate optimization benchmark tasks."""
        
        templates = [
            {
                "title": "Process Optimization",
                "description": "Optimize existing business processes",
                "prompt": "Our customer service process is inefficient. Help me identify bottlenecks and design an optimized workflow that improves both speed and quality.",
                "complexity": 3,
                "turns": 4
            },
            {
                "title": "Resource Allocation",
                "description": "Optimize resource allocation decisions",
                "prompt": "I need to allocate limited resources across multiple projects. Help me develop an optimization framework that maximizes value while managing constraints.",
                "complexity": 4,
                "turns": 5
            },
            {
                "title": "Performance Tuning",
                "description": "System performance optimization",
                "prompt": "Our application is running slowly. Help me systematically identify performance bottlenecks and implement optimization strategies.",
                "complexity": 4,
                "turns": 4
            }
        ]
        
        return self._build_tasks_from_templates(TaskDomain.OPTIMIZATION, "optimization", templates, count,
                                              ["analytical_rigor", "solution_effectiveness", "implementation_clarity", "measurability"])
    
    def _generate_planning_tasks(self, count: int) -> List[BenchmarkTask]:
        """Generate planning benchmark tasks."""
        
        templates = [
            {
                "title": "Project Planning Workshop",
                "description": "Comprehensive project planning session",
                "prompt": "I'm starting a complex project and need to create a detailed plan. Help me break down the work, identify dependencies, and create a realistic timeline.",
                "complexity": 3,
                "turns": 5
            },
            {
                "title": "Strategic Planning Session",
                "description": "Long-term strategic planning",
                "prompt": "Our organization needs a 3-year strategic plan. Help me develop goals, identify key initiatives, and create a roadmap for execution.",
                "complexity": 5,
                "turns": 6
            },
            {
                "title": "Event Planning Coordination",
                "description": "Coordinate complex event planning",
                "prompt": "I'm organizing a large conference and need to coordinate multiple aspects. Help me create a comprehensive plan that ensures nothing falls through the cracks.",
                "complexity": 2,
                "turns": 4
            }
        ]
        
        return self._build_tasks_from_templates(TaskDomain.PLANNING, "planning", templates, count,
                                              ["comprehensiveness", "logical_structure", "feasibility", "detail_level"])
    
    def _generate_research_synthesis_tasks(self, count: int) -> List[BenchmarkTask]:
        """Generate research synthesis benchmark tasks."""
        
        templates = [
            {
                "title": "Literature Review Synthesis",
                "description": "Synthesize research findings across studies",
                "prompt": "I need to synthesize findings from multiple research papers on AI ethics. Help me identify key themes, contradictions, and gaps in the literature.",
                "complexity": 4,
                "turns": 5
            },
            {
                "title": "Market Research Analysis",
                "description": "Synthesize market research data",
                "prompt": "I have data from multiple market research sources. Help me synthesize the findings into actionable insights for product development.",
                "complexity": 3,
                "turns": 4
            },
            {
                "title": "Technical Research Summary",
                "description": "Synthesize technical research findings",
                "prompt": "I need to understand the current state of quantum computing research. Help me synthesize the latest developments and identify promising directions.",
                "complexity": 5,
                "turns": 5
            }
        ]
        
        return self._build_tasks_from_templates(TaskDomain.RESEARCH_SYNTHESIS, "research_synthesis", templates, count,
                                              ["synthesis_quality", "insight_generation", "clarity", "comprehensiveness"])
    
    def _generate_design_thinking_tasks(self, count: int) -> List[BenchmarkTask]:
        """Generate design thinking benchmark tasks."""
        
        templates = [
            {
                "title": "User Experience Design",
                "description": "Design user-centered solutions",
                "prompt": "Users are struggling with our mobile app interface. Help me apply design thinking principles to identify pain points and create better user experiences.",
                "complexity": 3,
                "turns": 5
            },
            {
                "title": "Service Design Innovation",
                "description": "Redesign service delivery processes",
                "prompt": "Our customer service experience needs improvement. Help me use design thinking to reimagine how we deliver value to customers.",
                "complexity": 4,
                "turns": 5
            },
            {
                "title": "Product Innovation Workshop",
                "description": "Innovate new product concepts",
                "prompt": "We need to innovate new product ideas for changing market needs. Help me use design thinking to explore opportunities and prototype solutions.",
                "complexity": 4,
                "turns": 6
            }
        ]
        
        return self._build_tasks_from_templates(TaskDomain.DESIGN_THINKING, "design_thinking", templates, count,
                                              ["user_centricity", "creativity", "iterative_process", "solution_viability"])
    
    def _generate_technical_analysis_tasks(self, count: int) -> List[BenchmarkTask]:
        """Generate technical analysis benchmark tasks."""
        
        templates = [
            {
                "title": "Architecture Review",
                "description": "Analyze system architecture decisions",
                "prompt": "I need to review our system architecture for scalability issues. Help me analyze the current design and identify areas for improvement.",
                "complexity": 4,
                "turns": 4
            },
            {
                "title": "Code Quality Assessment",
                "description": "Evaluate code quality and maintainability",
                "prompt": "Our codebase has grown complex and hard to maintain. Help me analyze code quality issues and develop improvement strategies.",
                "complexity": 3,
                "turns": 4
            },
            {
                "title": "Security Analysis",
                "description": "Technical security vulnerability assessment",
                "prompt": "I need to assess our application for security vulnerabilities. Help me systematically analyze potential threats and mitigation approaches.",
                "complexity": 5,
                "turns": 5
            }
        ]
        
        return self._build_tasks_from_templates(TaskDomain.TECHNICAL_ANALYSIS, "technical_analysis", templates, count,
                                              ["technical_accuracy", "depth_of_analysis", "practical_recommendations", "clarity"])
    
    def _generate_data_interpretation_tasks(self, count: int) -> List[BenchmarkTask]:
        """Generate data interpretation benchmark tasks."""
        
        templates = [
            {
                "title": "Data Pattern Analysis",
                "description": "Identify patterns in complex datasets",
                "prompt": "I have customer behavior data that shows interesting patterns. Help me interpret these patterns and understand what they mean for our business.",
                "complexity": 3,
                "turns": 4
            },
            {
                "title": "Performance Metrics Review",
                "description": "Interpret business performance metrics",
                "prompt": "Our quarterly metrics show mixed results. Help me interpret what the data is telling us and identify actionable insights.",
                "complexity": 2,
                "turns": 3
            },
            {
                "title": "Experimental Results Analysis",
                "description": "Interpret experimental or A/B test results",
                "prompt": "We ran an A/B test with unexpected results. Help me interpret the data and understand the implications for our product strategy.",
                "complexity": 4,
                "turns": 4
            }
        ]
        
        return self._build_tasks_from_templates(TaskDomain.DATA_INTERPRETATION, "data_interpretation", templates, count,
                                              ["analytical_insight", "statistical_understanding", "business_relevance", "actionability"])
    
    def _generate_problem_solving_tasks(self, count: int) -> List[BenchmarkTask]:
        """Generate problem solving benchmark tasks."""
        
        templates = [
            {
                "title": "Complex Problem Breakdown",
                "description": "Break down complex problems systematically",
                "prompt": "I'm facing a complex operational problem with multiple interconnected issues. Help me break it down systematically and develop a solution approach.",
                "complexity": 4,
                "turns": 5
            },
            {
                "title": "Root Cause Analysis",
                "description": "Identify root causes of recurring issues",
                "prompt": "We keep having the same problems despite multiple fixes. Help me conduct a thorough root cause analysis to address the underlying issues.",
                "complexity": 3,
                "turns": 4
            },
            {
                "title": "Decision Framework Development",
                "description": "Create frameworks for complex decisions",
                "prompt": "I need to make a complex decision with many trade-offs. Help me develop a systematic framework for evaluating options and making the best choice.",
                "complexity": 4,
                "turns": 5
            }
        ]
        
        return self._build_tasks_from_templates(TaskDomain.PROBLEM_SOLVING, "problem_solving", templates, count,
                                              ["problem_decomposition", "logical_reasoning", "solution_creativity", "practical_application"])
    
    def _generate_generic_tasks(self, count: int) -> List[BenchmarkTask]:
        """Generate generic benchmark tasks."""
        return self._generate_problem_solving_tasks(count)  # Fallback to problem solving
    
    def _build_tasks_from_templates(self, 
                                  domain: TaskDomain,
                                  prefix: str,
                                  templates: List[Dict[str, Any]],
                                  count: int,
                                  evaluation_criteria: List[str]) -> List[BenchmarkTask]:
        """Build tasks from template definitions."""
        
        tasks = []
        for i in range(count):
            template = templates[i % len(templates)]
            
            task = BenchmarkTask(
                id=f"{prefix}_{i:03d}",
                domain=domain,
                complexity=max(1, min(5, template["complexity"] + random.randint(-1, 1))),
                title=template["title"],
                description=template["description"],
                initial_prompt=template["prompt"],
                expected_turns=max(2, min(7, template["turns"] + random.randint(-1, 1))),
                evaluation_criteria=evaluation_criteria,
                metadata={"template_id": i % len(templates)}
            )
            
            tasks.append(task)
        
        return tasks
    
    def get_tasks_by_domain(self, domain: TaskDomain) -> List[BenchmarkTask]:
        """Get all tasks for a specific domain."""
        return [task for task in self.tasks if task.domain == domain]
    
    def get_tasks_by_complexity(self, complexity: int) -> List[BenchmarkTask]:
        """Get all tasks of specific complexity level."""
        return [task for task in self.tasks if task.complexity == complexity]
    
    def get_balanced_sample(self, 
                          tasks_per_domain: int = 5,
                          complexity_distribution: Optional[Dict[int, float]] = None) -> List[BenchmarkTask]:
        """Get balanced sample across domains and complexity levels.
        
        Args:
            tasks_per_domain: Number of tasks per domain
            complexity_distribution: Desired complexity distribution (default: uniform)
            
        Returns:
            Balanced sample of benchmark tasks
        """
        
        if complexity_distribution is None:
            complexity_distribution = {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.2, 5: 0.1}
        
        sample_tasks = []
        
        for domain in TaskDomain:
            domain_tasks = self.get_tasks_by_domain(domain)
            
            # Sample by complexity distribution
            domain_sample = []
            for complexity, proportion in complexity_distribution.items():
                complexity_tasks = [t for t in domain_tasks if t.complexity == complexity]
                n_complexity = max(1, int(tasks_per_domain * proportion))
                
                if complexity_tasks:
                    domain_sample.extend(random.sample(
                        complexity_tasks, 
                        min(n_complexity, len(complexity_tasks))
                    ))
            
            # Fill remaining slots if needed
            while len(domain_sample) < tasks_per_domain and len(domain_sample) < len(domain_tasks):
                remaining = [t for t in domain_tasks if t not in domain_sample]
                if remaining:
                    domain_sample.append(random.choice(remaining))
            
            sample_tasks.extend(domain_sample[:tasks_per_domain])
        
        return sample_tasks
    
    def export_dataset(self, filename: str, format: str = "json"):
        """Export benchmark dataset to file.
        
        Args:
            filename: Output filename
            format: Export format ("json", "csv")
        """
        
        if format == "json":
            with open(filename, "w") as f:
                json.dump([task.to_dict() for task in self.tasks], f, indent=2)
        
        elif format == "csv":
            import csv
            
            with open(filename, "w", newline="") as f:
                if self.tasks:
                    fieldnames = list(self.tasks[0].to_dict().keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for task in self.tasks:
                        writer.writerow(task.to_dict())
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        
        domain_counts = {}
        complexity_counts = {}
        
        for task in self.tasks:
            domain_counts[task.domain.value] = domain_counts.get(task.domain.value, 0) + 1
            complexity_counts[task.complexity] = complexity_counts.get(task.complexity, 0) + 1
        
        return {
            "total_tasks": len(self.tasks),
            "domain_distribution": domain_counts,
            "complexity_distribution": complexity_counts,
            "average_complexity": sum(t.complexity for t in self.tasks) / len(self.tasks),
            "average_expected_turns": sum(t.expected_turns for t in self.tasks) / len(self.tasks)
        }


if __name__ == "__main__":
    # Example usage
    dataset = BenchmarkDatasets()
    
    print("Benchmark Dataset Statistics:")
    stats = dataset.get_statistics()
    print(f"Total tasks: {stats['total_tasks']}")
    print(f"Average complexity: {stats['average_complexity']:.1f}")
    print(f"Domain distribution: {stats['domain_distribution']}")
    
    # Export sample dataset
    sample = dataset.get_balanced_sample(tasks_per_domain=3)
    print(f"\nBalanced sample: {len(sample)} tasks")
    
    # Save to file
    dataset.export_dataset("benchmark_dataset.json")
    print("Dataset exported to benchmark_dataset.json")