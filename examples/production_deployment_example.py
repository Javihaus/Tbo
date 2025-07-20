"""Production deployment example.

Enterprise-ready deployment template showing how to integrate temporal
bandwidth optimization into production AI systems with monitoring,
scaling, and infrastructure considerations.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sys
import os
from contextlib import asynccontextmanager

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tboptimizer import (
    ClaudeOptimizedClient,
    BandwidthMonitor,
    OptimizationEngine,
    CollaborationContext,
    OptimizationLevel
)


@dataclass
class ProductionConfig:
    """Production deployment configuration."""
    
    optimization_level: str = "balanced"
    target_latency: float = 0.5
    cache_size: int = 10000
    enable_monitoring: bool = True
    max_concurrent_requests: int = 100
    rate_limit_per_minute: int = 1000
    circuit_breaker_threshold: float = 2.0
    health_check_interval: int = 60
    metrics_export_interval: int = 300


class ProductionAISystem:
    """Production-ready AI system with temporal bandwidth optimization."""
    
    def __init__(self, config: ProductionConfig, api_key: Optional[str] = None):
        """Initialize production AI system."""
        self.config = config
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        # Setup logging
        self._setup_logging()
        
        # Initialize optimized clients
        self.clients = self._initialize_client_pool()
        
        # Global monitoring
        self.system_monitor = BandwidthMonitor(
            degradation_threshold=config.circuit_breaker_threshold,
            baseline_efficiency=0.125
        )
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.circuit_breaker_active = False
        
        # Metrics for monitoring
        self.metrics = {
            "requests_per_minute": 0,
            "average_response_time": 0.0,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0,
            "bandwidth_efficiency": 0.0
        }
        
        self.logger.info("Production AI system initialized")
    
    def _setup_logging(self):
        """Setup production logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ai_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_client_pool(self) -> List[ClaudeOptimizedClient]:
        """Initialize pool of optimized clients for load balancing."""
        
        pool_size = min(10, self.config.max_concurrent_requests // 10)
        clients = []
        
        for i in range(pool_size):
            client = ClaudeOptimizedClient(
                api_key=self.api_key,
                optimization_level=OptimizationLevel(self.config.optimization_level),
                enable_monitoring=self.config.enable_monitoring,
                cache_size=self.config.cache_size // pool_size,
                target_latency=self.config.target_latency
            )
            clients.append(client)
        
        self.logger.info(f"Initialized {pool_size} optimized clients")
        return clients
    
    async def process_request(self, 
                            messages: List[Dict[str, str]],
                            user_id: str,
                            session_id: str,
                            task_type: str = "general",
                            priority: int = 5,
                            **kwargs) -> Dict[str, Any]:
        """Process AI request with production-grade error handling and monitoring."""
        
        request_start = time.time()
        self.request_count += 1
        
        try:
            # Rate limiting check
            if not await self._check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            # Circuit breaker check
            if self.circuit_breaker_active:
                raise Exception("Circuit breaker active - system overloaded")
            
            # Select optimal client from pool
            client = await self._select_optimal_client()
            
            # Create context
            context = CollaborationContext(
                session_id=session_id,
                user_id=user_id,
                task_type=task_type,
                priority=priority,
                metadata={
                    "production": True,
                    "request_id": f"req_{self.request_count}",
                    "timestamp": request_start
                }
            )
            
            # Process request with timeout
            response = await asyncio.wait_for(
                client.collaborate(messages=messages, context=context, **kwargs),
                timeout=self.config.target_latency * 4  # Allow 4x target latency
            )
            
            # Update monitoring
            response_time = time.time() - request_start
            self._update_system_metrics(response_time, response, success=True)
            
            # Log successful request
            self.logger.info(
                f"Request processed successfully - "
                f"user: {user_id}, session: {session_id}, "
                f"time: {response_time:.3f}s, "
                f"cache_hit: {response.cache_hit}"
            )
            
            return {
                "success": True,
                "content": response.content,
                "response_time": response_time,
                "optimization_applied": response.optimization_applied,
                "cache_hit": response.cache_hit,
                "bandwidth_metrics": response.bandwidth_metrics,
                "request_id": f"req_{self.request_count}"
            }
            
        except asyncio.TimeoutError:
            self.error_count += 1
            response_time = time.time() - request_start
            self._update_system_metrics(response_time, None, success=False)
            
            self.logger.error(f"Request timeout - user: {user_id}, time: {response_time:.3f}s")
            
            return {
                "success": False,
                "error": "Request timeout",
                "response_time": response_time,
                "request_id": f"req_{self.request_count}"
            }
            
        except Exception as e:
            self.error_count += 1
            response_time = time.time() - request_start
            self._update_system_metrics(response_time, None, success=False)
            
            self.logger.error(f"Request failed - user: {user_id}, error: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "response_time": response_time,
                "request_id": f"req_{self.request_count}"
            }
    
    async def _check_rate_limit(self) -> bool:
        """Check if request is within rate limits."""
        # Simplified rate limiting - in production, use Redis or similar
        current_minute = int(time.time() // 60)
        
        # This is a simplified implementation
        # In production, implement proper sliding window rate limiting
        return True  # Placeholder
    
    async def _select_optimal_client(self) -> ClaudeOptimizedClient:
        """Select optimal client from pool based on current load."""
        
        # Simple round-robin selection
        # In production, implement more sophisticated load balancing
        client_index = self.request_count % len(self.clients)
        return self.clients[client_index]
    
    def _update_system_metrics(self, 
                             response_time: float,
                             response: Optional[Any],
                             success: bool):
        """Update system-wide metrics."""
        
        # Update bandwidth monitor
        if self.system_monitor:
            self.system_monitor.track_interaction(
                response_time=response_time,
                turn_count=1,  # Single turn for this example
                context_length=0,
                task_complexity=1
            )
        
        # Update error rate
        self.metrics["error_rate"] = self.error_count / self.request_count if self.request_count > 0 else 0
        
        # Update average response time (moving average)
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = (current_avg * 0.9) + (response_time * 0.1)
        
        # Update cache hit rate if available
        if response and hasattr(response, 'cache_hit'):
            # Simplified cache hit rate calculation
            pass
        
        # Check circuit breaker conditions
        if self.metrics["error_rate"] > 0.1 or self.metrics["average_response_time"] > self.config.target_latency * 3:
            self.circuit_breaker_active = True
            self.logger.warning("Circuit breaker activated due to high error rate or latency")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "metrics": self.metrics.copy(),
            "circuit_breaker_active": self.circuit_breaker_active,
            "request_count": self.request_count,
            "error_count": self.error_count
        }
        
        # Check if system is healthy
        if self.metrics["error_rate"] > 0.05:
            health_status["status"] = "degraded"
        
        if self.circuit_breaker_active:
            health_status["status"] = "unhealthy"
        
        # Add bandwidth metrics if available
        if self.system_monitor:
            bandwidth_report = self.system_monitor.generate_performance_report()
            health_status["bandwidth_metrics"] = {
                "efficiency": bandwidth_report.turns_per_second,
                "degradation": bandwidth_report.bandwidth_degradation
            }
        
        return health_status
    
    async def export_metrics(self) -> Dict[str, Any]:
        """Export comprehensive metrics for monitoring systems."""
        
        metrics_export = {
            "timestamp": time.time(),
            "system_metrics": self.metrics.copy(),
            "performance_counters": {
                "total_requests": self.request_count,
                "total_errors": self.error_count,
                "uptime_seconds": time.time() - getattr(self, 'start_time', time.time())
            },
            "optimization_metrics": {}
        }
        
        # Add client-level metrics
        client_metrics = []
        for i, client in enumerate(self.clients):
            client_report = client.get_performance_report()
            client_metrics.append({
                "client_id": i,
                "report": client_report
            })
        
        metrics_export["client_metrics"] = client_metrics
        
        # Add bandwidth analysis
        if self.system_monitor:
            bandwidth_data = self.system_monitor.export_data("dict")
            metrics_export["bandwidth_analysis"] = bandwidth_data
        
        return metrics_export
    
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        
        self.start_time = time.time()
        
        async def health_check_loop():
            while True:
                await asyncio.sleep(self.config.health_check_interval)
                health = await self.health_check()
                self.logger.info(f"Health check: {health['status']}")
                
                # Auto-recovery from circuit breaker
                if self.circuit_breaker_active and health["metrics"]["error_rate"] < 0.02:
                    self.circuit_breaker_active = False
                    self.logger.info("Circuit breaker deactivated - system recovered")
        
        async def metrics_export_loop():
            while True:
                await asyncio.sleep(self.config.metrics_export_interval)
                metrics = await self.export_metrics()
                
                # In production, send to monitoring system (Prometheus, DataDog, etc.)
                self.logger.info(f"Metrics exported - requests: {metrics['performance_counters']['total_requests']}")
        
        # Start monitoring tasks
        asyncio.create_task(health_check_loop())
        asyncio.create_task(metrics_export_loop())
        
        self.logger.info("Monitoring tasks started")


class ProductionDeploymentDemo:
    """Demonstrates production deployment patterns."""
    
    def __init__(self):
        self.config = ProductionConfig(
            optimization_level="balanced",
            target_latency=0.5,
            cache_size=5000,
            max_concurrent_requests=50
        )
    
    async def run_production_demo(self) -> Dict[str, Any]:
        """Run production deployment demonstration."""
        
        print("🏭 Production Deployment Demonstration")
        print("=" * 45)
        print("Showing enterprise-ready AI system with temporal optimization")
        print()
        
        # Initialize production system
        system = ProductionAISystem(self.config)
        
        print("✅ Production system initialized")
        print(f"   - Optimization level: {self.config.optimization_level}")
        print(f"   - Target latency: {self.config.target_latency}s")
        print(f"   - Cache size: {self.config.cache_size}")
        print()
        
        # Start monitoring
        await system.start_monitoring()
        print("✅ Monitoring systems active")
        print()
        
        # Simulate production workload
        print("🔄 Simulating production workload...")
        
        workload_results = await self._simulate_production_workload(system)
        
        print("✅ Workload simulation completed")
        print()
        
        # Generate health report
        health_report = await system.health_check()
        metrics_report = await system.export_metrics()
        
        # Display results
        self._display_production_results(health_report, metrics_report, workload_results)
        
        return {
            "health_report": health_report,
            "metrics_report": metrics_report,
            "workload_results": workload_results,
            "config": asdict(self.config)
        }
    
    async def _simulate_production_workload(self, system: ProductionAISystem) -> Dict[str, Any]:
        """Simulate realistic production workload."""
        
        # Simulate various user requests
        test_scenarios = [
            {
                "messages": [{"role": "user", "content": "Help me analyze our quarterly sales data and identify trends."}],
                "user_id": "user_001", 
                "task_type": "data_analysis",
                "priority": 3
            },
            {
                "messages": [{"role": "user", "content": "Create a marketing strategy for our new product launch."}],
                "user_id": "user_002",
                "task_type": "strategy_planning", 
                "priority": 2
            },
            {
                "messages": [{"role": "user", "content": "Review this technical document for accuracy and clarity."}],
                "user_id": "user_003",
                "task_type": "content_review",
                "priority": 4
            }
        ]
        
        # Execute concurrent requests
        tasks = []
        for i in range(15):  # Simulate 15 concurrent requests
            scenario = test_scenarios[i % len(test_scenarios)]
            
            task = asyncio.create_task(
                system.process_request(
                    messages=scenario["messages"],
                    user_id=scenario["user_id"],
                    session_id=f"session_{i}",
                    task_type=scenario["task_type"],
                    priority=scenario["priority"]
                )
            )
            tasks.append(task)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_requests = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed_requests = [r for r in results if not isinstance(r, dict) or not r.get("success")]
        
        total_time = sum(r.get("response_time", 0) for r in successful_requests)
        avg_response_time = total_time / len(successful_requests) if successful_requests else 0
        
        cache_hits = sum(1 for r in successful_requests if r.get("cache_hit"))
        cache_hit_rate = cache_hits / len(successful_requests) if successful_requests else 0
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / len(results),
            "average_response_time": avg_response_time,
            "cache_hit_rate": cache_hit_rate,
            "results": results
        }
    
    def _display_production_results(self, 
                                  health_report: Dict[str, Any],
                                  metrics_report: Dict[str, Any],
                                  workload_results: Dict[str, Any]):
        """Display production demonstration results."""
        
        print("📊 PRODUCTION PERFORMANCE REPORT")
        print("=" * 40)
        print()
        
        # System health
        print(f"System Status: {health_report['status'].upper()}")
        print(f"Circuit Breaker: {'ACTIVE' if health_report['circuit_breaker_active'] else 'INACTIVE'}")
        print()
        
        # Workload performance
        print("Workload Performance:")
        print(f"  Total requests:     {workload_results['total_requests']}")
        print(f"  Success rate:       {workload_results['success_rate']:.1%}")
        print(f"  Avg response time:  {workload_results['average_response_time']:.3f}s")
        print(f"  Cache hit rate:     {workload_results['cache_hit_rate']:.1%}")
        print()
        
        # System metrics
        print("System Metrics:")
        print(f"  Error rate:         {health_report['metrics']['error_rate']:.1%}")
        print(f"  Bandwidth efficiency: {health_report['metrics']['bandwidth_efficiency']:.3f}")
        print()
        
        # Performance assessment
        print("🎯 Performance Assessment:")
        
        if workload_results['success_rate'] > 0.95:
            print("   ✅ Excellent reliability")
        elif workload_results['success_rate'] > 0.90:
            print("   ✓ Good reliability")
        else:
            print("   ⚠️ Reliability needs improvement")
        
        if workload_results['average_response_time'] < 1.0:
            print("   ✅ Excellent response times")
        elif workload_results['average_response_time'] < 2.0:
            print("   ✓ Good response times")
        else:
            print("   ⚠️ Response times need optimization")
        
        if workload_results['cache_hit_rate'] > 0.3:
            print("   ✅ Effective caching")
        elif workload_results['cache_hit_rate'] > 0.1:
            print("   ✓ Moderate caching")
        else:
            print("   ⚠️ Caching efficiency low")
        
        print()
        print("💡 Production Recommendations:")
        print("   - Monitor bandwidth efficiency trends")
        print("   - Set up alerts for circuit breaker activation")
        print("   - Scale client pool based on request volume")
        print("   - Implement proper rate limiting with Redis")
        print("   - Export metrics to monitoring system (Prometheus/DataDog)")


async def run_production_demo():
    """Run the production deployment demonstration."""
    
    demo = ProductionDeploymentDemo()
    
    try:
        results = await demo.run_production_demo()
        
        print("\n" + "=" * 50)
        print("Production deployment demo completed!")
        print("System ready for enterprise deployment.")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Production demo failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    print("Starting production deployment demonstration...")
    print("This shows enterprise-ready AI system integration.")
    print()
    
    # Run the demo
    results = asyncio.run(run_production_demo())
    
    if results and "error" not in results:
        print("\n🚀 Deployment Ready!")
        print("Key production features demonstrated:")
        print("  - Optimized client pooling")
        print("  - Real-time monitoring")
        print("  - Circuit breaker protection")
        print("  - Comprehensive metrics")
        print("  - Health check endpoints")
        print()
        print("📋 Next Steps for Production:")
        print("  1. Configure monitoring alerts")
        print("  2. Set up load balancing")
        print("  3. Implement proper rate limiting")
        print("  4. Add authentication/authorization")
        print("  5. Deploy with container orchestration")
    
    print("\n📖 For deployment guide, see docs/deployment_guide.md")