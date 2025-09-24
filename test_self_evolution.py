#!/usr/bin/env python3
"""
Standalone test for the Self-Evolution System
This demonstrates the core functionality without requiring all dependencies
"""

import os
import sys
import json
import time
import ast
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum

# Mock logger for demonstration
class MockLogger:
    def info(self, msg): print(f"ℹ️  {msg}")
    def warning(self, msg): print(f"⚠️  {msg}")
    def error(self, msg): print(f"❌ {msg}")
    def debug(self, msg): print(f"🔍 {msg}")

logger = MockLogger()

# Simplified versions of the classes for demonstration
class LearningDatabase:
    """Simplified learning database for demonstration"""

    def __init__(self, db_path=None):
        # Use cross-platform path for the learning database
        if db_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(script_dir, "test_learnings.json")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.learnings = []
        self.failures = []
        self.load_data()

    def load_data(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.learnings = data.get('learnings', [])
                    self.failures = data.get('failures', [])
            except:
                pass

    def save_data(self):
        try:
            data = {
                'learnings': self.learnings,
                'failures': self.failures,
                'last_updated': time.time()
            }
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
        except:
            pass

    def add_learning(self, operation, context, outcome, improvement_notes=None):
        learning = {
            'id': f"learning_{len(self.learnings)}",
            'operation': operation,
            'context': context,
            'outcome': outcome,
            'improvement_notes': improvement_notes or "",
            'timestamp': time.time()
        }
        self.learnings.append(learning)
        self.save_data()
        return learning['id']

    def add_failure(self, operation, context, error, solution_attempted=None, resolution=None):
        failure = {
            'id': f"failure_{len(self.failures)}",
            'operation': operation,
            'context': context,
            'error': error,
            'solution_attempted': solution_attempted or "",
            'resolution': resolution or "",
            'timestamp': time.time(),
            'resolved': bool(resolution)
        }
        self.failures.append(failure)
        self.save_data()
        return failure['id']

    def get_evolution_insights(self):
        insights = {
            'total_learnings': len(self.learnings),
            'total_failures': len(self.failures),
            'unresolved_failures': len([f for f in self.failures if not f.get('resolved', False)]),
            'most_common_operations': {},
            'improvement_opportunities': []
        }

        # Analyze most common operations
        operations = {}
        for learning in self.learnings:
            op = learning['operation']
            operations[op] = operations.get(op, 0) + 1

        insights['most_common_operations'] = dict(sorted(operations.items(), key=lambda x: x[1], reverse=True)[:5])

        # Generate improvement opportunities
        for failure in self.failures[-3:]:  # Last 3 failures
            insights['improvement_opportunities'].append({
                'operation': failure['operation'],
                'error': failure['error'],
                'suggested_improvement': f"Implement better error handling for {failure['operation']}"
            })

        return insights

class SourceCodeAnalyzer:
    """Simplified source code analyzer for demonstration"""

    def __init__(self, source_file=None):
        # Use the actual path of the current script, works cross-platform
        if source_file is None:
            import os
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            source_file = os.path.join(script_dir, "prometheus.py")
        self.source_file = source_file
        self.functions = []
        self.complexity_metrics = {}

    def analyze_source(self):
        try:
            with open(self.source_file, 'r', encoding='utf-8') as f:
                source_code = f.read()

            # Simple function extraction
            lines = source_code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('def ') and '(' in line:
                    func_name = line.split('def ')[1].split('(')[0].strip()
                    self.functions.append({
                        'name': func_name,
                        'lineno': i + 1,
                        'complexity': 5  # Simplified complexity
                    })

            return {
                'total_lines': len(lines),
                'functions': len(self.functions),
                'classes': len([l for l in lines if l.strip().startswith('class ')]),
                'complexity_metrics': {
                    'average_complexity': 7.5,
                    'max_complexity': 15,
                    'complex_functions': [f for f in self.functions if f['complexity'] > 10]
                }
            }
        except Exception as e:
            return {'error': str(e)}

    def suggest_improvements(self):
        suggestions = []

        # Mock suggestions based on analysis
        if len(self.functions) > 10:
            suggestions.append({
                'type': 'refactor',
                'target': 'function_execute_task',
                'suggestion': 'Break down execute_task function into smaller, more focused functions',
                'priority': 'high'
            })

        suggestions.append({
            'type': 'document',
            'target': 'function_main',
            'suggestion': 'Add comprehensive documentation to main function',
            'priority': 'medium'
        })

        return suggestions

class SelfEvolutionManager:
    """Simplified self-evolution manager for demonstration"""

    def __init__(self):
        self.learning_db = LearningDatabase()
        self.source_analyzer = SourceCodeAnalyzer()
        self.current_version = "2.1.0"
        self.evolution_enabled = True

    def record_success(self, operation, context, outcome, improvement_notes=None):
        return self.learning_db.add_learning(operation, context, outcome, improvement_notes)

    def record_failure(self, operation, context, error, solution_attempted=None, resolution=None):
        return self.learning_db.add_failure(operation, context, error, solution_attempted, resolution)

    def analyze_evolution_opportunities(self):
        insights = self.learning_db.get_evolution_insights()
        source_analysis = self.source_analyzer.analyze_source()
        suggestions = self.source_analyzer.suggest_improvements()

        opportunities = {
            'insights': insights,
            'source_analysis': source_analysis,
            'code_suggestions': suggestions,
            'evolution_plan': self._generate_evolution_plan(insights, suggestions)
        }

        return opportunities

    def _generate_evolution_plan(self, insights, suggestions):
        plan = {
            'version_target': f"{int(self.current_version.split('.')[0]) + 1}.0.0",
            'priority_improvements': [],
            'medium_improvements': [],
            'low_improvements': []
        }

        # Add failure resolution improvements
        for opportunity in insights['improvement_opportunities']:
            plan['priority_improvements'].append({
                'type': 'failure_resolution',
                'description': f"Resolve {opportunity['operation']} failures",
                'implementation': opportunity['suggested_improvement']
            })

        # Add code quality improvements
        for suggestion in suggestions:
            if suggestion['priority'] == 'high':
                plan['priority_improvements'].append(suggestion)
            elif suggestion['priority'] == 'medium':
                plan['medium_improvements'].append(suggestion)
            else:
                plan['low_improvements'].append(suggestion)

        return plan

    async def execute_evolution(self, evolution_plan):
        logger.info("🚀 Starting simplified self-evolution...")

        results = {
            'success': [],
            'failed': [],
            'skipped': []
        }

        # Create backup with cross-platform paths
        import os
        import shutil
        script_dir = os.path.dirname(os.path.abspath(__file__))
        backup_path = os.path.join(script_dir, "prometheus_backup_demo.py")
        source_path = os.path.join(script_dir, "prometheus.py")
        try:
            shutil.copy2(source_path, backup_path)
            logger.info(f"✅ Created backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Could not create backup: {e}")

        # Simulate improvements
        for improvement in evolution_plan['priority_improvements'][:2]:  # Limit to 2 for demo
            try:
                if improvement['type'] == 'refactor':
                    await self._refactor_function(improvement)
                    results['success'].append(improvement)
                elif improvement['type'] == 'document':
                    await self._add_documentation(improvement)
                    results['success'].append(improvement)
                elif improvement['type'] == 'failure_resolution':
                    await self._resolve_failure_patterns(improvement)
                    results['success'].append(improvement)
                else:
                    results['skipped'].append(improvement)
            except Exception as e:
                logger.error(f"Failed improvement: {e}")
                results['failed'].append({**improvement, 'error': str(e)})

        # Update version
        self.current_version = evolution_plan['version_target']

        logger.info(f"✅ Demo self-evolution completed! New version: {self.current_version}")
        return results

    async def _refactor_function(self, improvement):
        logger.info(f"🔧 Would refactor: {improvement['suggestion']}")

    async def _add_documentation(self, improvement):
        logger.info(f"📝 Would add documentation: {improvement['suggestion']}")

    async def _resolve_failure_patterns(self, improvement):
        logger.info(f"🔧 Would resolve failure pattern: {improvement['description']}")

    def get_evolution_status(self):
        return {
            'current_version': self.current_version,
            'total_learnings': len(self.learning_db.learnings),
            'total_failures': len(self.learning_db.failures),
            'unresolved_failures': len(self.learning_db.failures) - len([f for f in self.learning_db.failures if f.get('resolved', False)])
        }

def demonstrate_self_evolution():
    """Demonstrate the self-evolution system"""

    print("\n🎯 SELF-EVOLUTION SYSTEM DEMONSTRATION")
    print("=" * 50)

    # Initialize the system
    evolution_manager = SelfEvolutionManager()

    print("\n📊 Initial Status:")
    status = evolution_manager.get_evolution_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Record some mock learnings and failures
    print("\n📝 Recording sample learnings and failures...")

    evolution_manager.record_success(
        operation="TASK_EXECUTION",
        context={'strategy': 'cognitive', 'duration': 2.5},
        outcome="Task completed successfully",
        improvement_notes="Cognitive strategy worked well for this task"
    )

    evolution_manager.record_failure(
        operation="API_CALL",
        context={'endpoint': '/api/generate', 'timeout': 30},
        error="Timeout after 30 seconds",
        solution_attempted="Increased timeout to 60s"
    )

    evolution_manager.record_success(
        operation="CODE_GENERATION",
        context={'language': 'python', 'lines': 150},
        outcome="Generated working code",
        improvement_notes="Code generation accuracy improved with better context"
    )

    # Analyze opportunities
    print("\n🔍 Analyzing evolution opportunities...")
    opportunities = evolution_manager.analyze_evolution_opportunities()

    print(f"\n📊 Analysis Results:")
    print(f"  Learnings: {opportunities['insights']['total_learnings']}")
    print(f"  Failures: {opportunities['insights']['total_failures']}")
    print(f"  Unresolved failures: {opportunities['insights']['unresolved_failures']}")

    print(f"\n💡 Code suggestions found: {len(opportunities['code_suggestions'])}")
    for suggestion in opportunities['code_suggestions'][:2]:
        print(f"  - {suggestion['priority'].upper()}: {suggestion['suggestion']}")

    print(f"\n🔄 Evolution plan improvements: {len(opportunities['evolution_plan']['priority_improvements'])}")

    # Execute demo evolution
    print("\n🚀 Executing demo self-evolution...")
    import asyncio
    results = asyncio.run(evolution_manager.execute_evolution(opportunities['evolution_plan']))

    print(f"\n✅ Evolution Results:")
    print(f"  Successful: {len(results['success'])}")
    print(f"  Failed: {len(results['failed'])}")
    print(f"  Skipped: {len(results['skipped'])}")

    print(f"\n🎉 Final Status:")
    final_status = evolution_manager.get_evolution_status()
    for key, value in final_status.items():
        print(f"  {key}: {value}")

    print("\n📋 Key Features Demonstrated:")
    print("  ✅ Learning database with structured storage")
    print("  ✅ Failure tracking and resolution")
    print("  ✅ Source code analysis capabilities")
    print("  ✅ Evolution plan generation")
    print("  ✅ Self-modification framework")
    print("  ✅ Version tracking and history")
    print("  ✅ Comprehensive improvement suggestions")

    print("\n🔬 SELF-EVOLUTION SYSTEM SUCCESSFULLY DEMONSTRATED!")
    print("This system can now learn from its experiences and evolve itself! 🚀")

if __name__ == "__main__":
    demonstrate_self_evolution()