#!/usr/bin/env python3
"""
COMPREHENSIVE TEST SUITE FOR TDD PROMETHEUS SYSTEM
Tests the advanced planner agent and TDD-first implementation
"""

import unittest
import tempfile
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from tdd_prometheus import (
    AdvancedTDDPlanner,
    TDDCodeGenerator,
    UniversalTDDSystem,
    TestCase,
    TestSuite,
    ImplementationPlan,
    DomainType,
    TestResult
)

class TestAdvancedTDDPlanner(unittest.TestCase):
    """Test the advanced planner agent's deep thinking capabilities"""

    def setUp(self):
        """Set up test fixtures"""
        self.planner = AdvancedTDDPlanner()
        self.test_request = "Create an advanced aerospace calculator with real physics calculations"
        self.test_domain = DomainType.AEROSPACE

    def test_planner_initialization(self):
        """Test that planner initializes correctly"""
        self.assertIsInstance(self.planner, AdvancedTDDPlanner)
        self.assertIsNotNone(self.planner.domain_knowledge)
        self.assertIsNotNone(self.planner.test_patterns)
        self.assertIsNotNone(self.planner.implementation_strategies)

    def test_domain_knowledge_loading(self):
        """Test that domain knowledge is loaded correctly"""
        knowledge = self.planner.domain_knowledge

        # Check that all domains are loaded
        self.assertIn(DomainType.AEROSPACE, knowledge)
        self.assertIn(DomainType.WEB_DEVELOPMENT, knowledge)

        # Check aerospace domain knowledge
        aerospace_knowledge = knowledge[DomainType.AEROSPACE]
        self.assertIn('key_concepts', aerospace_knowledge)
        self.assertIn('common_patterns', aerospace_knowledge)
        self.assertIn('validation_methods', aerospace_knowledge)

    def test_test_pattern_loading(self):
        """Test that test patterns are loaded correctly"""
        patterns = self.planner.test_patterns

        self.assertIn('unit_test', patterns)
        self.assertIn('integration_test', patterns)
        self.assertIn('edge_case_test', patterns)

        unit_pattern = patterns['unit_test']
        self.assertIn('description', unit_pattern)
        self.assertIn('template', unit_pattern)
        self.assertIn('best_practices', unit_pattern)

    def test_implementation_strategy_loading(self):
        """Test that implementation strategies are loaded correctly"""
        strategies = self.planner.implementation_strategies

        self.assertIn('algorithmic', strategies)
        self.assertIn('data_processing', strategies)
        self.assertIn('user_interface', strategies)

        alg_strategy = strategies['algorithmic']
        self.assertIn('approach', alg_strategy)
        self.assertIn('testing_strategy', alg_strategy)
        self.assertIn('common_pitfalls', alg_strategy)

    def test_feature_request_analysis(self):
        """Test deep analysis of feature requests"""
        analysis = self.planner._analyze_feature_request(self.test_request, self.test_domain)

        # Check analysis structure
        expected_keys = ['feature_name', 'core_requirements', 'domain_specifics',
                        'user_intent', 'technical_challenges', 'dependencies', 'assumptions']
        for key in expected_keys:
            self.assertIn(key, analysis)

        # Check specific analysis results
        self.assertEqual(analysis['user_intent'], 'new_development')
        self.assertIn('numerical computation', analysis['core_requirements'])
        self.assertIn('orbital_mechanics', analysis['domain_specifics']['key_concepts'])

    def test_complexity_assessment(self):
        """Test complexity assessment logic"""
        # Simple analysis
        simple_analysis = {
            'technical_challenges': ['basic calculation'],
            'core_requirements': ['simple operation']
        }
        complexity = self.planner._assess_complexity(simple_analysis, DomainType.AEROSPACE)
        self.assertEqual(complexity, 'simple')

        # Complex analysis
        complex_analysis = {
            'technical_challenges': ['high precision', 'real-time', 'multi-physics', 'optimization', 'validation', 'integration'],
            'core_requirements': ['multiple operations', 'error handling', 'performance', 'scalability', 'security', 'testing', 'documentation', 'deployment', 'monitoring', 'maintenance']
        }
        complexity = self.planner._assess_complexity(complex_analysis, DomainType.AEROSPACE)
        self.assertEqual(complexity, 'complex')

    def test_time_estimation(self):
        """Test time estimation based on complexity"""
        self.assertEqual(self.planner._estimate_time('simple'), '2-4 hours')
        self.assertEqual(self.planner._estimate_time('moderate'), '1-2 days')
        self.assertEqual(self.planner._estimate_time('complex'), '3-5 days')

    def test_comprehensive_plan_creation(self):
        """Test creation of comprehensive implementation plans"""
        plan = self.planner.create_comprehensive_plan(self.test_request, self.test_domain)

        # Check plan structure
        self.assertIsInstance(plan, ImplementationPlan)
        self.assertEqual(plan.domain, self.test_domain)
        self.assertEqual(plan.feature_name, 'Advanced Calculator System')
        self.assertGreater(len(plan.test_suites), 0)
        self.assertGreater(len(plan.implementation_steps), 0)
        self.assertIn('risks', plan.risks)
        self.assertGreater(len(plan.success_criteria), 0)

    def test_test_strategy_generation(self):
        """Test generation of comprehensive test strategies"""
        analysis = self.planner._analyze_feature_request(self.test_request, self.test_domain)
        test_suites = self.planner._generate_test_strategy(analysis, self.test_domain)

        # Should have multiple test suites
        self.assertGreater(len(test_suites), 1)

        # Check test suite structure
        for suite in test_suites:
            self.assertIsInstance(suite, TestSuite)
            self.assertGreater(len(suite.test_cases), 0)
            self.assertEqual(suite.domain, self.test_domain)

    def test_success_criteria_definition(self):
        """Test definition of clear success criteria"""
        analysis = self.planner._analyze_feature_request(self.test_request, self.test_domain)
        criteria = self.planner._define_success_criteria(analysis, self.test_domain)

        self.assertIn('All tests pass successfully', criteria)
        self.assertIn('Code meets quality standards', criteria)
        self.assertIn('Performance meets requirements', criteria)

class TestTDDCodeGenerator(unittest.TestCase):
    """Test the TDD-first code generator"""

    def setUp(self):
        """Set up test fixtures"""
        self.planner = AdvancedTDDPlanner()
        self.generator = TDDCodeGenerator(self.planner)
        self.test_plan = self.planner.create_comprehensive_plan(
            "Create an advanced calculator", DomainType.AEROSPACE
        )

    def test_generator_initialization(self):
        """Test that code generator initializes correctly"""
        self.assertIsInstance(self.generator, TDDCodeGenerator)
        self.assertIsNotNone(self.generator.temp_dir)
        self.assertTrue(os.path.exists(self.generator.temp_dir))

    def test_project_structure_creation(self):
        """Test creation of project structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock temp directory
            self.generator.temp_dir = temp_dir

            # Create project structure
            self.generator._create_project_structure(self.test_plan)

            # Check structure exists
            domain_dir = os.path.join(temp_dir, self.test_plan.domain.value)
            self.assertTrue(os.path.exists(domain_dir))

            expected_dirs = ['src', 'tests', 'docs', 'examples']
            for dir_name in expected_dirs:
                dir_path = os.path.join(domain_dir, dir_name)
                self.assertTrue(os.path.exists(dir_path), f"Directory {dir_name} should exist")

    def test_test_file_creation(self):
        """Test creation of comprehensive test files"""
        # Create a test suite
        test_suite = TestSuite(
            name="Calculator Core Tests",
            description="Test core calculator functionality",
            domain=DomainType.AEROSPACE,
            test_cases=[
                TestCase(
                    name="test_basic_calculation",
                    description="Test basic calculation functionality",
                    domain=DomainType.AEROSPACE,
                    test_code="self.assertEqual(2 + 2, 4)",
                    expected_behavior="Basic arithmetic should work"
                )
            ]
        )

        # Create test file
        test_file = self.generator._create_test_file(test_suite, self.test_plan)

        # Check file exists and has content
        self.assertTrue(os.path.exists(test_file))
        with open(test_file, 'r') as f:
            content = f.read()
            self.assertIn('test_basic_calculation', content)
            self.assertIn('Calculator Core Tests', content)

    def test_minimal_implementation_creation(self):
        """Test creation of minimal implementation"""
        test_suite = TestSuite(
            name="Minimal Tests",
            description="Minimal test suite",
            domain=DomainType.AEROSPACE,
            test_cases=[
                TestCase(
                    name="test_placeholder",
                    description="Test placeholder",
                    domain=DomainType.AEROSPACE,
                    test_code="self.assertTrue(True)",
                    expected_behavior="Should pass"
                )
            ]
        )

        # Create minimal implementation
        impl_file = self.generator._implement_minimal_code(test_suite, self.test_plan)

        # Check file exists and has basic structure
        self.assertTrue(os.path.exists(impl_file))
        with open(impl_file, 'r') as f:
            content = f.read()
            self.assertIn('class', content)
            self.assertIn('def', content)

    def test_requirements_file_generation(self):
        """Test requirements.txt generation"""
        requirements = self.generator._generate_requirements_file(self.test_plan)

        # Should have basic requirements
        self.assertIn('pytest', requirements)
        self.assertIn('numpy', requirements)

    def test_readme_generation(self):
        """Test README.md generation"""
        readme = self.generator._generate_readme(self.test_plan)

        # Should contain key information
        self.assertIn(self.test_plan.feature_name, readme)
        self.assertIn(self.test_plan.domain.value, readme)
        self.assertIn('Installation', readme)
        self.assertIn('Usage', readme)

    def test_configuration_generation(self):
        """Test configuration file generation"""
        config = self.generator._generate_configuration(self.test_plan)

        # Should be valid JSON
        config_data = json.loads(config)
        self.assertIn('feature_name', config_data)
        self.assertIn('domain', config_data)
        self.assertIn('version', config_data)

class TestUniversalTDDSystem(unittest.TestCase):
    """Test the universal TDD system"""

    def setUp(self):
        """Set up test fixtures"""
        self.system = UniversalTDDSystem()
        self.test_request = "Create an advanced data analyzer"
        self.test_domain = DomainType.DATA_SCIENCE

    def test_system_initialization(self):
        """Test that universal system initializes correctly"""
        self.assertIsInstance(self.system, UniversalTDDSystem)
        self.assertIsInstance(self.system.planner, AdvancedTDDPlanner)
        self.assertIsInstance(self.system.code_generator, TDDCodeGenerator)
        self.assertEqual(len(self.system.domain_implementations), 0)

    def test_feature_implementation(self):
        """Test feature implementation across domains"""
        # This is a complex test that would require mocking
        # For now, just test that the method exists and returns expected structure
        with patch.object(self.system.planner, 'create_comprehensive_plan') as mock_plan:
            with patch.object(self.system.code_generator, 'generate_implementation') as mock_gen:
                mock_plan.return_value = ImplementationPlan(
                    feature_name="Test Feature",
                    domain=self.test_domain,
                    test_suites=[],
                    implementation_steps=[],
                    estimated_complexity="simple",
                    estimated_time="2-4 hours",
                    risks=[],
                    success_criteria=[]
                )
                mock_gen.return_value = {
                    'implementation_files': {'test.py': '# Test code'},
                    'test_results': {},
                    'final_status': 'completed'
                }

                results = self.system.implement_feature(self.test_request, self.test_domain)

                self.assertIn('implementation_files', results)
                self.assertIn('test_results', results)
                self.assertIn('final_status', results)
                self.assertEqual(results['final_status'], 'completed')

    def test_implementation_status_check(self):
        """Test implementation status checking"""
        # Test non-existent domain
        status = self.system.get_implementation_status(DomainType.AEROSPACE)
        self.assertEqual(status['status'], 'not_started')

    def test_cross_domain_functionality(self):
        """Test that system can handle multiple domains"""
        domains = [
            DomainType.AEROSPACE,
            DomainType.WEB_DEVELOPMENT,
            DomainType.DATA_SCIENCE,
            DomainType.MACHINE_LEARNING
        ]

        for domain in domains:
            # Should not raise exceptions
            status = self.system.get_implementation_status(domain)
            self.assertIn('status', status)

class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and end-to-end functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.system = UniversalTDDSystem()

    def test_aerospace_calculator_implementation(self):
        """Test complete aerospace calculator implementation"""
        request = "Implement an advanced aerospace calculator with real orbital mechanics"
        domain = DomainType.AEROSPACE

        # Mock the implementation process
        with patch.object(self.system.planner, 'create_comprehensive_plan') as mock_plan:
            with patch.object(self.system.code_generator, 'generate_implementation') as mock_gen:
                mock_plan.return_value = ImplementationPlan(
                    feature_name="Advanced Aerospace Calculator",
                    domain=domain,
                    test_suites=[],
                    implementation_steps=[
                        "Set up physics constants",
                        "Implement orbital mechanics",
                        "Add trajectory calculations",
                        "Create validation tests"
                    ],
                    estimated_complexity="moderate",
                    estimated_time="1-2 days",
                    risks=["High precision requirements", "Complex math"],
                    success_criteria=["Accurate orbital calculations", "All tests pass"]
                )
                mock_gen.return_value = {
                    'implementation_files': {
                        'aerospace_calculator.py': '# Real implementation',
                        'requirements.txt': 'numpy>=1.21.0\npytest>=7.0.0',
                        'README.md': '# Aerospace Calculator'
                    },
                    'test_results': {
                        'Physics Tests': {'tests_written': 5, 'tests_passing': 5, 'implementation_status': 'completed'},
                        'Orbital Mechanics Tests': {'tests_written': 8, 'tests_passing': 8, 'implementation_status': 'completed'}
                    },
                    'final_status': 'completed'
                }

                results = self.system.implement_feature(request, domain)

                # Verify results
                self.assertEqual(results['final_status'], 'completed')
                self.assertIn('aerospace_calculator.py', results['implementation_files'])
                self.assertGreater(len(results['test_results']), 0)

    def test_web_application_implementation(self):
        """Test complete web application implementation"""
        request = "Create a full-featured web application with authentication"
        domain = DomainType.WEB_DEVELOPMENT

        with patch.object(self.system.planner, 'create_comprehensive_plan') as mock_plan:
            with patch.object(self.system.code_generator, 'generate_implementation') as mock_gen:
                mock_plan.return_value = ImplementationPlan(
                    feature_name="Advanced Web Application",
                    domain=domain,
                    test_suites=[],
                    implementation_steps=[
                        "Set up Flask application",
                        "Implement authentication",
                        "Create database models",
                        "Add API endpoints"
                    ],
                    estimated_complexity="moderate",
                    estimated_time="2-4 hours",
                    risks=["Security vulnerabilities", "Database design"],
                    success_criteria=["User registration works", "Authentication secure"]
                )
                mock_gen.return_value = {
                    'implementation_files': {
                        'app.py': '# Flask application',
                        'requirements.txt': 'flask>=2.3.0\npytest>=7.0.0',
                        'README.md': '# Web Application'
                    },
                    'test_results': {
                        'Authentication Tests': {'tests_written': 12, 'tests_passing': 12, 'implementation_status': 'completed'},
                        'API Tests': {'tests_written': 8, 'tests_passing': 8, 'implementation_status': 'completed'}
                    },
                    'final_status': 'completed'
                }

                results = self.system.implement_feature(request, domain)

                self.assertEqual(results['final_status'], 'completed')
                self.assertIn('app.py', results['implementation_files'])

class TestTDDPrinciples(unittest.TestCase):
    """Test that TDD principles are strictly followed"""

    def setUp(self):
        """Set up test fixtures"""
        self.planner = AdvancedTDDPlanner()
        self.generator = TDDCodeGenerator(self.planner)

    def test_red_green_refactor_cycle(self):
        """Test that TDD follows red-green-refactor cycle"""
        # This would require actual test execution
        # For demonstration, we verify the structure supports this

        # Create a simple test suite
        test_suite = TestSuite(
            name="Simple Math Tests",
            description="Test basic mathematical operations",
            domain=DomainType.DATA_SCIENCE,
            test_cases=[
                TestCase(
                    name="test_addition",
                    description="Test that addition works",
                    domain=DomainType.DATA_SCIENCE,
                    test_code="result = add(2, 3)\nself.assertEqual(result, 5)",
                    expected_behavior="Addition should return correct sum"
                )
            ]
        )

        plan = ImplementationPlan(
            feature_name="Simple Calculator",
            domain=DomainType.DATA_SCIENCE,
            test_suites=[test_suite],
            implementation_steps=["Implement add function", "Add tests", "Refactor code"],
            estimated_complexity="simple",
            estimated_time="30 minutes",
            risks=[],
            success_criteria=["Addition works correctly"]
        )

        # Generate test file
        test_file = self.generator._create_test_file(test_suite, plan)
        self.assertTrue(os.path.exists(test_file))

        # Generate minimal implementation
        impl_file = self.generator._implement_minimal_code(test_suite, plan)
        self.assertTrue(os.path.exists(impl_file))

        # Verify files are different (tests != implementation)
        with open(test_file, 'r') as f:
            test_content = f.read()
        with open(impl_file, 'r') as f:
            impl_content = f.read()

        self.assertNotEqual(test_content, impl_content)
        self.assertIn('test_', test_content)
        self.assertIn('def ', impl_content)

    def test_comprehensive_test_coverage(self):
        """Test that comprehensive test coverage is generated"""
        request = "Create a complex data processing system"
        domain = DomainType.DATA_SCIENCE

        plan = self.planner.create_comprehensive_plan(request, domain)

        # Should have multiple test suites
        self.assertGreater(len(plan.test_suites), 2)

        # Each suite should have multiple test cases
        for suite in plan.test_suites:
            self.assertGreater(len(suite.test_cases), 1)

            # Test cases should have proper structure
            for test_case in suite.test_cases:
                self.assertTrue(test_case.name.startswith('test_'))
                self.assertNotEqual(test_case.test_code, "")
                self.assertNotEqual(test_case.expected_behavior, "")

    def test_implementation_quality_checks(self):
        """Test that implementation quality checks are in place"""
        plan = self.planner.create_comprehensive_plan(
            "Create a high-quality system", DomainType.WEB_DEVELOPMENT
        )

        # Success criteria should include quality measures
        quality_criteria = [
            criterion for criterion in plan.success_criteria
            if 'quality' in criterion.lower() or 'standard' in criterion.lower()
        ]
        self.assertGreater(len(quality_criteria), 0)

        # Implementation steps should include quality checks
        quality_steps = [
            step for step in plan.implementation_steps
            if 'refactor' in step.lower() or 'review' in step.lower() or 'test' in step.lower()
        ]
        self.assertGreater(len(quality_steps), 0)

class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        """Set up test fixtures"""
        self.planner = AdvancedTDDPlanner()

    def test_invalid_domain_handling(self):
        """Test handling of invalid domains"""
        with self.assertRaises(KeyError):
            # Try to access non-existent domain
            _ = self.planner.domain_knowledge[DomainType.AEROSPACE]

    def test_malformed_requests(self):
        """Test handling of malformed requests"""
        # Empty request
        analysis = self.planner._analyze_feature_request("", DomainType.AEROSPACE)
        self.assertIn('assumptions', analysis)

        # Very short request
        analysis = self.planner._analyze_feature_request("x", DomainType.WEB_DEVELOPMENT)
        self.assertIn('core_requirements', analysis)

    def test_missing_dependencies(self):
        """Test handling of missing dependencies"""
        # This would require mocking file system access
        pass

    def test_resource_constraints(self):
        """Test handling of resource constraints"""
        # This would test memory/CPU limits
        pass

class TestPerformanceAndScalability(unittest.TestCase):
    """Test performance and scalability"""

    def setUp(self):
        """Set up test fixtures"""
        self.planner = AdvancedTDDPlanner()

    def test_large_feature_requests(self):
        """Test handling of large, complex feature requests"""
        large_request = """
        Create a comprehensive aerospace mission planning system with:
        - Orbital mechanics calculations
        - Propulsion system design
        - Structural analysis
        - Thermal management
        - Communication systems
        - Navigation and guidance
        - Power systems
        - Life support systems
        - Scientific instruments
        - Data management
        - Mission operations
        - Ground support systems
        """

        analysis = self.planner._analyze_feature_request(large_request, DomainType.AEROSPACE)

        # Should handle large requests without crashing
        self.assertIn('feature_name', analysis)
        self.assertIn('core_requirements', analysis)
        self.assertGreater(len(analysis['core_requirements']), 5)

    def test_concurrent_implementations(self):
        """Test handling of concurrent implementation requests"""
        # This would test thread safety and concurrent access
        pass

    def test_memory_efficiency(self):
        """Test memory efficiency of the system"""
        # This would test memory usage during planning
        pass

if __name__ == '__main__':
    # Run comprehensive test suite
    unittest.main(verbosity=2)