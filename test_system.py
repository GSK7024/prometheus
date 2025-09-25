#!/usr/bin/env python3
"""
Simple test script for Enhanced Prometheus AI
Run this after installation to verify everything works
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all core modules can be imported"""
    print("🔍 Testing imports...")

    try:
        # Test basic Python modules
        import math
        import json
        import os
        import sys
        import time
        print("  ✅ Basic Python modules: OK")

        # Test data science modules
        import numpy as np
        print(f"  ✅ NumPy: {np.__version__}")

        # Test machine learning modules
        import torch
        print(f"  ✅ PyTorch: {torch.__version__}")

        # Test NLP modules
        from transformers import AutoTokenizer, AutoModel
        print("  ✅ Transformers: OK")

        # Test our main module
        from prometheus import (
            QuantumCognitiveCore,
            AerospaceEngineeringModule,
            AdvancedTestingFramework,
            EvolutionEngine
        )
        print("  ✅ Prometheus core modules: OK")

        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of core components"""
    print("\n🔍 Testing basic functionality...")

    try:
        from prometheus import AerospaceEngineeringModule, AdvancedTestingFramework

        # Test aerospace calculations
        aerospace = AerospaceEngineeringModule()
        result = aerospace.orbital_mechanics.calculate_delta_v(
            payload_mass=1000,
            target_orbit={"type": "LEO", "altitude": 400000}
        )
        print(f"  ✅ Orbital mechanics calculation: {result} m/s")

        # Test testing framework
        testing = AdvancedTestingFramework()
        test_config = testing.generate_comprehensive_test_suite(
            code="print('hello')",
            language="python",
            requirements=["unit_test"]
        )
        print(f"  ✅ Testing framework: Generated {len(test_config['test_suite'])} test types")

        return True
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False

def test_multi_language_configs():
    """Test multi-language configuration support"""
    print("\n🔍 Testing multi-language support...")

    try:
        from prometheus import Config
        config = Config()

        languages = list(config.language_configs.keys())
        print(f"  ✅ Total supported languages: {len(languages)}")

        # Check key languages
        key_languages = {
            "python": "High-level scripting",
            "javascript": "Web development",
            "java": "Enterprise applications",
            "rust": "Systems programming",
            "go": "Cloud services",
            "cpp": "High-performance computing",
            "fortran": "Scientific computing",
            "verilog": "Hardware design"
        }

        found_languages = []
        for lang, description in key_languages.items():
            if lang in languages:
                found_languages.append(f"{lang} ({description})")
            else:
                print(f"  ⚠️  {lang} not configured")

        print(f"  ✅ Found {len(found_languages)} key languages:")
        for lang in found_languages:
            print(f"    - {lang}")

        return True
    except Exception as e:
        print(f"  ❌ Multi-language test failed: {e}")
        return False

def test_quantum_core():
    """Test quantum-inspired cognitive core"""
    print("\n🔍 Testing quantum cognitive core...")

    try:
        from prometheus import QuantumCognitiveCore

        # Create quantum core
        quantum_core = QuantumCognitiveCore(
            input_dim=128,
            hidden_dim=256,
            output_dim=64,
            num_qubits=8,
            language_support=True
        )

        print("  ✅ Quantum core initialized with 8 qubits")
        print(f"    - Input dimension: {quantum_core.input_dim}")
        print(f"    - Hidden dimension: {quantum_core.hidden_dim}")
        print(f"    - Output dimension: {quantum_core.output_dim}")
        print(f"    - Language support: {quantum_core.language_support}")

        return True
    except Exception as e:
        print(f"  ❌ Quantum core test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Enhanced Prometheus AI - System Test Suite")
    print("=" * 60)
    print("Testing the most advanced AI coding system ever created!")
    print()

    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Multi-language Support", test_multi_language_configs),
        ("Quantum Cognitive Core", test_quantum_core)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print("
" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your Enhanced Prometheus AI system is ready to use!")
        print()
        print("🚀 Try these example commands:")
        print("python prometheus.py --goal 'Create a web app' --project my_app --agent app.py")
        print("python prometheus.py --goal 'Design a rocket system' --project rocket --agent rocket.py")
        print("python mars_mission_final_demo.py")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("Try running: pip install -r requirements.txt")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)