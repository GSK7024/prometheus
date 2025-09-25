#!/usr/bin/env python3
"""
Enhanced Prometheus AI - Startup Script
This script checks your system and helps you get started
"""

import sys
import os
import importlib.util
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🔍 Python Version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required!")
        return False
    print("✅ Python version OK")
    return True

def check_files():
    """Check that all required files exist"""
    required_files = [
        "prometheus.py",
        "setup.py",
        "requirements.txt",
        "README.md",
        "test_system.py",
        "mars_mission_final_demo.py"
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print(f"❌ Missing files: {', '.join(missing)}")
        return False

    print("✅ All required files present")
    return True

def check_directories():
    """Check that required directories exist or create them"""
    required_dirs = ["logs", "memory", "sessions", "sandbox", "tests"]

    for dir_name in required_dirs:
        Path(dir_name).mkdir(exist_ok=True)

    print("✅ Required directories ready")
    return True

def test_imports():
    """Test basic imports"""
    try:
        import torch
        import numpy as np
        import transformers

        print(f"✅ Core libraries: PyTorch {torch.__version__}, NumPy {np.__version__}")
        return True
    except ImportError as e:
        print(f"⚠️  Some libraries missing: {e}")
        return False

def main():
    """Main startup function"""
    print("🚀 Enhanced Prometheus AI - System Check")
    print("=" * 50)
    print("Checking your system for the most advanced AI coding system ever created!")
    print()

    checks = [
        ("Python Version", check_python_version),
        ("Required Files", check_files),
        ("Directories", check_directories),
        ("Core Libraries", test_imports)
    ]

    all_passed = True
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}:")
        if not check_func():
            all_passed = False

    print("\n" + "=" * 50)

    if all_passed:
        print("🎉 SYSTEM CHECK PASSED!")
        print("✅ Your system is ready for Enhanced Prometheus AI!")
        print()
        print("🚀 Next steps:")
        print("1. Install dependencies: python setup.py")
        print("2. Run system tests: python test_system.py")
        print("3. Try the demo: python mars_mission_final_demo.py")
        print("4. Launch main system: python prometheus_launcher.py --help")
        print()
        print("📚 For detailed instructions: cat TESTING_GUIDE.md")
    else:
        print("⚠️  Some checks failed. Please review the errors above.")
        print("📚 See TESTING_GUIDE.md for troubleshooting help.")

    print("\n" + "=" * 50)
    print("Enhanced Prometheus AI Features:")
    print("✓ NASA-level aerospace engineering")
    print("✓ 50+ programming language support")
    print("✓ Quantum-inspired algorithms")
    print("✓ Advanced testing framework")
    print("✓ Continuous evolution system")
    print("✓ Multi-language code generation")
    print("✓ Real-time optimization")
    print("\n🎯 Ready to revolutionize your development workflow!")

if __name__ == "__main__":
    main()