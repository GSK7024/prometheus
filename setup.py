#!/usr/bin/env python3
"""
Setup script for Enhanced Prometheus AI
This will install all required dependencies and set up the environment
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"🔧 {description}")
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\n🚀 Installing Python dependencies...")

    # Install core dependencies
    if not run_command("pip install -r requirements.txt", "Installing core dependencies"):
        return False

    # Install optional dependencies
    print("\n🔧 Installing optional dependencies...")
    run_command("pip install docker kubernetes", "Installing optional cloud dependencies (may fail if not needed)")

    return True

def setup_environment():
    """Set up environment variables and directories"""
    print("\n📁 Setting up environment...")

    # Create necessary directories
    dirs = ["logs", "memory", "sessions", "sandbox", "tests"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

    # Create .env file
    env_content = """# Enhanced Prometheus AI Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
MEMORY_DIR=./memory
SESSION_DIR=./sessions
SANDBOX_BASE_PATH=./sandbox
MAX_TOKENS=8192
REQUEST_TIMEOUT=300.0
METAMORPHOSIS_THRESHOLD=2
MAX_DEBUG_ATTEMPTS=8

# API Keys (add your own)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here

# Database
DATABASE_URL=sqlite:///./prometheus.db

# Security
ENCRYPTION_KEY=your_encryption_key_here

# Performance
ENABLE_CUDA=false
MAX_WORKERS=4
CACHE_SIZE=1000
"""

    with open(".env", "w") as f:
        f.write(env_content)
    print("✅ Created .env configuration file")

    return True

def run_health_check():
    """Run basic health check"""
    print("\n🩺 Running health check...")

    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")

        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")

        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")

        # Test basic functionality
        import prometheus
        print("✅ Prometheus AI module imported successfully")

        return True
    except ImportError as e:
        print(f"❌ Health check failed: {e}")
        return False

def create_test_script():
    """Create a simple test script"""
    test_script = '''#!/usr/bin/env python3
"""
Simple test script for Enhanced Prometheus AI
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    try:
        from prometheus import (
            UltraCognitiveForgeOrchestrator,
            AerospaceEngineeringModule,
            AdvancedTestingFramework,
            EvolutionEngine,
            QuantumCognitiveCore
        )
        print("✅ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        # Test aerospace engineering
        from prometheus import AerospaceEngineeringModule
        aerospace = AerospaceEngineeringModule()

        result = aerospace.orbital_mechanics.calculate_delta_v(
            payload_mass=1000,
            target_orbit={"type": "LEO", "altitude": 400000}
        )
        print(f"✅ Aerospace calculation: {result} m/s")

        # Test testing framework
        from prometheus import AdvancedTestingFramework
        testing = AdvancedTestingFramework()
        print("✅ Testing framework initialized")

        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_multi_language_support():
    """Test multi-language configuration"""
    try:
        from prometheus import Config
        config = Config()

        # Check if we have comprehensive language configs
        languages = list(config.language_configs.keys())
        print(f"✅ Supported languages: {len(languages)}")

        # Check for key languages
        key_languages = ["python", "javascript", "java", "rust", "go", "cpp"]
        for lang in key_languages:
            if lang in languages:
                print(f"  ✅ {lang}")
            else:
                print(f"  ⚠️  {lang} (not configured)")

        return True
    except Exception as e:
        print(f"❌ Multi-language test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Enhanced Prometheus AI - Test Suite")
    print("=" * 50)

    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Multi-language Support", test_multi_language_support)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 Try running:")
        print("python prometheus.py --goal 'Create a web application' --project my_app --agent app.py")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
'''

    with open("test_system.py", "w") as f:
        f.write(test_script)

    print("✅ Created test_system.py")

def main():
    """Main setup function"""
    print("🚀 Enhanced Prometheus AI Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        return False

    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        return False

    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed")
        return False

    # Create test script
    create_test_script()

    # Health check
    if not run_health_check():
        print("⚠️  Health check failed, but setup may still be usable")
    else:
        print("✅ Health check passed")

    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python test_system.py")
    print("3. Try the demo: python mars_mission_final_demo.py")
    print("4. Run main system: python prometheus.py --goal 'Your task' --project my_project")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)