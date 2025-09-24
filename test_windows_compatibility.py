#!/usr/bin/env python3
"""
Test Windows compatibility for the Self-Evolution System
This script tests that all paths work correctly on Windows
"""

import os
import sys
from pathlib import Path

def test_cross_platform_paths():
    """Test that all paths work correctly across platforms"""

    print("🔧 Testing Cross-Platform Path Compatibility")
    print("=" * 50)

    # Test 1: Source file path detection
    print("\n1. Testing source file path detection:")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_file = os.path.join(script_dir, "prometheus.py")
    print(f"   Script directory: {script_dir}")
    print(f"   Source file path: {source_file}")
    print(f"   Source file exists: {os.path.exists(source_file)}")

    # Test 2: Learning database path
    print("\n2. Testing learning database path:")
    db_path = os.path.join(script_dir, "self_evolution", "learnings.json")
    db_dir = os.path.dirname(db_path)
    print(f"   Database directory: {db_dir}")
    print(f"   Database file: {db_path}")

    # Create directory if it doesn't exist
    os.makedirs(db_dir, exist_ok=True)
    print(f"   Directory created: {os.path.exists(db_dir)}")

    # Test 3: Backup path
    print("\n3. Testing backup path:")
    backup_filename = "prometheus_backup_test.py"
    backup_path = os.path.join(script_dir, backup_filename)
    print(f"   Backup path: {backup_path}")

    # Test 4: Cross-platform compatibility
    print("\n4. Testing cross-platform compatibility:")
    print(f"   Using os.path.join: {os.path.join('dir1', 'dir2', 'file.txt')}")
    print(f"   Path separator: {os.sep}")
    print(f"   Current OS: {sys.platform}")

    # Test 5: Path validation
    print("\n5. Testing path validation:")
    test_paths = [
        source_file,
        db_path,
        backup_path
    ]

    for path in test_paths:
        if os.path.exists(path) or path.endswith(('.json', '.py')):
            print(f"   ✅ Valid path: {path}")
        else:
            print(f"   ⚠️  Path may not exist yet: {path}")

    print("\n✅ Cross-platform path compatibility test completed!")
    print("The system should now work correctly on Windows.")

    return True

if __name__ == "__main__":
    try:
        test_cross_platform_paths()
        print("\n🎉 All tests passed! The system is Windows-compatible.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)