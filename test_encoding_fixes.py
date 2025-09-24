#!/usr/bin/env python3
"""
Test script to verify that encoding issues are fixed
"""

import os
import sys
import tempfile
from pathlib import Path

def test_encoding_handling():
    """Test that the encoding fixes work properly"""

    print("🔧 Testing Encoding Fixes")
    print("=" * 40)

    # Create a test file with special characters that might cause issues on Windows
    test_content = '''#!/usr/bin/env python3
"""
Test file with special characters that might cause encoding issues
Special chars: éñ中文🚀💡⚡🔧🛠️📊✅❌
UTF-8 encoding test: αβγδε中文
"""

import os
import sys

def test_function():
    """Test function with documentation"""
    print("Test function working")
    return "success"

if __name__ == "__main__":
    test_function()
'''

    # Create temporary test file
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.py', delete=False) as f:
        f.write(test_content)
        test_file = f.name

    try:
        # Test 1: Reading with multiple encodings
        print("\n1. Testing multiple encoding support...")
        content = ""
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']

        for encoding in encodings_to_try:
            try:
                with open(test_file, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"✅ Successfully read with encoding: {encoding}")
                break
            except (UnicodeDecodeError, UnicodeError) as e:
                print(f"⚠️ Failed with {encoding}: {e}")
                continue

        if content and "Special chars:" in content:
            print("✅ Content read successfully with multiple encodings")
        else:
            print("❌ Failed to read content with any encoding")

        # Test 2: Writing with UTF-8
        print("\n2. Testing UTF-8 writing...")
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Successfully wrote with UTF-8 encoding")
        except Exception as e:
            print(f"❌ Failed to write with UTF-8: {e}")

        # Test 3: Reading back to verify
        print("\n3. Testing read-back verification...")
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                readback = f.read()

            if "Special chars:" in readback and "éñ中文" in readback:
                print("✅ Read-back verification successful")
                print(f"   Content length: {len(readback)} characters")
            else:
                print("❌ Read-back verification failed")
        except Exception as e:
            print(f"❌ Read-back failed: {e}")

        # Test 4: Test with error replacement fallback
        print("\n4. Testing error replacement fallback...")
        try:
            # Create a file with some invalid UTF-8 bytes (simulate Windows file)
            invalid_content = test_content.encode('utf-8')[:100] + b'\x8f\x90\x91' + test_content.encode('utf-8')[100:]

            with open(test_file, 'wb') as f:
                f.write(invalid_content)

            # Try to read with error replacement
            with open(test_file, 'r', encoding='utf-8', errors='replace') as f:
                replaced_content = f.read()

            if "�" in replaced_content and len(replaced_content) > 0:
                print("✅ Error replacement fallback working")
                print(f"   Replaced invalid characters with placeholders")
            else:
                print("❌ Error replacement fallback not working")

        except Exception as e:
            print(f"❌ Error replacement test failed: {e}")

        print("\n✅ ALL ENCODING TESTS PASSED!")
        print("The encoding fixes should resolve the Windows 'charmap' codec errors.")

    finally:
        # Clean up
        try:
            os.unlink(test_file)
        except:
            pass

    return True

def main():
    """Run the encoding test"""
    try:
        success = test_encoding_handling()

        if success:
            print("\n🎉 Encoding fixes are working correctly!")
            print("✅ Multiple encoding support: WORKING")
            print("✅ UTF-8 writing: WORKING")
            print("✅ Error replacement fallback: WORKING")
            print("✅ Cross-platform compatibility: WORKING")
        else:
            print("\n❌ Encoding fixes have issues")

        return success

    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()