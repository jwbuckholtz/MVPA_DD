#!/usr/bin/env python3
"""
Demo Script for Testing System
==============================

This script demonstrates how to use the new pytest-based testing system
for the MVPA pipeline.

Author: Cognitive Neuroscience Lab, Stanford University
"""

import subprocess
import sys
from pathlib import Path


def run_demo():
    """Run a demonstration of the testing system"""
    print("MVPA Pipeline Testing System Demo")
    print("=" * 50)
    
    # Check if pytest is available
    try:
        import pytest
        print("✓ pytest is available")
    except ImportError:
        print("✗ pytest is not available")
        print("Please install: pip install -r requirements.txt")
        return False
    
    # Show available test files
    print("\nAvailable Test Files:")
    test_files = [
        "test_pipeline_pytest.py",
        "test_pipeline.py",
        "run_tests.py",
        "pytest.ini"
    ]
    
    for file in test_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
    
    # Show testing options
    print("\nTesting Options:")
    print("  python run_tests.py                  # Run all tests")
    print("  python run_tests.py --behavioral     # Run behavioral tests")
    print("  python run_tests.py --mvpa           # Run MVPA tests")
    print("  python run_tests.py --geometry       # Run geometry tests")
    print("  python run_tests.py --unit           # Run unit tests")
    print("  python run_tests.py --integration    # Run integration tests")
    print("  python run_tests.py --fast           # Run fast tests only")
    print("  python run_tests.py --coverage       # Run with coverage")
    print("  python run_tests.py --html           # Generate HTML coverage")
    print("  python run_tests.py --parallel       # Run in parallel")
    print("  python run_tests.py --help           # Show all options")
    
    # Run a simple test demonstration
    print("\n" + "=" * 50)
    print("Running Quick Test Demo...")
    print("=" * 50)
    
    try:
        # Run pytest with collection only (no actual test execution)
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'test_pipeline_pytest.py', 
            '--collect-only', 
            '--quiet'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Count number of tests collected
            lines = result.stdout.split('\n')
            test_count = 0
            for line in lines:
                if '::test_' in line:
                    test_count += 1
            
            print(f"✓ Test collection successful")
            print(f"  Found {test_count} tests")
            print(f"  Tests are organized in categories:")
            print(f"    - Analysis Base Classes")
            print(f"    - Behavioral Analysis")
            print(f"    - MVPA Analysis")
            print(f"    - Geometry Analysis")
            print(f"    - Mask Creation")
            print(f"    - Data Loading")
            print(f"    - Error Handling")
            print(f"    - Integration Tests")
            print(f"    - Performance Tests")
            
        else:
            print(f"✗ Test collection failed")
            print(f"  Error: {result.stderr}")
            
    except Exception as e:
        print(f"✗ Error running test demo: {e}")
    
    # Show example test execution
    print("\n" + "=" * 50)
    print("Example Test Execution:")
    print("=" * 50)
    
    example_commands = [
        "# Run just the behavioral analysis tests",
        "python run_tests.py --behavioral",
        "",
        "# Run a specific test class",
        "python run_tests.py --class TestBehavioralAnalysis",
        "",
        "# Run a specific test method",
        "python run_tests.py --method test_hyperbolic_discount_function",
        "",
        "# Run tests with coverage and generate HTML report",
        "python run_tests.py --coverage --html",
        "",
        "# Run tests in parallel for faster execution",
        "python run_tests.py --parallel 4",
        "",
        "# Run only fast tests (exclude slow ones)",
        "python run_tests.py --fast",
        "",
        "# Run with verbose output for debugging",
        "python run_tests.py --verbose",
    ]
    
    for cmd in example_commands:
        print(cmd)
    
    # Show test structure
    print("\n" + "=" * 50)
    print("Test Structure:")
    print("=" * 50)
    
    test_structure = """
    test_pipeline_pytest.py
    ├── TestAnalysisBase          # Base classes and factory
    ├── TestBehavioralAnalysis    # Behavioral modeling
    ├── TestMVPAAnalysis          # Pattern extraction & decoding
    ├── TestGeometryAnalysis      # Neural geometry
    ├── TestMaskCreation          # Mask loading & validation
    ├── TestDataLoading           # Data loading utilities
    ├── TestErrorHandling         # Error conditions
    ├── TestIntegration           # End-to-end workflows
    └── TestPerformance           # Performance benchmarks
    """
    
    print(test_structure)
    
    print("\n" + "=" * 50)
    print("Next Steps:")
    print("=" * 50)
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run quick test: python run_tests.py --fast")
    print("3. Run full test suite: python run_tests.py")
    print("4. Generate coverage report: python run_tests.py --html")
    print("5. Read the testing guide: TESTING_GUIDE.md")
    
    return True


if __name__ == '__main__':
    success = run_demo()
    sys.exit(0 if success else 1) 