#!/usr/bin/env python3
"""
Test Launcher for MVPA Pipeline
==============================

This script provides convenient ways to run the comprehensive test suite
for the MVPA pipeline with different configurations and options.

Usage:
    python run_tests.py [OPTIONS]

Examples:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --behavioral       # Run only behavioral tests
    python run_tests.py --mvpa             # Run only MVPA tests
    python run_tests.py --geometry         # Run only geometry tests
    python run_tests.py --fast             # Run fast tests only
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --parallel         # Run tests in parallel
    python run_tests.py --verbose          # Run with verbose output
    python run_tests.py --html             # Generate HTML coverage report
    python run_tests.py --class TestBehavioralAnalysis  # Run specific test class
    python run_tests.py --help             # Show help

Author: Cognitive Neuroscience Lab, Stanford University
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, verbose=False):
    """Run a command and return the result"""
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if verbose or result.returncode != 0:
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run MVPA Pipeline Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      # Run all tests
  %(prog)s --unit               # Run only unit tests
  %(prog)s --behavioral         # Run only behavioral tests
  %(prog)s --mvpa               # Run only MVPA tests
  %(prog)s --geometry           # Run only geometry tests
  %(prog)s --fast               # Run fast tests only
  %(prog)s --coverage           # Run with coverage report
  %(prog)s --parallel           # Run tests in parallel
  %(prog)s --html               # Generate HTML coverage report
  %(prog)s --class TestBehavioralAnalysis  # Run specific test class
        """
    )
    
    # Test selection options
    parser.add_argument(
        '--unit', action='store_true',
        help='Run only unit tests'
    )
    parser.add_argument(
        '--integration', action='store_true',
        help='Run only integration tests'
    )
    parser.add_argument(
        '--behavioral', action='store_true',
        help='Run only behavioral analysis tests'
    )
    parser.add_argument(
        '--mvpa', action='store_true',
        help='Run only MVPA analysis tests'
    )
    parser.add_argument(
        '--geometry', action='store_true',
        help='Run only geometry analysis tests'
    )
    parser.add_argument(
        '--factory', action='store_true',
        help='Run only analysis factory tests'
    )
    parser.add_argument(
        '--error', action='store_true',
        help='Run only error handling tests'
    )
    parser.add_argument(
        '--performance', action='store_true',
        help='Run only performance tests'
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='Run fast tests only (exclude slow tests)'
    )
    parser.add_argument(
        '--slow', action='store_true',
        help='Run slow tests only'
    )
    parser.add_argument(
        '--class', dest='test_class',
        help='Run specific test class (e.g., TestBehavioralAnalysis)'
    )
    parser.add_argument(
        '--method', dest='test_method',
        help='Run specific test method (e.g., test_hyperbolic_discount_function)'
    )
    
    # Output options
    parser.add_argument(
        '--coverage', action='store_true',
        help='Run with coverage report'
    )
    parser.add_argument(
        '--html', action='store_true',
        help='Generate HTML coverage report'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Quiet output'
    )
    parser.add_argument(
        '--parallel', '-n', type=int, metavar='N',
        help='Run tests in parallel with N processes'
    )
    parser.add_argument(
        '--no-cov', action='store_true',
        help='Disable coverage reporting'
    )
    parser.add_argument(
        '--tb', choices=['short', 'long', 'auto', 'no'],
        default='short',
        help='Traceback format'
    )
    
    # File options
    parser.add_argument(
        '--file', '-f',
        help='Run tests from specific file'
    )
    parser.add_argument(
        '--old-tests', action='store_true',
        help='Run the old test_pipeline.py instead of pytest suite'
    )
    
    args = parser.parse_args()
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("Error: pytest is not installed.")
        print("Please install testing dependencies: pip install -r requirements.txt")
        return 1
    
    # Handle old tests
    if args.old_tests:
        print("Running old test_pipeline.py...")
        cmd = [sys.executable, 'test_pipeline.py']
        success = run_command(cmd, verbose=args.verbose)
        return 0 if success else 1
    
    # Build pytest command
    cmd = [sys.executable, '-m', 'pytest']
    
    # Add test file
    if args.file:
        cmd.append(args.file)
    else:
        cmd.append('test_pipeline_pytest.py')
    
    # Add markers for test selection
    markers = []
    if args.unit:
        markers.append('unit')
    if args.integration:
        markers.append('integration')
    if args.behavioral:
        markers.append('behavioral')
    if args.mvpa:
        markers.append('mvpa')
    if args.geometry:
        markers.append('geometry')
    if args.factory:
        markers.append('factory')
    if args.error:
        markers.append('error')
    if args.performance:
        markers.append('performance')
    if args.fast:
        markers.append('not slow')
    if args.slow:
        markers.append('slow')
    
    if markers:
        cmd.extend(['-m', ' and '.join(markers)])
    
    # Add specific test class or method
    if args.test_class:
        cmd.append(f'::{args.test_class}')
        if args.test_method:
            cmd.append(f'::{args.test_method}')
    elif args.test_method:
        cmd.append(f'::{args.test_method}')
    
    # Add output options
    if args.verbose:
        cmd.append('-v')
    if args.quiet:
        cmd.append('-q')
    
    cmd.extend(['--tb', args.tb])
    
    # Add coverage options
    if args.coverage or args.html:
        if not args.no_cov:
            cmd.extend(['--cov=.', '--cov-report=term-missing'])
            if args.html:
                cmd.append('--cov-report=html')
    elif not args.no_cov:
        # Default coverage settings from pytest.ini
        pass
    else:
        cmd.append('--no-cov')
    
    # Add parallel processing
    if args.parallel:
        cmd.extend(['-n', str(args.parallel)])
    
    # Add other useful options
    cmd.extend([
        '--color=yes',
        '--durations=10',
        '--strict-markers'
    ])
    
    # Run the tests
    print("Running MVPA Pipeline Tests...")
    print("=" * 50)
    
    success = run_command(cmd, verbose=args.verbose)
    
    if success:
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        
        if args.html:
            html_report = Path('htmlcov/index.html')
            if html_report.exists():
                print(f"HTML coverage report generated: {html_report.absolute()}")
        
        return 0
    else:
        print("\n" + "=" * 50)
        print("Some tests failed! ✗")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 