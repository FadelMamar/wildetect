"""
Comprehensive test runner for all wildetect functionality.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Test runner for executing all test scripts."""

    def __init__(self):
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def run_test_script(self, script_path: Path) -> Dict[str, Any]:
        """Run a single test script.

        Args:
            script_path (Path): Path to the test script

        Returns:
            Dict[str, Any]: Test results
        """
        logger.info(f"Running test script: {script_path.name}")

        start_time = time.time()

        try:
            # Run the test script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            end_time = time.time()
            duration = end_time - start_time

            success = result.returncode == 0

            return {
                "script_name": script_path.name,
                "success": success,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired:
            logger.error(f"Test script timed out: {script_path.name}")
            return {
                "script_name": script_path.name,
                "success": False,
                "duration": 300,
                "stdout": "",
                "stderr": "Test timed out after 5 minutes",
                "returncode": -1,
            }
        except Exception as e:
            logger.error(f"Failed to run test script {script_path.name}: {e}")
            return {
                "script_name": script_path.name,
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
            }

    def run_all_tests(self, test_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Run all test scripts in the test directory.

        Args:
            test_dir (Optional[Path]): Directory containing test scripts

        Returns:
            Dict[str, Any]: Summary of all test results
        """
        if test_dir is None:
            test_dir = Path(__file__).parent

        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE TEST SUITE")
        logger.info("=" * 80)

        # Find all test scripts
        test_scripts = list(test_dir.glob("test_*.py"))
        test_scripts.sort()  # Sort for consistent order

        logger.info(f"Found {len(test_scripts)} test scripts:")
        for script in test_scripts:
            logger.info(f"  - {script.name}")

        # Run each test script
        for script in test_scripts:
            if script.name == "run_all_tests.py":  # Skip this script
                continue

            result = self.run_test_script(script)
            self.test_results[script.name] = result

            if result["success"]:
                self.passed_tests += 1
                logger.info(f"✓ {script.name} PASSED ({result['duration']:.2f}s)")
            else:
                self.failed_tests += 1
                logger.error(f"✗ {script.name} FAILED ({result['duration']:.2f}s)")
                if result["stderr"]:
                    logger.error(f"  Error: {result['stderr']}")

        self.total_tests = len(self.test_results)

        # Generate summary
        summary = self.generate_summary()

        logger.info("=" * 80)
        logger.info("TEST SUITE COMPLETED")
        logger.info("=" * 80)
        logger.info(summary["summary_text"])

        return summary

    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of test results.

        Returns:
            Dict[str, Any]: Test summary
        """
        # Calculate statistics
        total_duration = sum(
            result["duration"] for result in self.test_results.values()
        )
        success_rate = (
            (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        )

        # Generate summary text
        summary_lines = [
            f"Total Tests: {self.total_tests}",
            f"Passed: {self.passed_tests}",
            f"Failed: {self.failed_tests}",
            f"Success Rate: {success_rate:.1f}%",
            f"Total Duration: {total_duration:.2f}s",
            "",
            "Detailed Results:",
        ]

        for script_name, result in self.test_results.items():
            status = "PASSED" if result["success"] else "FAILED"
            summary_lines.append(
                f"  {script_name}: {status} ({result['duration']:.2f}s)"
            )

        summary_text = "\n".join(summary_lines)

        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "success_rate": success_rate,
            "total_duration": total_duration,
            "summary_text": summary_text,
            "results": self.test_results,
        }

    def print_detailed_results(self):
        """Print detailed results for failed tests."""
        if self.failed_tests == 0:
            logger.info("All tests passed! No detailed results needed.")
            return

        logger.info("\n" + "=" * 80)
        logger.info("DETAILED FAILURE ANALYSIS")
        logger.info("=" * 80)

        for script_name, result in self.test_results.items():
            if not result["success"]:
                logger.error(f"\nFAILED TEST: {script_name}")
                logger.error(f"Duration: {result['duration']:.2f}s")
                logger.error(f"Return Code: {result['returncode']}")

                if result["stderr"]:
                    logger.error("STDERR:")
                    logger.error(result["stderr"])

                if result["stdout"]:
                    logger.info("STDOUT:")
                    logger.info(result["stdout"])

    def save_results(self, output_file: Path):
        """Save test results to a file.

        Args:
            output_file (Path): Path to save results
        """
        import json

        summary = self.generate_summary()

        # Prepare data for JSON serialization
        serializable_results = {}
        for script_name, result in self.test_results.items():
            serializable_results[script_name] = {
                "script_name": result["script_name"],
                "success": result["success"],
                "duration": result["duration"],
                "returncode": result["returncode"],
                "stderr": result["stderr"][:1000]
                if result["stderr"]
                else "",  # Truncate long output
                "stdout": result["stdout"][:1000]
                if result["stdout"]
                else "",  # Truncate long output
            }

        output_data = {
            "summary": {
                "total_tests": summary["total_tests"],
                "passed_tests": summary["passed_tests"],
                "failed_tests": summary["failed_tests"],
                "success_rate": summary["success_rate"],
                "total_duration": summary["total_duration"],
            },
            "results": serializable_results,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Test results saved to: {output_file}")


def run_specific_test(test_name: str):
    """Run a specific test by name.

    Args:
        test_name (str): Name of the test to run (e.g., 'test_data_loading')
    """
    test_dir = Path(__file__).parent
    test_script = test_dir / f"test_{test_name}.py"

    if not test_script.exists():
        logger.error(f"Test script not found: {test_script}")
        return

    runner = TestRunner()
    result = runner.run_test_script(test_script)

    if result["success"]:
        logger.info(f"✓ {test_name} PASSED")
    else:
        logger.error(f"✗ {test_name} FAILED")
        if result["stderr"]:
            logger.error(f"Error: {result['stderr']}")


def main():
    """Main function to run all tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Run wildetect test suite")
    parser.add_argument(
        "--test",
        type=str,
        help="Run a specific test (e.g., 'data_loading', 'census_data')",
    )
    parser.add_argument("--output", type=str, help="Save results to file")
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed failure analysis"
    )

    args = parser.parse_args()

    if args.test:
        # Run specific test
        run_specific_test(args.test)
    else:
        # Run all tests
        runner = TestRunner()
        summary = runner.run_all_tests()

        if args.detailed:
            runner.print_detailed_results()

        if args.output:
            runner.save_results(Path(args.output))

        # Exit with appropriate code
        if summary["failed_tests"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()
