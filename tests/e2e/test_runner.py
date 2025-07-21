"""
E2E Test Runner and Utilities

This module provides utilities for running comprehensive E2E tests
and generating detailed reports.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .framework import e2e_framework


class E2ETestRunner:
    """Comprehensive E2E test runner with detailed reporting."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("./e2e_test_results")
        self.output_dir.mkdir(exist_ok=True)

    def run_full_test_suite(  # noqa: C901
        self, backends: Optional[list[str]] = None, workflows: Optional[list[str]] = None, save_results: bool = True
    ) -> dict:
        """Run the complete E2E test suite and generate detailed report."""

        print("🚀 Starting Comprehensive E2E Test Suite")
        print("=" * 60)

        start_time = time.time()
        timestamp = datetime.now().isoformat()

        # Get available backends and workflows
        available_backends = e2e_framework.get_available_backends()
        available_workflows = e2e_framework.get_available_workflows()

        # Use provided or default to all available
        test_backends = backends or available_backends
        test_workflows = workflows or available_workflows

        print(f"Available Backends: {available_backends}")
        print(f"Available Workflows: {available_workflows}")
        print(f"Testing Backends: {test_backends}")
        print(f"Testing Workflows: {test_workflows}")
        print()

        # Initialize results structure
        report = {
            "test_info": {
                "timestamp": timestamp,
                "backends_tested": test_backends,
                "workflows_tested": test_workflows,
                "total_combinations": len(test_backends) * len(test_workflows),
            },
            "environment_info": self._get_environment_info(),
            "results": {
                "summary": {"total_tests": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0},
                "backend_breakdown": {},
                "workflow_breakdown": {},
                "detailed_results": [],
                "performance_metrics": {},
                "error_analysis": [],
            },
            "execution_time": 0,
        }

        # Run tests for each backend/workflow combination
        for backend_name in test_backends:
            print(f"🔧 Testing Backend: {backend_name.upper()}")

            backend_results = {
                "tests_run": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "execution_time": 0,
                "workflows": {},
            }

            try:
                with e2e_framework.backend_environment(backend_name) as backend_config:
                    backend_start_time = time.time()

                    for workflow_name in test_workflows:
                        print(f"  📊 Running workflow: {workflow_name}")

                        test_result = self._run_single_test(workflow_name, backend_config)

                        # Update counters
                        report["results"]["summary"]["total_tests"] += 1
                        backend_results["tests_run"] += 1

                        if test_result["status"] == "passed":
                            report["results"]["summary"]["passed"] += 1
                            backend_results["passed"] += 1
                        elif test_result["status"] == "failed":
                            report["results"]["summary"]["failed"] += 1
                            backend_results["failed"] += 1
                        elif test_result["status"] == "skipped":
                            report["results"]["summary"]["skipped"] += 1
                            backend_results["skipped"] += 1
                        elif test_result["status"] == "error":
                            report["results"]["summary"]["errors"] += 1
                            backend_results["failed"] += 1
                            report["results"]["error_analysis"].append({
                                "backend": backend_name,
                                "workflow": workflow_name,
                                "error": test_result.get("error", "Unknown error"),
                                "timestamp": datetime.now().isoformat(),
                            })

                        # Store detailed result
                        backend_results["workflows"][workflow_name] = test_result
                        report["results"]["detailed_results"].append({
                            "backend": backend_name,
                            "workflow": workflow_name,
                            **test_result,
                        })

                        # Update workflow breakdown
                        if workflow_name not in report["results"]["workflow_breakdown"]:
                            report["results"]["workflow_breakdown"][workflow_name] = {
                                "total": 0,
                                "passed": 0,
                                "failed": 0,
                                "skipped": 0,
                            }

                        report["results"]["workflow_breakdown"][workflow_name]["total"] += 1
                        if test_result["status"] == "passed":
                            report["results"]["workflow_breakdown"][workflow_name]["passed"] += 1
                        elif test_result["status"] in ["failed", "error"]:
                            report["results"]["workflow_breakdown"][workflow_name]["failed"] += 1
                        else:
                            report["results"]["workflow_breakdown"][workflow_name]["skipped"] += 1

                        print(f"    ✅ {test_result['status'].upper()} " f"({test_result['execution_time']:.2f}s)")

                    backend_results["execution_time"] = time.time() - backend_start_time

            except Exception as e:
                print(f"  ❌ Backend {backend_name} setup failed: {e}")
                backend_results["setup_error"] = str(e)

                # Mark all workflows as skipped for this backend
                for _workflow_name in test_workflows:
                    report["results"]["summary"]["total_tests"] += 1
                    report["results"]["summary"]["skipped"] += 1
                    backend_results["tests_run"] += 1
                    backend_results["skipped"] += 1

            report["results"]["backend_breakdown"][backend_name] = backend_results
            print(
                f"  📈 Backend {backend_name} summary: "
                f"{backend_results['passed']} passed, "
                f"{backend_results['failed']} failed, "
                f"{backend_results['skipped']} skipped"
            )
            print()

        # Calculate overall metrics
        report["execution_time"] = time.time() - start_time

        # Generate performance analysis
        report["results"]["performance_metrics"] = self._analyze_performance(report["results"]["detailed_results"])

        # Print final summary
        self._print_final_summary(report)

        # Save results if requested
        if save_results:
            self._save_report(report, timestamp)

        return report

    def _run_single_test(self, workflow_name: str, backend_config: dict) -> dict:
        """Run a single workflow/backend test combination."""
        start_time = time.time()

        try:
            # Check if workflow is available
            if workflow_name not in e2e_framework.workflows:
                return {
                    "status": "skipped",
                    "reason": f"Workflow {workflow_name} not available",
                    "execution_time": time.time() - start_time,
                }

            # Run the workflow
            results = e2e_framework.run_workflow(workflow_name, backend_config)

            execution_time = time.time() - start_time

            if results["success"]:
                return {
                    "status": "passed",
                    "execution_time": execution_time,
                    "workflow_results": results,
                    "metrics_logged": len(results.get("metrics", [])),
                    "final_accuracy": results.get("final_accuracy", results.get("best_val_accuracy", "N/A")),
                }
            else:
                return {
                    "status": "failed",
                    "execution_time": execution_time,
                    "error": results.get("error", "Workflow validation failed"),
                    "workflow_results": results,
                }

        except Exception as e:
            return {"status": "error", "execution_time": time.time() - start_time, "error": str(e)}

    def _get_environment_info(self) -> dict:
        """Collect environment information."""
        import platform
        import sys

        env_info = {"python_version": sys.version, "platform": platform.platform(), "available_packages": {}}

        # Check for optional packages
        packages_to_check = [
            "torch",
            "torchvision",
            "pytorch_lightning",
            "mlflow",
            "clearml",
            "wandb",
            "numpy",
            "matplotlib",
        ]

        for package in packages_to_check:
            try:
                module = __import__(package)
                env_info["available_packages"][package] = getattr(module, "__version__", "unknown")
            except ImportError:
                env_info["available_packages"][package] = "not installed"

        return env_info

    def _analyze_performance(self, detailed_results: list[dict]) -> dict:
        """Analyze performance metrics from test results."""
        if not detailed_results:
            return {}

        # Filter successful tests
        successful_tests = [r for r in detailed_results if r["status"] == "passed"]

        if not successful_tests:
            return {"note": "No successful tests to analyze"}

        # Calculate performance statistics
        execution_times = [r["execution_time"] for r in successful_tests]

        performance = {
            "execution_time_stats": {
                "mean": sum(execution_times) / len(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "total": sum(execution_times),
            },
            "backend_performance": {},
            "workflow_performance": {},
        }

        # Backend performance breakdown
        backends = {r["backend"] for r in successful_tests}
        for backend in backends:
            backend_times = [r["execution_time"] for r in successful_tests if r["backend"] == backend]
            if backend_times:
                performance["backend_performance"][backend] = {
                    "mean_time": sum(backend_times) / len(backend_times),
                    "test_count": len(backend_times),
                    "total_time": sum(backend_times),
                }

        # Workflow performance breakdown
        workflows = {r["workflow"] for r in successful_tests}
        for workflow in workflows:
            workflow_times = [r["execution_time"] for r in successful_tests if r["workflow"] == workflow]
            if workflow_times:
                performance["workflow_performance"][workflow] = {
                    "mean_time": sum(workflow_times) / len(workflow_times),
                    "test_count": len(workflow_times),
                    "total_time": sum(workflow_times),
                }

        return performance

    def _print_final_summary(self, report: dict):
        """Print comprehensive final summary."""
        summary = report["results"]["summary"]

        print("🏁 FINAL TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests Run: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⏭️  Skipped: {summary['skipped']}")
        print(f"💥 Errors: {summary['errors']}")

        if summary["total_tests"] > 0:
            pass_rate = summary["passed"] / summary["total_tests"] * 100
            print(f"📊 Pass Rate: {pass_rate:.1f}%")

        print(f"⏱️  Total Execution Time: {report['execution_time']:.2f}s")
        print()

        # Backend breakdown
        print("🔧 BACKEND BREAKDOWN:")
        for backend, stats in report["results"]["backend_breakdown"].items():
            if "setup_error" in stats:
                print(f"  {backend}: ❌ Setup Error - {stats['setup_error']}")
            else:
                print(
                    f"  {backend}: {stats['passed']}/{stats['tests_run']} passed " f"({stats['execution_time']:.2f}s)"
                )
        print()

        # Workflow breakdown
        print("📊 WORKFLOW BREAKDOWN:")
        for workflow, stats in report["results"]["workflow_breakdown"].items():
            print(f"  {workflow}: {stats['passed']}/{stats['total']} passed")
        print()

        # Performance summary
        if "execution_time_stats" in report["results"]["performance_metrics"]:
            perf = report["results"]["performance_metrics"]["execution_time_stats"]
            print("⚡ PERFORMANCE SUMMARY:")
            print(f"  Average test time: {perf['mean']:.2f}s")
            print(f"  Fastest test: {perf['min']:.2f}s")
            print(f"  Slowest test: {perf['max']:.2f}s")
            print()

        # Error analysis
        if report["results"]["error_analysis"]:
            print("🔍 ERROR ANALYSIS:")
            for error in report["results"]["error_analysis"]:
                print(f"  {error['backend']}/{error['workflow']}: {error['error']}")
            print()

    def _save_report(self, report: dict, timestamp: str):
        """Save detailed report to files."""
        # Save JSON report
        json_file = self.output_dir / f"e2e_report_{timestamp.replace(':', '-')}.json"
        with open(json_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Save markdown summary
        md_file = self.output_dir / f"e2e_summary_{timestamp.replace(':', '-')}.md"
        with open(md_file, "w") as f:
            self._write_markdown_report(f, report)

        print("📁 Reports saved:")
        print(f"  JSON: {json_file}")
        print(f"  Markdown: {md_file}")

    def _write_markdown_report(self, f, report: dict):
        """Write markdown-formatted report."""
        f.write("# Tracelet E2E Test Report\n\n")
        f.write(f"**Generated:** {report['test_info']['timestamp']}\n\n")

        # Summary table
        summary = report["results"]["summary"]
        f.write("## Test Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total Tests | {summary['total_tests']} |\n")
        f.write(f"| Passed | {summary['passed']} |\n")
        f.write(f"| Failed | {summary['failed']} |\n")
        f.write(f"| Skipped | {summary['skipped']} |\n")
        f.write(f"| Errors | {summary['errors']} |\n")

        if summary["total_tests"] > 0:
            pass_rate = summary["passed"] / summary["total_tests"] * 100
            f.write(f"| Pass Rate | {pass_rate:.1f}% |\n")

        f.write(f"| Total Time | {report['execution_time']:.2f}s |\n\n")

        # Backend results
        f.write("## Backend Results\n\n")
        f.write("| Backend | Passed | Total | Time (s) | Status |\n")
        f.write("|---------|--------|-------|----------|--------|\n")

        for backend, stats in report["results"]["backend_breakdown"].items():
            if "setup_error" in stats:
                f.write(f"| {backend} | 0 | 0 | - | ❌ Setup Error |\n")
            else:
                f.write(
                    f"| {backend} | {stats['passed']} | {stats['tests_run']} | "
                    f"{stats['execution_time']:.2f} | ✅ OK |\n"
                )

        f.write("\n")

        # Workflow results
        f.write("## Workflow Results\n\n")
        f.write("| Workflow | Passed | Total | Success Rate |\n")
        f.write("|----------|--------|-------|---------------|\n")

        for workflow, stats in report["results"]["workflow_breakdown"].items():
            success_rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
            f.write(f"| {workflow} | {stats['passed']} | {stats['total']} | {success_rate:.1f}% |\n")

        f.write("\n")

        # Environment info
        f.write("## Environment Information\n\n")
        f.write(f"**Python:** {report['environment_info']['python_version']}\n\n")
        f.write(f"**Platform:** {report['environment_info']['platform']}\n\n")

        f.write("**Package Versions:**\n\n")
        for pkg, version in report["environment_info"]["available_packages"].items():
            f.write(f"- {pkg}: {version}\n")


# Standalone test runner script
def main():
    """Main function for running E2E tests from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Tracelet E2E Tests")
    parser.add_argument("--backends", nargs="*", help="Backends to test (default: all available)")
    parser.add_argument("--workflows", nargs="*", help="Workflows to test (default: all available)")
    parser.add_argument("--output-dir", type=Path, help="Output directory for reports")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")

    args = parser.parse_args()

    runner = E2ETestRunner(args.output_dir)
    results = runner.run_full_test_suite(
        backends=args.backends, workflows=args.workflows, save_results=not args.no_save
    )

    # Exit with appropriate code
    summary = results["results"]["summary"]
    if summary["failed"] > 0 or summary["errors"] > 0:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
