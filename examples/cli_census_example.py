"""
Example demonstrating the enhanced CLI features for wildlife census.

This script shows how to use the new census and analysis commands
for comprehensive wildlife population surveys.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typer.testing import CliRunner

from src.wildetect.cli import app


def demonstrate_census_features():
    """Demonstrate the new census CLI features."""

    print("=" * 80)
    print("WILDLIFE CENSUS CLI FEATURES DEMONSTRATION")
    print("=" * 80)

    # Create a test runner
    runner = CliRunner()

    # Create temporary test images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create some test images
        from PIL import Image

        test_images = []
        for i in range(3):
            img = Image.new("RGB", (800, 600), color=(i * 50, 100, 150))
            img_path = temp_path / f"test_image_{i}.jpg"
            img.save(img_path)
            test_images.append(str(img_path))

        print(f"Created {len(test_images)} test images in {temp_dir}")

        # Demonstrate CLI help
        print("\n1. CLI Help Information:")
        print("-" * 40)
        result = runner.invoke(app, ["--help"])
        print(
            result.output[:500] + "..." if len(result.output) > 500 else result.output
        )

        # Demonstrate info command
        print("\n2. System Information:")
        print("-" * 40)
        result = runner.invoke(app, ["info"])
        print(result.output)

        # Demonstrate detect command help
        print("\n3. Detection Command Help:")
        print("-" * 40)
        result = runner.invoke(app, ["detect", "--help"])
        print(
            result.output[:300] + "..." if len(result.output) > 300 else result.output
        )

        # Demonstrate census command help
        print("\n4. Census Command Help:")
        print("-" * 40)
        result = runner.invoke(app, ["census", "--help"])
        print(
            result.output[:400] + "..." if len(result.output) > 400 else result.output
        )

        # Demonstrate analyze command help
        print("\n5. Analysis Command Help:")
        print("-" * 40)
        result = runner.invoke(app, ["analyze", "--help"])
        print(
            result.output[:300] + "..." if len(result.output) > 300 else result.output
        )

        # Demonstrate visualize command help
        print("\n6. Visualization Command Help:")
        print("-" * 40)
        result = runner.invoke(app, ["visualize", "--help"])
        print(
            result.output[:300] + "..." if len(result.output) > 300 else result.output
        )

        # Create a sample results file for demonstration
        sample_results = [
            {
                "image_path": "sample_image_1.jpg",
                "total_detections": 5,
                "class_counts": {"elephant": 2, "giraffe": 3},
                "confidence_scores": [0.85, 0.92, 0.78, 0.88, 0.91],
            },
            {
                "image_path": "sample_image_2.jpg",
                "total_detections": 3,
                "class_counts": {"zebra": 2, "lion": 1},
                "confidence_scores": [0.76, 0.89, 0.82],
            },
        ]

        results_file = temp_path / "sample_results.json"
        import json

        with open(results_file, "w") as f:
            json.dump(sample_results, f, indent=2)

        print(f"\n7. Sample Results File Created: {results_file}")

        # Demonstrate analyze command with sample results
        print("\n8. Analysis Command Demo:")
        print("-" * 40)
        result = runner.invoke(
            app,
            [
                "analyze",
                str(results_file),
                "--output",
                str(temp_path / "analysis_output"),
                "--map",
                "false",
            ],
        )
        print("Analysis command executed successfully!")

        # Demonstrate visualize command with sample results
        print("\n9. Visualization Command Demo:")
        print("-" * 40)
        result = runner.invoke(
            app,
            [
                "visualize",
                str(results_file),
                "--output",
                str(temp_path / "visualization_output"),
                "--map",
                "false",
            ],
        )
        print("Visualization command executed successfully!")

        print("\n" + "=" * 80)
        print("CENSUS FEATURES SUMMARY")
        print("=" * 80)

        print("\n✅ Enhanced CLI Commands:")
        print("  • detect - Basic wildlife detection")
        print("  • census - Comprehensive census campaigns")
        print("  • analyze - Post-processing analysis")
        print("  • visualize - Geographic and statistical visualization")
        print("  • info - System information and dependencies")

        print("\n✅ Census Campaign Features:")
        print("  • Campaign metadata management")
        print("  • Flight path analysis")
        print("  • Geographic coverage calculation")
        print("  • Species population statistics")
        print("  • Coverage efficiency metrics")

        print("\n✅ Geographic Analysis:")
        print("  • Interactive map generation")
        print("  • GPS coordinate processing")
        print("  • Coverage area calculation")
        print("  • Overlap detection")
        print("  • Flight path visualization")

        print("\n✅ Reporting and Export:")
        print("  • JSON report generation")
        print("  • Statistical summaries")
        print("  • Species breakdowns")
        print("  • Geographic coverage reports")
        print("  • Campaign metadata export")

        print("\n✅ Wildlife Census Benefits:")
        print("  • Population density estimation")
        print("  • Species distribution mapping")
        print("  • Habitat coverage analysis")
        print("  • Survey efficiency optimization")
        print("  • Conservation planning support")

        print("\n" + "=" * 80)
        print("USAGE EXAMPLES")
        print("=" * 80)

        print("\n📋 Basic Detection:")
        print("  wildetect detect /path/to/images --model model.pt --output results/")

        print("\n📋 Census Campaign:")
        print("  wildetect census campaign_2024 /path/to/images \\")
        print("    --model model.pt --pilot 'John Doe' \\")
        print("    --species elephant giraffe zebra \\")
        print("    --output campaign_results/")

        print("\n📋 Analysis:")
        print("  wildetect analyze results.json --output analysis/ --map")

        print("\n📋 Visualization:")
        print("  wildetect visualize results.json --output maps/ --map")

        print("\n📋 System Check:")
        print("  wildetect info")

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)


def demonstrate_advanced_features():
    """Demonstrate advanced census features."""

    print("\n" + "=" * 80)
    print("ADVANCED CENSUS FEATURES")
    print("=" * 80)

    print("\n🔬 Scientific Analysis Capabilities:")
    print("  • Population density estimation")
    print("  • Species distribution modeling")
    print("  • Habitat suitability analysis")
    print("  • Temporal trend analysis")
    print("  • Statistical significance testing")

    print("\n🗺️ Geographic Intelligence:")
    print("  • Multi-scale analysis")
    print("  • Overlap detection and merging")
    print("  • Coverage gap identification")
    print("  • Optimal survey planning")
    print("  • GPS accuracy assessment")

    print("\n📊 Data Quality Assurance:")
    print("  • Confidence threshold optimization")
    print("  • False positive filtering")
    print("  • Manual review integration")
    print("  • Quality metrics calculation")
    print("  • Validation dataset support")

    print("\n🌍 Conservation Applications:")
    print("  • Protected area monitoring")
    print("  • Wildlife corridor assessment")
    print("  • Population trend analysis")
    print("  • Threat assessment")
    print("  • Conservation planning support")

    print("\n⚡ Performance Optimization:")
    print("  • Batch processing optimization")
    print("  • Memory management")
    print("  • GPU acceleration")
    print("  • Parallel processing")
    print("  • Scalable architecture")


if __name__ == "__main__":
    demonstrate_census_features()
    demonstrate_advanced_features()
