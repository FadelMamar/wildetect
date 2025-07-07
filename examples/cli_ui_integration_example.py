#!/usr/bin/env python3
"""
CLI-UI Integration Example

This example demonstrates how to use the CLI-UI integration functionality
programmatically without the Streamlit UI.
"""

import json
import tempfile
from pathlib import Path

from wildetect.cli_ui_integration import cli_ui_integration


def main():
    """Demonstrate CLI-UI integration functionality."""
    print("🦁 WildDetect CLI-UI Integration Example")
    print("=" * 50)

    # 1. Get system information
    print("\n1. System Information:")
    system_info = cli_ui_integration.get_system_info_ui()

    print("Components:")
    for component, info in system_info["components"].items():
        status = "✓" if info["status"] == "✓" else "✗"
        print(f"  {component}: {status} - {info['details']}")

    print("\nDependencies:")
    for dep, info in system_info["dependencies"].items():
        status = "✓" if info["status"] == "✓" else "✗"
        print(f"  {dep}: {status} - {info['details']}")

    # 2. Create sample results for analysis
    print("\n2. Results Analysis:")
    sample_results = [
        {
            "image_path": "sample1.jpg",
            "total_detections": 5,
            "class_counts": {"elephant": 3, "lion": 2},
        },
        {
            "image_path": "sample2.jpg",
            "total_detections": 3,
            "class_counts": {"elephant": 1, "giraffe": 2},
        },
        {"image_path": "sample3.jpg", "total_detections": 0, "class_counts": {}},
    ]

    # Save sample results to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_results, f)
        results_file = f.name

    try:
        # Analyze results
        analysis_result = cli_ui_integration.analyze_results_ui(
            results_file, output_dir="example_analysis", create_map=False
        )

        if analysis_result["success"]:
            print("✓ Analysis completed successfully!")
            analysis = analysis_result["analysis_results"]
            print(f"  Total Images: {analysis['total_images']}")
            print(f"  Total Detections: {analysis['total_detections']}")
            print("  Species Breakdown:")
            for species, count in analysis["species_breakdown"].items():
                print(f"    {species}: {count}")
        else:
            print(f"✗ Analysis failed: {analysis_result['error']}")

    finally:
        # Clean up temporary file
        Path(results_file).unlink(missing_ok=True)

    # 3. Visualize results
    print("\n3. Results Visualization:")

    # Create another sample for visualization
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_results, f)
        viz_results_file = f.name

    try:
        viz_result = cli_ui_integration.visualize_results_ui(
            viz_results_file,
            output_dir="example_visualizations",
            show_confidence=True,
            create_map=False,
        )

        if viz_result["success"]:
            print("✓ Visualization completed successfully!")
            viz_data = viz_result["visualization_data"]
            print(f"  Total Images: {viz_data['total_images']}")
            print(f"  Total Detections: {viz_data['total_detections']}")
            print("  Species Detected:")
            for species, count in viz_data["species_counts"].items():
                print(f"    {species}: {count}")
        else:
            print(f"✗ Visualization failed: {viz_result['error']}")

    finally:
        # Clean up temporary file
        Path(viz_results_file).unlink(missing_ok=True)

    # 4. Demonstrate detection simulation (without actual model)
    print("\n4. Detection Simulation:")
    print("Note: This is a simulation since no model files are available")

    # Simulate detection results
    detection_result = cli_ui_integration.run_detection_ui(
        images=["sample1.jpg", "sample2.jpg"],
        confidence=0.5,
        progress_bar=None,
        status_text=None,
    )

    if detection_result["success"]:
        print("✓ Detection simulation completed!")
        print(f"  Total Images: {detection_result['total_images']}")
        print(f"  Total Detections: {detection_result['total_detections']}")
    else:
        print(f"✗ Detection simulation failed: {detection_result['error']}")

    # 5. Demonstrate census campaign simulation
    print("\n5. Census Campaign Simulation:")
    print("Note: This is a simulation since no model files are available")

    campaign_result = cli_ui_integration.run_census_ui(
        campaign_id="example_campaign",
        images=["sample1.jpg", "sample2.jpg"],
        confidence=0.5,
        pilot_name="Example Pilot",
        target_species=["elephant", "lion", "giraffe"],
        create_map=False,
        progress_bar=None,
        status_text=None,
    )

    if campaign_result["success"]:
        print("✓ Census campaign simulation completed!")
        print(f"  Campaign ID: {campaign_result['campaign_id']}")
    else:
        print(f"✗ Census campaign simulation failed: {campaign_result['error']}")

    print("\n" + "=" * 50)
    print("✅ CLI-UI Integration Example Completed!")
    print("\nTo use the full UI with Streamlit:")
    print("  wildetect ui")
    print("\nTo use CLI commands directly:")
    print("  wildetect detect images/ --confidence 0.5")
    print("  wildetect census campaign_001 images/ --species elephant lion")


if __name__ == "__main__":
    main()
