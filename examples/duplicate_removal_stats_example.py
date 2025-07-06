"""
Example demonstrating how to use duplicate removal statistics from GeographicMerger.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from wildetect.core.config import FlightSpecs
from wildetect.core.data.detection import Detection
from wildetect.core.data.drone_image import DroneImage
from wildetect.core.flight.geographic_merger import GeographicMerger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_duplicate_removal_stats(stats: Dict[str, Any]) -> None:
    """Analyze and display duplicate removal statistics."""

    print("=" * 60)
    print("DUPLICATE REMOVAL STATISTICS")
    print("=" * 60)

    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  • Total image pairs processed: {stats['total_image_pairs_processed']}")
    print(f"  • Total detections removed: {stats['total_detections_removed']}")
    print(f"  • Duplicate removal rate: {stats['duplicate_removal_rate']:.2%}")

    # Class-specific statistics
    print(f"\n🎯 CLASS-SPECIFIC STATISTICS:")
    for class_name, class_stats in stats["class_duplicate_stats"].items():
        print(f"  • {class_name}:")
        print(f"    - Total duplicates: {class_stats['total_duplicates']}")
        print(f"    - Average confidence: {class_stats['avg_confidence']:.3f}")
        print(f"    - Removal rate: {class_stats['removal_rate']:.2%}")

    # Duplicate groups by class
    print(f"\n📈 DUPLICATE GROUPS BY CLASS:")
    for class_name, count in stats["duplicate_groups_by_class"].items():
        print(f"  • {class_name}: {count} duplicate groups")

    # Quality insights
    print(f"\n🔍 QUALITY INSIGHTS:")
    if stats["total_detections_removed"] > 0:
        print(
            f"  • Average confidence improvement: {stats['avg_confidence_improvement']:.3f}"
        )
        print(
            f"  • Detections with confidence boost: {stats.get('detections_with_confidence_boost', 0)}"
        )

    # Geographic insights
    geo_stats = stats.get("geographic_spread", {})
    if geo_stats:
        print(f"\n🗺️  GEOGRAPHIC INSIGHTS:")
        print(
            f"  • Average duplicate distance: {geo_stats.get('avg_duplicate_distance', 0):.2f} meters"
        )
        print(f"  • Boundary duplicates: {geo_stats.get('boundary_duplicates', 0)}")
        if geo_stats.get("duplicate_hotspots"):
            print(
                f"  • Duplicate hotspots identified: {len(geo_stats['duplicate_hotspots'])}"
            )


def generate_duplicate_removal_report(
    stats: Dict[str, Any], output_path: str = "duplicate_removal_report.json"
) -> None:
    """Generate a detailed JSON report of duplicate removal statistics."""

    report = {
        "summary": {
            "total_detections_removed": stats["total_detections_removed"],
            "duplicate_removal_rate": stats["duplicate_removal_rate"],
            "total_image_pairs_processed": stats["total_image_pairs_processed"],
        },
        "class_analysis": stats["class_duplicate_stats"],
        "duplicate_groups": stats["duplicate_groups_by_class"],
        "geographic_analysis": stats.get("geographic_spread", {}),
        "quality_metrics": {
            "avg_confidence_improvement": stats.get("avg_confidence_improvement", 0.0),
            "detections_with_confidence_boost": stats.get(
                "detections_with_confidence_boost", 0
            ),
        },
    }

    # Save to file
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Detailed report saved to: {output_path}")


def compute_additional_insights(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Compute additional insights from the statistics."""

    insights = {
        "efficiency_metrics": {},
        "quality_improvements": {},
        "recommendations": [],
    }

    # Efficiency metrics
    total_removed = stats["total_detections_removed"]
    removal_rate = stats["duplicate_removal_rate"]

    insights["efficiency_metrics"] = {
        "removal_efficiency": removal_rate,
        "memory_savings_estimate": total_removed * 0.1,  # Rough estimate
        "processing_impact": "high" if removal_rate > 0.1 else "low",
    }

    # Quality improvements
    insights["quality_improvements"] = {
        "confidence_boost": stats.get("avg_confidence_improvement", 0.0),
        "duplicate_reduction": removal_rate,
        "data_quality_score": 1.0 - removal_rate,  # Higher is better
    }

    # Recommendations
    recommendations = []

    if removal_rate > 0.2:
        recommendations.append(
            "High duplicate rate detected - consider adjusting IoU threshold"
        )

    if stats.get("avg_confidence_improvement", 0.0) < 0.05:
        recommendations.append("Low confidence improvement - review detection quality")

    class_with_most_duplicates = max(
        stats["duplicate_groups_by_class"].items(), key=lambda x: x[1], default=("", 0)
    )
    if class_with_most_duplicates[1] > 10:
        recommendations.append(
            f"Class '{class_with_most_duplicates[0]}' has many duplicates - review detection parameters"
        )

    insights["recommendations"] = recommendations

    return insights


def main():
    """Main example demonstrating duplicate removal statistics."""

    # Initialize the geographic merger
    merger = GeographicMerger(merge_distance_threshold_m=50.0)

    # Example: Load your drone images here
    # drone_images = load_drone_images_from_directory("path/to/images")

    # For demonstration, we'll show how to use the statistics
    print("🚁 Geographic Merger Duplicate Removal Statistics")
    print("=" * 60)

    # Example statistics (replace with actual data from your merger.run() call)
    example_stats = {
        "total_image_pairs_processed": 45,
        "total_detections_removed": 23,
        "duplicate_removal_rate": 0.15,
        "duplicate_groups_by_class": {"person": 12, "car": 8, "truck": 3},
        "class_duplicate_stats": {
            "person": {
                "total_duplicates": 12,
                "avg_confidence": 0.85,
                "removal_rate": 0.08,
            },
            "car": {
                "total_duplicates": 8,
                "avg_confidence": 0.78,
                "removal_rate": 0.05,
            },
            "truck": {
                "total_duplicates": 3,
                "avg_confidence": 0.92,
                "removal_rate": 0.02,
            },
        },
        "avg_confidence_improvement": 0.12,
        "geographic_spread": {
            "avg_duplicate_distance": 15.5,
            "duplicate_hotspots": ["region_a", "region_b"],
            "boundary_duplicates": 5,
        },
    }

    # Analyze the statistics
    analyze_duplicate_removal_stats(example_stats)

    # Generate detailed report
    generate_duplicate_removal_report(example_stats)

    # Compute additional insights
    insights = compute_additional_insights(example_stats)

    print(f"\n💡 ADDITIONAL INSIGHTS:")
    print(
        f"  • Processing impact: {insights['efficiency_metrics']['processing_impact']}"
    )
    print(
        f"  • Data quality score: {insights['quality_improvements']['data_quality_score']:.2f}"
    )
    print(
        f"  • Estimated memory savings: {insights['efficiency_metrics']['memory_savings_estimate']:.1f} MB"
    )

    if insights["recommendations"]:
        print(f"\n🔧 RECOMMENDATIONS:")
        for rec in insights["recommendations"]:
            print(f"  • {rec}")

    print(
        f"\n✅ Analysis complete! Check 'duplicate_removal_report.json' for detailed results."
    )


if __name__ == "__main__":
    main()
