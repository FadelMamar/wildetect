#!/usr/bin/env python3
"""
WildDetect Detection Script

Run wildlife detection on aerial images with command-line interface.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.core.detector import WildlifeDetector
from app.utils.config import get_config, create_directories

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main detection script."""
    parser = argparse.ArgumentParser(description="WildDetect Detection")
    parser.add_argument('--images', nargs='+', required=True, help='Image paths')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', help='Output directory for visualizations')
    parser.add_argument('--results', help='Output file for detection results (JSON)')
    parser.add_argument('--model', help='Model path (optional)')
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Initialize detector
    detector = WildlifeDetector(model_path=args.model)
    
    # Run detection
    logger.info(f"Running detection on {len(args.images)} images...")
    results = []
    
    for i, image_path in enumerate(args.images):
        logger.info(f"Processing {i+1}/{len(args.images)}: {image_path}")
        
        try:
            result = detector.detect(image_path, args.confidence)
            results.append(result)
            
            # Save visualization if output_dir specified
            if args.output and result.get('detections'):
                os.makedirs(args.output, exist_ok=True)
                vis_path = os.path.join(args.output, f"detection_{i:04d}.jpg")
                detector.visualize_detections(image_path, result['detections'], vis_path)
                logger.info(f"Saved visualization: {vis_path}")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'error': str(e),
                'detections': [],
                'total_count': 0,
                'species_counts': {}
            })
    
    # Print summary
    total_detections = sum(r.get('total_count', 0) for r in results)
    logger.info(f"Detection completed: {total_detections} wildlife found in {len(results)} images")
    
    print(f"\nDetection Summary:")
    print(f"  Images processed: {len(results)}")
    print(f"  Total detections: {total_detections}")
    
    # Species breakdown
    all_species = {}
    for result in results:
        for species, count in result.get('species_counts', {}).items():
            all_species[species] = all_species.get(species, 0) + count
    
    if all_species:
        print(f"  Species found:")
        for species, count in sorted(all_species.items(), key=lambda x: x[1], reverse=True):
            print(f"    {species}: {count}")
    
    # Save results if requested
    if args.results:
        with open(args.results, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.results}")
    
    return results


if __name__ == "__main__":
    main() 