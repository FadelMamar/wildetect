#!/usr/bin/env python3
"""
Unified example demonstrating all TileUtils versions:
- TileUtils: Original implementation with tensor-based offset calculation
- TileUtilsv2: SAHI-based implementation for advanced slicing
- TileUtilsv3: Simplified arithmetic-based implementation for performance
"""


import time
import torch

from wildetect.core.data.utils import TileUtils, TileUtilsv2, TileUtilsv3


def demonstrate_basic_functionality():
    """Demonstrate basic functionality of all TileUtils versions."""
    print("=" * 80)
    print("BASIC FUNCTIONALITY COMPARISON")
    print("=" * 80)

    # Create a test image
    image_tensor = torch.rand(3, 2000, 2000)
    
    print(f"Original image shape: {image_tensor.shape}")

    # Test parameters
    patch_size = 800
    stride = 400
    channels = 3

    try:
        # Validate parameters
        is_valid = TileUtils.validate_patch_parameters(image_tensor.shape, patch_size, stride)
        print(f"Parameters valid: {is_valid}")

        # Calculate expected patch count
        height, width = image_tensor.shape[1], image_tensor.shape[2]
        expected_count = TileUtils.get_patch_count(height, width, patch_size, stride)
        print(f"Expected patch count: {expected_count}")

        # Test all three versions
        versions = [
            ("Original TileUtils", TileUtils),
            ("TileUtilsv2 (SAHI)", TileUtilsv2),
            ("TileUtilsv3 (Simple)", TileUtilsv3)
        ]

        results = {}

        for name, utils_class in versions:
            print(f"\n--- Testing {name} ---")
            
            try:
                patches, offset_info = utils_class.get_patches_and_offset_info(
                    image=image_tensor,
                    patch_size=patch_size,
                    stride=stride,
                    channels=channels,
                    file_name=f"sample_image_{name.lower().replace(' ', '_')}.jpg",
                    validate=True,
                )

                print(f"{name} - Extracted patches shape: {patches.shape}")
                print(f"{name} - Number of patches: {patches.shape[0]}")
                print(f"{name} - Offset info keys: {list(offset_info.keys())}")
                print(f"{name} - Number of offsets: {len(offset_info['x_offset'])}")
                
                results[name] = {
                    'patches': patches,
                    'offset_info': offset_info,
                    'success': True
                }
                
            except Exception as e:
                print(f"{name} - Error: {e}")
                results[name] = {'success': False, 'error': str(e)}

        # Compare results
        print("\n--- Comparing Results ---")
        
        successful_results = {k: v for k, v in results.items() if v['success']}
        
        if len(successful_results) >= 2:
            # Compare patch counts
            patch_counts = {name: data['patches'].shape[0] for name, data in successful_results.items()}
            print(f"Patch counts: {patch_counts}")
            
            # Compare patch shapes
            patch_shapes = {name: data['patches'].shape for name, data in successful_results.items()}
            print(f"Patch shapes: {patch_shapes}")
            
            # Check if patches are identical between versions
            if len(successful_results) == 3:
                patches_original = successful_results["Original TileUtils"]['patches']
                patches_v3 = successful_results["TileUtilsv3 (Simple)"]['patches']
                
                patches_identical = torch.allclose(patches_original, patches_v3, atol=1e-6)
                print(f"Original vs v3 patches identical: {patches_identical}")

        print("\n✓ All basic functionality tests completed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_performance_comparison():
    """Compare performance between all TileUtils versions."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    # Create a larger test image
    large_image = torch.rand(3, 8000, 8000)
    
    patch_size = 256
    stride = 128

    versions = [
        ("Original TileUtils", TileUtils),
        ("TileUtilsv2 (SAHI)", TileUtilsv2),
        ("TileUtilsv3 (Simple)", TileUtilsv3)
    ]

    results = {}

    for name, utils_class in versions:
        print(f"\n--- Testing {name} ---")
        
        start_time = time.perf_counter()
        try:
            for _ in range(1):
                patches, _ = utils_class.get_patches_and_offset_info(
                    image=large_image,
                    patch_size=patch_size,
                    stride=stride,
                    channels=3,
                    validate=False,  # Disable validation for fair comparison
                )
            execution_time = (time.perf_counter() - start_time) / 10
            print(f"{name} time: {execution_time:.4f}s")
            print(f"{name} patches: {patches.shape[0]}")
            results[name] = {
                'time': execution_time,
                'patches': patches.shape[0],
                'success': True
            }
            
        except Exception as e:
            print(f"{name} error: {e}")
            results[name] = {'success': False, 'error': str(e)}

    # Calculate speedups
    print("\n--- Performance Summary ---")
    
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if len(successful_results) >= 2:
        # Find the slowest version as baseline
        baseline_name = max(successful_results.keys(), key=lambda x: successful_results[x]['time'])
        baseline_time = successful_results[baseline_name]['time']
        
        print(f"Baseline ({baseline_name}): {baseline_time:.4f}s")
        
        for name, data in successful_results.items():
            if name != baseline_name:
                speedup = baseline_time / data['time']
                print(f"{name}: {data['time']:.4f}s (speedup: {speedup:.2f}x)")


def test_edge_cases():
    """Test edge cases for all TileUtils versions."""
    print("\n" + "=" * 80)
    print("EDGE CASES TESTING")
    print("=" * 80)

    versions = [
        ("Original TileUtils", TileUtils),
        ("TileUtilsv2 (SAHI)", TileUtilsv2),
        ("TileUtilsv3 (Simple)", TileUtilsv3)
    ]

    # Test with small image
    print("\n--- Small Image Test ---")
    small_image = torch.randn(3, 100, 100)
    small_image = (small_image - small_image.min()) / (small_image.max() - small_image.min())
    
    for name, utils_class in versions:
        try:
            patches, offset_info = utils_class.get_patches_and_offset_info(
                image=small_image,
                patch_size=256,  # Larger than image
                stride=128,
                channels=3,
            )
            print(f"{name} - patches shape: {patches.shape}")
            print(f"{name} - should return original image: {patches.shape[0] == 1}")
        except Exception as e:
            print(f"{name} - error: {e}")

    # Test with exact patch size
    print("\n--- Exact Patch Size Test ---")
    exact_image = torch.randn(3, 256, 256)
    exact_image = (exact_image - exact_image.min()) / (exact_image.max() - exact_image.min())
    
    for name, utils_class in versions:
        try:
            patches, offset_info = utils_class.get_patches_and_offset_info(
                image=exact_image,
                patch_size=256,
                stride=256,
                channels=3,
            )
            print(f"{name} - patches shape: {patches.shape}")
            print(f"{name} - should return 1 patch: {patches.shape[0] == 1}")
        except Exception as e:
            print(f"{name} - error: {e}")


def test_special_features():
    """Test special features of each version."""
    print("\n" + "=" * 80)
    print("SPECIAL FEATURES TESTING")
    print("=" * 80)

    # Test SAHI-specific features (TileUtilsv2)
    print("\n--- SAHI Features (TileUtilsv2) ---")
    image_tensor = torch.randn(3, 1000, 1000)
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
    
    try:
        # Test get_sliced_images
        sliced_images = TileUtilsv2.get_sliced_images(
            image=image_tensor,
            patch_size=256,
            stride=128,
        )
        print(f"SAHI sliced images count: {len(sliced_images)}")
        print(f"First sliced image size: {sliced_images[0].size if sliced_images else 'N/A'}")
        
        # Test get_slice_metadata
        metadata = TileUtilsv2.get_slice_metadata(
            image=image_tensor,
            patch_size=256,
            stride=128,
        )
        print(f"SAHI metadata: {metadata}")
    except Exception as e:
        print(f"SAHI features error: {e}")

    # Test simple arithmetic features (TileUtilsv3)
    print("\n--- Simple Arithmetic Features (TileUtilsv3) ---")
    try:
        # Test direct offset calculation
        offset_info = TileUtilsv3._calculate_offset_info_simple(1000, 1000, 256, 128)
        print(f"Simple offset calculation - number of patches: {len(offset_info['x_offset'])}")
        print(f"First patch offset: ({offset_info['x_offset'][0]}, {offset_info['y_offset'][0]})")
    except Exception as e:
        print(f"Simple arithmetic features error: {e}")




if __name__ == "__main__":
    print("TILEUTILS UNIFIED EXAMPLE")
    print("Comparing Original, v2 (SAHI), and v3 (Simple) implementations")
    
    demonstrate_basic_functionality()
    #test_performance_comparison()
    #test_edge_cases()
    #test_special_features()