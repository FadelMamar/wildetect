"""
Example script demonstrating how to use the WildDetect API.

This script shows how to interact with the FastAPI endpoints
for wildlife detection, census campaigns, and data analysis.
"""

import requests
import json
import time
from pathlib import Path
from typing import Dict, List

# API base URL
API_BASE_URL = "http://localhost:8000"

def check_api_status():
    """Check if the API server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            print("✅ API server is running")
            return True
        else:
            print("❌ API server returned error")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure it's running with:")
        print("   wildetect api")
        return False

def get_system_info():
    """Get system information from the API."""
    response = requests.get(f"{API_BASE_URL}/info")
    if response.status_code == 200:
        info = response.json()
        print("\n📊 System Information:")
        print(f"   PyTorch Version: {info['pytorch_version']}")
        print(f"   CUDA Available: {info['cuda_available']}")
        if info['cuda_device']:
            print(f"   CUDA Device: {info['cuda_device']}")
        
        print("\n📦 Dependencies:")
        for dep, available in info['dependencies'].items():
            status = "✅" if available else "❌"
            print(f"   {status} {dep}")
        return info
    else:
        print("❌ Failed to get system info")
        return None

def upload_images(image_paths: List[str]):
    """Upload images to the API."""
    if not image_paths:
        print("❌ No image paths provided")
        return None
    
    files = []
    for image_path in image_paths:
        if Path(image_path).exists():
            files.append(('files', open(image_path, 'rb')))
        else:
            print(f"⚠️  Image not found: {image_path}")
    
    if not files:
        print("❌ No valid images to upload")
        return None
    
    response = requests.post(f"{API_BASE_URL}/upload", files=files)
    
    # Close file handles
    for _, file in files:
        file.close()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Uploaded {len(result['files'])} images")
        print(f"   Upload directory: {result['upload_dir']}")
        return result
    else:
        print(f"❌ Upload failed: {response.text}")
        return None

def start_detection_job(upload_result: Dict, model_path: str = None):
    """Start a wildlife detection job."""
    detection_request = {
        "model_path": model_path,
        "confidence": 0.3,
        "device": "auto",
        "batch_size": 16,
        "tile_size": 800,
        "output": "api_results",
        "pipeline_type": "single"
    }
    
    response = requests.post(f"{API_BASE_URL}/detect", json=detection_request)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Detection job started")
        print(f"   Job ID: {result['job_id']}")
        return result['job_id']
    else:
        print(f"❌ Failed to start detection job: {response.text}")
        return None

def start_census_campaign(upload_result: Dict, campaign_id: str, model_path: str = None):
    """Start a census campaign."""
    census_request = {
        "campaign_id": campaign_id,
        "model_path": model_path,
        "confidence": 0.3,
        "device": "auto",
        "batch_size": 16,
        "tile_size": 800,
        "output": f"census_results/{campaign_id}",
        "pilot_name": "API User",
        "target_species": ["deer", "elk", "moose"],
        "export_to_fiftyone": True,
        "create_map": True
    }
    
    response = requests.post(f"{API_BASE_URL}/census", json=census_request)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Census campaign started")
        print(f"   Job ID: {result['job_id']}")
        print(f"   Campaign ID: {result['campaign_id']}")
        return result['job_id']
    else:
        print(f"❌ Failed to start census campaign: {response.text}")
        return None

def monitor_job(job_id: str, max_wait_time: int = 300):
    """Monitor a background job until completion."""
    print(f"🔄 Monitoring job {job_id}...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        response = requests.get(f"{API_BASE_URL}/jobs/{job_id}")
        
        if response.status_code == 200:
            job_info = response.json()
            status = job_info['status']
            message = job_info['message']
            progress = job_info.get('progress', 0)
            
            print(f"   Status: {status} ({progress}%) - {message}")
            
            if status == "completed":
                print("✅ Job completed successfully!")
                if 'results' in job_info:
                    results = job_info['results']
                    print(f"   Results: {results}")
                return job_info
            elif status == "failed":
                print(f"❌ Job failed: {message}")
                return job_info
        
        time.sleep(5)  # Check every 5 seconds
    
    print("⏰ Job monitoring timed out")
    return None

def visualize_results(results_path: str):
    """Create visualizations from detection results."""
    visualization_request = {
        "show_confidence": True
    }
    
    response = requests.post(
        f"{API_BASE_URL}/visualize",
        data={"results_path": results_path},
        json=visualization_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Visualization created successfully")
        print(f"   Map file: {result['map_file']}")
        print(f"   Output directory: {result['output_dir']}")
        return result
    else:
        print(f"❌ Visualization failed: {response.text}")
        return None

def analyze_results(results_path: str):
    """Analyze detection results."""
    analysis_request = {
        "create_map": True
    }
    
    response = requests.post(
        f"{API_BASE_URL}/analyze",
        data={"results_path": results_path},
        json=analysis_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Analysis completed successfully")
        print(f"   Analysis: {result['analysis']}")
        print(f"   Report file: {result['report_file']}")
        return result
    else:
        print(f"❌ Analysis failed: {response.text}")
        return None

def launch_fiftyone():
    """Launch FiftyOne app."""
    response = requests.get(f"{API_BASE_URL}/fiftyone/launch")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ FiftyOne app launched successfully")
        return result
    else:
        print(f"❌ Failed to launch FiftyOne: {response.text}")
        return None

def main():
    """Main example function."""
    print("🚀 WildDetect API Example")
    print("=" * 50)
    
    # Check API status
    if not check_api_status():
        return
    
    # Get system info
    system_info = get_system_info()
    
    # Example image paths (replace with actual paths)
    example_images = [
        "assets/image.png",  # Replace with actual image paths
    ]
    
    # Filter existing images
    existing_images = [img for img in example_images if Path(img).exists()]
    
    if not existing_images:
        print("\n⚠️  No example images found. Please update the image paths in the script.")
        print("   You can use any image files for testing.")
        return
    
    print(f"\n📸 Using {len(existing_images)} images for testing")
    
    # Upload images
    upload_result = upload_images(existing_images)
    if not upload_result:
        return
    
    # Start detection job
    print("\n🔍 Starting wildlife detection...")
    detection_job_id = start_detection_job(upload_result)
    if detection_job_id:
        detection_result = monitor_job(detection_job_id)
        if detection_result and 'results' in detection_result:
            results_path = detection_result['results']['results_path']
            
            # Visualize results
            print("\n🗺️  Creating visualizations...")
            visualize_results(results_path)
            
            # Analyze results
            print("\n📊 Analyzing results...")
            analyze_results(results_path)
    
    # Start census campaign
    print("\n📋 Starting census campaign...")
    campaign_id = f"api_test_{int(time.time())}"
    census_job_id = start_census_campaign(upload_result, campaign_id)
    if census_job_id:
        monitor_job(census_job_id)
    
    # Launch FiftyOne
    print("\n🔍 Launching FiftyOne...")
    launch_fiftyone()
    
    print("\n✅ API example completed!")
    print("\n📚 API Documentation:")
    print(f"   Swagger UI: {API_BASE_URL}/docs")
    print(f"   ReDoc: {API_BASE_URL}/redoc")

if __name__ == "__main__":
    main() 