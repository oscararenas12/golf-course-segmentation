"""
Local testing script for the Golf Course Segmentation API.

Run this to test the API locally before deploying to HF Spaces.
"""

import os
import sys
import base64
import requests
from pathlib import Path

# Set model path for local testing
os.environ["MODEL_PATH"] = str(Path(__file__).parent.parent / "models" / "best_unet_resnet50_large.keras")

API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_segment_file(image_path: str):
    """Test file upload segmentation."""
    print(f"\nTesting file upload with {image_path}...")

    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return False

    with open(image_path, "rb") as f:
        response = requests.post(
            f"{API_URL}/segment",
            files={"file": f}
        )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Message: {result['message']}")
        print("Statistics:")
        for cls, stats in result['statistics'].items():
            print(f"  {cls}: {stats['percentage']:.1f}%")

        # Save overlay image
        overlay_data = result['overlayData'].split(",")[1]
        overlay_bytes = base64.b64decode(overlay_data)
        output_path = image_path.replace(".jpg", "_segmented.png").replace(".jpeg", "_segmented.png")
        with open(output_path, "wb") as f:
            f.write(overlay_bytes)
        print(f"Saved overlay to: {output_path}")
        return True
    else:
        print(f"Error: {response.json()}")
        return False


def test_segment_base64(image_path: str):
    """Test base64 segmentation."""
    print(f"\nTesting base64 input with {image_path}...")

    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return False

    with open(image_path, "rb") as f:
        image_bytes = f.read()
        b64 = base64.b64encode(image_bytes).decode()

    response = requests.post(
        f"{API_URL}/segment-base64",
        json={"imageData": f"data:image/jpeg;base64,{b64}"}
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print("Statistics:")
        for cls, stats in result['statistics'].items():
            print(f"  {cls}: {stats['percentage']:.1f}%")
        return True
    else:
        print(f"Error: {response.json()}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Golf Course Segmentation API - Local Tests")
    print("=" * 50)
    print(f"API URL: {API_URL}")
    print(f"Model Path: {os.environ.get('MODEL_PATH')}")
    print("=" * 50)

    # Check if API is running
    try:
        requests.get(f"{API_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print("\nError: API is not running!")
        print("Start the API first with:")
        print("  cd api && uvicorn main:app --reload --port 8000")
        sys.exit(1)

    # Run tests
    test_health()

    # Test with an image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_segment_file(image_path)
        test_segment_base64(image_path)
    else:
        print("\nTo test with an image, run:")
        print("  python test_local.py /path/to/golf_image.jpg")
