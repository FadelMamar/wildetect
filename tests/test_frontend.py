"""
Simple tests for WildDetect Reflex frontend.
"""

import pytest
import reflex as rx
from wildetect.ui.frontend.state import WildDetectState


def test_state_initialization():
    """Test that the state initializes correctly."""
    state = WildDetectState()

    # Check default values
    assert state.api_base_url == "http://localhost:8000"
    assert state.uploaded_files == []
    assert state.upload_progress == 0
    assert state.upload_status == ""
    assert state.detection_jobs == {}
    assert state.census_jobs == {}
    assert state.loading == False
    assert state.error_message == ""


def test_state_methods_exist():
    """Test that all required state methods exist."""
    state = WildDetectState()

    # Check that async methods exist
    assert hasattr(state, "get_system_info")
    assert hasattr(state, "upload_files")
    assert hasattr(state, "start_detection")
    assert hasattr(state, "start_census")
    assert hasattr(state, "check_job_status")
    assert hasattr(state, "poll_jobs")
    assert hasattr(state, "launch_fiftyone")
    assert hasattr(state, "get_fiftyone_datasets")


def test_app_creation():
    """Test that the Reflex app can be created."""
    from .app import app

    # Check that app is a Reflex app
    assert isinstance(app, rx.App)


def test_page_components():
    """Test that page components can be created."""
    from .pages.index import census_panel, index, jobs_panel, upload_panel

    # Check that components are Reflex components
    assert callable(index)
    assert callable(upload_panel)
    assert callable(jobs_panel)
    assert callable(census_panel)


if __name__ == "__main__":
    # Run basic tests
    print("Running WildDetect frontend tests...")

    try:
        test_state_initialization()
        print("✓ State initialization test passed")

        test_state_methods_exist()
        print("✓ State methods test passed")

        test_app_creation()
        print("✓ App creation test passed")

        test_page_components()
        print("✓ Page components test passed")

        print("\nAll tests passed! 🎉")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise
