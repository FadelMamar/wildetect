"""
Main Streamlit UI for WildDetect.

This module provides a web interface for uploading images, running detection,
and visualizing results with FiftyOne integration.
"""

import json
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).parent.parent.parent


from wildetect.cli_ui_integration import cli_ui_integration
from wildetect.core.visualization.fiftyone_manager import FiftyOneManager

# Page configuration
st.set_page_config(
    page_title="WildDetect - Wildlife Detection System",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "cli_integration" not in st.session_state:
    st.session_state.cli_integration = cli_ui_integration
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "detection_results" not in st.session_state:
    st.session_state.detection_results = []


def initialize_components():
    """Initialize CLI integration and session state."""
    try:
        # Initialize session state for components
        if "fo_manager" not in st.session_state:
            st.session_state.fo_manager = None
        if "detector" not in st.session_state:
            st.session_state.detector = None
    except Exception as e:
        st.error(f"Error initializing components: {e}")


def main():
    """Main application function."""
    st.title("🦁 WildDetect - Wildlife Detection System")
    st.markdown("Semi-automated wildlife detection from aerial images")

    # Initialize components
    initialize_components()

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Model settings
        st.subheader("Model settings")
        with st.form("model_form"):
            model_path = st.text_input(
                "Model Path", value=r"D:\workspace\repos\wildetect\weights\best.pt"
            )
            roi_weights = st.text_input(
                "ROI Weights Path",
                value=r"D:\workspace\repos\wildetect\weights\roi_classifier.torchscript",
            )
            if st.form_submit_button("Load Model"):
                os.environ["WILDETECT_MODEL_PATH"] = model_path
                os.environ["ROI_MODEL_PATH"] = roi_weights
                st.success("Model updated successfully!")

        # Dataset settings
        st.subheader("Data visualization")
        with st.form("dataset_form"):
            dataset_name = st.text_input("Dataset Name", value="wildlife_detection")
            if st.form_submit_button("Create Dataset"):
                if st.session_state.fo_manager is None:
                    st.session_state.fo_manager = FiftyOneManager(dataset_name)
                    st.success("FiftyOne dataset manager initialized")
                else:
                    st.session_state.fo_manager.dataset_name = dataset_name

        if st.button("Launch FiftyOne App"):
            try:
                with st.expander("Logs"):
                    log_placeholder = st.empty()
                    result = st.session_state.cli_integration.fiftyone_ui(
                        action="launch",
                        log_placeholder=log_placeholder,
                        status_text=log_placeholder,
                    )

                if result["success"]:
                    st.success("FiftyOne app launched!")
                else:
                    st.error(f"Error launching FiftyOne: {result['error']}")
            except Exception as e:
                st.error(f"Error launching FiftyOne: {e}")

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Upload & Detect",
            "Detection Results",
            "Dataset Stats",
            "Census Campaign",
        ]
    )

    with tab1:
        upload_and_detect_tab()

    with tab2:
        results_tab()

    with tab3:
        dataset_stats_tab()

    with tab4:
        census_campaign_tab()


def upload_and_detect_tab():
    """Handle image upload and detection."""
    st.header("Upload Images & Run Detection")

    # File upload
    with st.form("upload_form"):
        uploaded_files = st.file_uploader(
            "Upload aerial images",
            type=["jpg", "jpeg", "png", "tiff", "bmp"],
            accept_multiple_files=True,
            help="Upload one or more aerial images for wildlife detection",
        )

        if st.session_state.fo_manager is None:
            st.info(
                "FiftyOne dataset manager is not initialized. Detections will not be visualized"
                ". Set the dataset name in the settings."
            )

        button = st.form_submit_button("Run Detection")
        saved_paths = []
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            for uploaded_file in uploaded_files:
                # Save to temporary location
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
                ) as tmp_file:
                    shutil.copyfileobj(uploaded_file, tmp_file)
                    saved_paths.append(tmp_file.name)

            st.session_state.saved_paths = saved_paths

        if button and saved_paths:
            # Detection button
            run_detection(saved_paths)
        if not saved_paths:
            st.warning("No images uploaded")

    with st.container():
        with st.form("batch_form"):
            st.subheader("Batch Processing")
            batch_dir = st.text_input(
                "Process Directory",
                value="data/images",
                help="Path to directory containing images to process",
            )
            button = st.form_submit_button("Process Directory")

            if button:
                if os.path.exists(batch_dir):
                    # Run your processing and store results in session_state
                    st.session_state.batch_results = st.empty()
                    dataset_name = getattr(
                        st.session_state.fo_manager, "dataset_name", None
                    )
                    run_detection([batch_dir], dataset_name=dataset_name)
                else:
                    st.error(f"Directory {batch_dir} does not exist")


def run_detection(image_paths: List[str], dataset_name: Optional[str] = None):
    """Run detection on uploaded images using CLI integration."""

    # Use CLI integration for detection
    with st.expander("Logs"):
        log_placeholder = st.empty()
        st.session_state.cli_integration.run_detection_ui(
            images=image_paths,
            dataset_name=dataset_name,
            status_text=log_placeholder,
            log_placeholder=log_placeholder,
        )


def results_tab():
    """Display detection results."""
    st.header("Detection Results")

    with st.form("results_form"):
        results_file = st.file_uploader("Upload results file (JSON)", type=["json"])
        if results_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
                tmp_file.write(results_file.getvalue())
                tmp_path = tmp_file.name
            detection_results = json.load(open(tmp_path))

        if st.form_submit_button("Display Results"):
            # Summary statistics
            st.subheader("Summary")
            total_images = len(detection_results)
            total_detections = sum(
                [data.get("total_detections", None) for data in detection_results]
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Images Processed", total_images)
            with col2:
                st.metric("Total Detections", total_detections)

            # Species breakdown
            st.subheader("Species Breakdown")
            all_species_counts = {}
            for result in detection_results:
                species_counts = result.get("class_counts", {})
                for species, count in species_counts.items():
                    all_species_counts[species] = (
                        all_species_counts.get(species, 0) + count
                    )

            if all_species_counts:
                species_df = pd.DataFrame(
                    [
                        {"Species": species, "Count": count}
                        for species, count in all_species_counts.items()
                    ]
                ).sort_values("Count", ascending=False)

                st.bar_chart(species_df.set_index("Species"))

            # Detailed results
            st.subheader("Detailed Results")
            for i, result in enumerate(detection_results):
                with st.expander(f"Image {i+1}: {result['image_path']}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(
                            f"**Total Detections:** {result.get('total_detections', 0)}"
                        )
                        st.write(f"**Species Found:**")
                        for species, count in result.get("class_counts", {}).items():
                            st.write(f"- {species}: {count}")

                    with col2:
                        if "error" in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            st.success("Detection successful")


def dataset_stats_tab():
    """Display dataset statistics."""
    st.header("Dataset Statistics")

    with st.form("dataset_stats_form"):
        dataset_name = st.text_input("Dataset Name", value="wildlife_detection")
        button = st.form_submit_button("Get Dataset Stats")

        if button:
            if st.session_state.fo_manager is None:
                st.session_state.fo_manager = FiftyOneManager(dataset_name)
                st.success("FiftyOne dataset manager initialized")
            else:
                st.session_state.fo_manager.dataset_name = dataset_name
            dataset_stats = st.session_state.fo_manager.get_annotation_stats()
            st.write(dataset_stats)

            try:
                # Dataset info
                dataset_info = st.session_state.fo_manager.get_dataset_info()
                st.subheader("Dataset Information")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dataset Name", dataset_info["name"])
                with col2:
                    st.metric("Total Samples", dataset_info["num_samples"])
                with col3:
                    st.metric("Fields", len(dataset_info["fields"]))

                # Annotation statistics
                annotation_stats = st.session_state.fo_manager.get_annotation_stats()
                st.subheader("Annotation Statistics")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Annotated Samples", annotation_stats["annotated_samples"]
                    )
                with col2:
                    st.metric("Total Detections", annotation_stats["total_detections"])
                with col3:
                    annotation_rate = (
                        (
                            annotation_stats["annotated_samples"]
                            / annotation_stats["total_samples"]
                            * 100
                        )
                        if annotation_stats["total_samples"] > 0
                        else 0
                    )
                    st.metric("Annotation Rate", f"{annotation_rate:.1f}%")

                # Species distribution
                if annotation_stats["species_counts"]:
                    st.subheader("Species Distribution")
                    species_df = pd.DataFrame(
                        [
                            {"Species": species, "Count": count}
                            for species, count in annotation_stats[
                                "species_counts"
                            ].items()
                        ]
                    ).sort_values("Count", ascending=False)

                    st.bar_chart(species_df.set_index("Species"))

            except Exception:
                st.error(f"Error getting dataset statistics: {traceback.format_exc()}")


def census_campaign_tab():
    """Display census campaign functionality."""
    st.header("Census Campaign Management")
    st.markdown("Run comprehensive wildlife census campaigns with geographic analysis")

    # Campaign Configuration
    st.subheader("Campaign Configuration")

    campaign_id = st.text_input(
        "Campaign ID",
        value="campaign_001",
        help="Unique identifier for the campaign",
    )
    # pilot_name = st.text_input(
    #    "Pilot Name", value="", help="Name of the pilot conducting the survey"
    # )

    # Target Species
    st.subheader("Target Species")
    default_species = ["wildlife"]
    target_species = st.multiselect(
        "Select target species",
        options=default_species,
        default=default_species,
        help="Species to detect in the campaign",
    )

    # Image Input
    st.subheader("Image Input")
    input_type = st.radio(
        "Input Type",
        ["Upload Images", "Directory Path"],
        help="Choose how to provide images for the campaign",
    )

    if input_type == "Upload Images":
        uploaded_images = st.file_uploader(
            "Upload campaign images",
            type=["jpg", "jpeg", "png", "tiff", "bmp"],
            accept_multiple_files=True,
            help="Upload images for the census campaign",
        )
        image_paths = []
        if uploaded_images:
            # Save uploaded files temporarily
            for uploaded_file in uploaded_images:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
                ) as tmp_file:
                    shutil.copyfileobj(uploaded_file, tmp_file)
                    image_paths.append(tmp_file.name)
    else:
        directory_path = st.text_input(
            "Image Directory Path",
            value="data/images",
            help="Path to directory containing campaign images",
        )
        image_paths = [directory_path] if directory_path else []

    # Campaign Execution
    st.subheader("Run Campaign")

    if st.button("Start Census Campaign", type="primary"):
        if not image_paths:
            st.error("Please provide images for the campaign")
        elif not campaign_id:
            st.error("Please provide a campaign ID")
        else:
            with st.spinner("Running census campaign..."):
                with st.expander("Logs"):
                    log_placeholder = st.empty()
                    campaign_result = st.session_state.cli_integration.run_census_ui(
                        campaign_id=campaign_id,
                        images=image_paths,
                        target_species=target_species,
                        create_map=True,
                        log_placeholder=log_placeholder,
                        status_text=log_placeholder,
                    )

            if campaign_result["success"]:
                st.success("Census campaign completed successfully!")

                # Display campaign results
                campaign_manager = campaign_result["campaign_manager"]
                results = campaign_result["results"]

                st.write(f"**Campaign ID:** {campaign_id}")
                st.write(
                    f"**Total Images:** {len(campaign_manager.census_manager.image_paths)}"
                )
                st.write(
                    f"**Drone Images Created:** {len(campaign_manager.census_manager.drone_images)}"
                )

                if "statistics" in results:
                    stats = results["statistics"]
                    if "flight_analysis" in stats:
                        flight_stats = stats["flight_analysis"]
                        st.write(
                            f"**Images with GPS:** {flight_stats.get('num_images_with_gps', 0)}"
                        )
                        st.write(
                            f"**Total Waypoints:** {flight_stats.get('total_waypoints', 0)}"
                        )
                        st.write(
                            f"**Total Distance:** {flight_stats.get('total_distance_km', 0):.2f} km"
                        )

                # Detection statistics
                if campaign_manager.census_manager.drone_images:
                    detection_stats = get_detection_statistics(
                        campaign_manager.census_manager.drone_images
                    )
                    if detection_stats["total_detections"] > 0:
                        st.write("**Detection Results:**")
                        st.write(
                            f"- Total detections: {detection_stats['total_detections']}"
                        )
                        st.write(
                            f"- Species detected: {len(detection_stats['species_counts'])}"
                        )

                        if detection_stats["species_counts"]:
                            st.write("**Species breakdown:**")
                            for species, count in detection_stats[
                                "species_counts"
                            ].items():
                                st.write(f"- {species}: {count}")
            else:
                st.error(f"Campaign failed: {campaign_result['error']}")


def get_detection_statistics(drone_images: List) -> dict:
    """Calculate detection statistics from drone images."""
    total_detections = 0
    species_counts = {}

    for drone_image in drone_images:
        detections = drone_image.get_all_predictions()
        total_detections += len(detections)

        for detection in detections:
            if not detection.is_empty:
                species = detection.class_name
                species_counts[species] = species_counts.get(species, 0) + 1

    return {
        "total_detections": total_detections,
        "species_counts": species_counts,
    }


if __name__ == "__main__":
    main()
