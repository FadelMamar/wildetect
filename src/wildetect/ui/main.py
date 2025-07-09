"""
Main Streamlit UI for WildDetect.

This module provides a web interface for uploading images, running detection,
and visualizing results with FiftyOne integration.
"""

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import List, Optional

import mlflow
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from wildetect.cli_ui_integration import cli_ui_integration
from wildetect.core.config import ROOT
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
        if "temp_dir" not in st.session_state:
            temp_dir = tempfile.TemporaryDirectory()
            st.session_state.temp_dir = temp_dir
            atexit.register(temp_dir.cleanup)
        if "alias_options" not in st.session_state:
            st.session_state.alias_options = []
        if "selected_alias" not in st.session_state:
            st.session_state.selected_alias = None
    except Exception:
        st.error(f"Error initializing components: {traceback.format_exc()}")


def main():
    """Main application function."""
    st.title("🦁 WildDetect - Wildlife Detection System")
    st.markdown("Semi-automated wildlife detection from aerial images")

    load_dotenv(ROOT / ".env", override=False)

    # Initialize components
    initialize_components()

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        st.subheader("MLflow Server")
        launch_mlflow()

        # Model settings
        st.subheader("Model settings")
        model_settings_tab()

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

    return None


def launch_mlflow():
    """Launch MLflow server."""

    if st.button("Launch MLflow Server"):
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_CONSOLE
            path = str(ROOT / "scripts/launch_mlflow.bat")
            # st.write(path)
            subprocess.Popen([path], creationflags=creationflags)
        else:
            raise NotImplementedError(
                "MLflow server is not supported on this platform."
            )
        st.success("MLflow server launched!")
    return None


def get_registered_model_names():
    """Return a list of all registered model names from MLflow."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    st.write("Mlflow server:", tracking_uri)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    try:
        models = mlflow.MlflowClient().search_registered_models()
        return [m.name for m in models]
    except Exception as e:
        st.error(f"Error fetching registered model names: {e}")
        return []


def get_model_versions_and_aliases(model_name):
    """Return (model_versions, alias_to_version, version_to_aliases) for a model."""
    model_versions = []
    alias_to_version = {}
    version_to_aliases = {}
    try:
        versions = mlflow.MlflowClient().search_model_versions(f"name='{model_name}'")
        for v in versions:
            model_versions.append(str(v.version))
            for alias in getattr(v, "aliases", []):
                alias_to_version[alias] = str(v.version)
                version_to_aliases.setdefault(str(v.version), []).append(alias)
    except Exception as e:
        st.error(f"Error fetching model versions/aliases: {e}")
    return model_versions, alias_to_version, version_to_aliases


def model_settings_tab():
    """Handle model settings."""
    st.info("Set the model to use for detection. This will be used for all detections.")
    with st.form("model_settings_form"):
        registry_name = st.text_input("Registry Name", value="labeler").strip()
        roi_weights = st.text_input(
            "ROI Weights",
            value=r"D:\workspace\repos\wildetect\weights\roi_classifier.torchscript",
        ).strip()

        if st.form_submit_button("View available models") and registry_name:
            (
                model_versions,
                alias_to_version,
                version_to_aliases,
            ) = get_model_versions_and_aliases(registry_name)
            alias_options = list(alias_to_version.keys())
            st.session_state.alias_options = alias_options
            st.session_state.selected_alias = None

            if st.session_state.alias_options:
                st.session_state.selected_alias = st.selectbox(
                    "Select Model Alias", st.session_state.alias_options
                )
            else:
                st.warning("No versions or aliases found for this model name.")
                return

    if st.button(
        "Set Model",
        disabled=st.session_state.selected_alias is None or not registry_name,
    ):
        os.environ["MLFLOW_MODEL_NAME"] = registry_name
        os.environ["MLFLOW_MODEL_ALIAS"] = st.session_state.selected_alias
        st.success("Model set successfully")

        if roi_weights:
            if not os.path.exists(roi_weights):
                st.error(f"ROI weights file not found: {roi_weights}")
                return
            os.environ["ROI_MODEL_PATH"] = roi_weights
            st.success("ROI weights set successfully")
        else:
            st.warning("No ROI weights provided")


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
            temp_dir = st.session_state.temp_dir.name
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(file_path)

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
                value=r"D:\workspace\data\savmap_dataset_v2\raw\tmp",
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
        value=f"campaign_{uuid.uuid4()}",
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
        [
            # "Upload Images",
            "Directory Path"
        ],
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
            temp_dir = st.session_state.temp_dir.name
            # Save uploaded files temporarily
            for uploaded_file in uploaded_images:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image_paths.append(file_path)

            if isinstance(st.session_state.saved_paths, list):
                st.session_state.saved_paths.extend(image_paths)
            else:
                st.session_state.saved_paths = image_paths
    else:
        directory_path = st.text_input(
            "Image Directory Path",
            value=r"D:\workspace\data\savmap_dataset_v2\raw\tmp",
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
                        log_placeholder=log_placeholder,
                        status_text=log_placeholder,
                    )

            if campaign_result["success"]:
                st.success("Census campaign completed successfully!")

                # Display campaign results
            #    results = campaign_result["results"]
            #    st.write(f"**Campaign ID:** {campaign_id}")

            #    if "statistics" in results:
            #        stats = results["statistics"]
            #        if "flight_analysis" in stats:
            #        flight_stats = stats["flight_analysis"]
            #        st.write(
            #            f"**Images with GPS:** {flight_stats.get('num_images_with_gps', 0)}"
            #        )
            #        st.write(
            #            f"**Total Waypoints:** {flight_stats.get('total_waypoints', 0)}"
            #        )
            #        st.write(
            #            f"**Total Distance:** {flight_stats.get('total_distance_km', 0):.2f} km"
            #        )


if __name__ == "__main__":
    main()
