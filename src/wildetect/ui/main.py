"""
Main Streamlit UI for WildDetect.

This module provides a web interface for uploading images, running detection,
and visualizing results with FiftyOne integration.
"""

import atexit
import json
import os
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

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
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "detection_results" not in st.session_state:
    st.session_state.detection_results = []


def run_cli_command(
    command: str,
    args: List[str],
    log_placeholder: Optional["st.empty"] = None,
) -> None:
    """Run a CLI command through subprocess and return results."""
    load_dotenv(ROOT / ".env", override=False)
    try:
        cmd = ["wildetect", command] + args

        if log_placeholder:
            log_placeholder.code(f"Running command: {' '.join(cmd)}")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
            universal_newlines=True,
        )
        logs = ""
        if log_placeholder is not None:
            for line in process.stdout:
                logs += line
                log_placeholder.code(logs)

        process.stdout.close()
        return_code = process.wait()

    except Exception:
        st.error(f"Subprocess error: {traceback.format_exc()}")

    return None


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
        if "roi_alias_options" not in st.session_state:
            st.session_state.roi_alias_options = []
        if "roi_selected_alias" not in st.session_state:
            st.session_state.roi_selected_alias = None

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

        st.divider()
        st.subheader("Servers")

        st.warning("Mlflow is used to load registered models.")
        launch_mlflow()

        # st.subheader("LabelStudio Server")
        st.warning("LabelStudio is used to manually correct the detections.")
        launch_labelstudio()

        # st.subheader("LabelStudio Server")
        st.warning("FiftyOne is used to visualize the images and detections.")
        launch_fiftyone()

        st.divider()
        # Model settings
        st.subheader("Model settings")
        model_settings_tab()

        st.divider()

        st.divider()
        st.subheader("Launch annotation job")
        with st.form("launch_job_form"):
            dataset_name = st.text_input("Dataset Name", value="campaing-000")
            annot_key = st.text_input(
                "Annotation Key",
                value=f"ls_review_{dataset_name.replace('-','_').replace(' ','_')}",
            )
            if st.form_submit_button("Launch Job"):
                annot_key = annot_key.strip() if len(annot_key.strip()) > 0 else None
                launch_job(dataset_name.strip(), annot_key=annot_key)

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Detect",
            "Detection Results",
            "Dataset Stats",
            "Census Campaign",
        ]
    )

    with tab1:
        detect_tab()

    with tab2:
        results_tab()

    with tab3:
        dataset_stats_tab()

    with tab4:
        census_campaign_tab()

    return None


def launch_fiftyone():
    if st.button("Launch FiftyOne"):
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_CONSOLE
            path = str(ROOT / "scripts/launch_fiftyone.bat")
            subprocess.Popen([path], creationflags=creationflags, cwd=str(ROOT))
        else:
            raise NotImplementedError(
                "FiftyOne server is not supported on this platform."
            )
        st.success("FiftyOne server launched!")


def launch_mlflow():
    """Launch MLflow server."""

    if st.button("Launch MLflow"):
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


def launch_labelstudio():
    """Launch LabelStudio"""
    if st.button("Launch LabelStudio"):
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_CONSOLE
            path = str(ROOT / "scripts/launch_labelstudio.bat")
            subprocess.Popen([path], creationflags=creationflags, cwd=str(ROOT))
        else:
            raise NotImplementedError(
                "LabelStudio server is not supported on this platform."
            )
        st.success("LabelStudio server launched!")


def launch_job(dataset_name: str, annot_key: Optional[str] = None):
    if annot_key is None:
        annot_key = f"ls_review_{dataset_name.replace('-','_').replace(' ','_')}"
    FiftyOneManager(dataset_name).send_predictions_to_labelstudio(
        annot_key, dotenv_path=str(ROOT / ".env")
    )
    st.success(
        f"Annotation job '{annot_key}' launched for dataset '{dataset_name}' to Label Studio"
    )


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
    return alias_to_version


def model_settings_tab():
    """Handle model settings."""
    st.info("Set the model to use for detection. This will be used for all detections.")
    with st.form("model_settings_form"):
        registry_name = st.text_input("Registry Name", value="labeler").strip()
        roi_registry_name = st.text_input(
            "ROI Registry Name", value="classifier"
        ).strip()

        if st.form_submit_button("View available models") and registry_name:
            alias_to_version = get_model_versions_and_aliases(registry_name)
            roi_alias_to_version = get_model_versions_and_aliases(roi_registry_name)

            alias_options = list(alias_to_version.keys())
            roi_alias_options = list(roi_alias_to_version.keys())
            st.session_state.alias_options = alias_options
            st.session_state.roi_alias_options = roi_alias_options
            st.session_state.selected_alias = None
            st.session_state.roi_selected_alias = None

            if st.session_state.alias_options:
                st.session_state.selected_alias = st.selectbox(
                    "Select Detector Model Alias", st.session_state.alias_options
                )
            else:
                st.warning("No versions or aliases found for Detector model name.")
                return

            if st.session_state.roi_alias_options:
                st.session_state.roi_selected_alias = st.selectbox(
                    "Select ROI Classifier Model Alias",
                    st.session_state.roi_alias_options,
                )
            else:
                st.warning(
                    "No versions or aliases found for ROI Classifier model name."
                )
                return

    if st.button(
        "Set Model",
        disabled=st.session_state.selected_alias is None or not registry_name,
    ):
        os.environ["MLFLOW_DETECTOR_NAME"] = registry_name
        os.environ["MLFLOW_DETECTOR_ALIAS"] = st.session_state.selected_alias or ""

        os.environ["MLFLOW_ROI_NAME"] = roi_registry_name
        os.environ["MLFLOW_ROI_ALIAS"] = st.session_state.roi_selected_alias or ""

        st.success("Model set successfully")


def detect_tab():
    """Handle image upload and detection."""
    st.header("Run Detection")

    # File upload
    with st.form("batch_form"):
        config = st.text_input(
            "Config Path",
            value=r"config/detection.yaml",
            help="Path to configuration file",
        )
        button = st.form_submit_button("Run Detection")

        if button:
            if os.path.exists(config) and os.path.isfile(config):
                with st.expander("Logs"):
                    log_placeholder = st.empty()
                    run_cli_command(
                        command="detection",
                        args=["detect", "-c", config],
                        log_placeholder=log_placeholder,
                    )
            else:
                st.error(f"File {config} does not exist")


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

    with st.form("census_campaign_form"):
        config = st.text_input(
            "Config Path",
            value=r"config/census.yaml",
            help="Path to configuration file",
        )
        button = st.form_submit_button("Start Census Campaign")

    if button:
        if os.path.exists(config) and os.path.isfile(config):
            with st.spinner("Running census campaign..."):
                with st.expander("Logs"):
                    log_placeholder = st.empty()
                    run_cli_command(
                        command="detection",
                        args=["census", "-c", config],
                        log_placeholder=log_placeholder,
                    )
        else:
            st.error(f"File {config} does not exist")
            return


if __name__ == "__main__":
    main()
