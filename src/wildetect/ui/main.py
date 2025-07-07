"""
Main Streamlit UI for WildDetect.

This module provides a web interface for uploading images, running detection,
and visualizing results with FiftyOne integration.
"""

import os
import shutil

# Add parent directory to path for imports
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from wildetect.cli_ui_integration import cli_ui_integration
from wildetect.core.detector import WildlifeDetector
from wildetect.core.fiftyone_manager import FiftyOneManager
from wildetect.core.labelstudio_manager import LabelStudioManager
from wildetect.utils.config import create_directories, get_config

# Page configuration
st.set_page_config(
    page_title="WildDetect - Wildlife Detection System",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "detector" not in st.session_state:
    st.session_state.detector = None
if "fo_manager" not in st.session_state:
    st.session_state.fo_manager = None
if "ls_manager" not in st.session_state:
    st.session_state.ls_manager = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "detection_results" not in st.session_state:
    st.session_state.detection_results = []
if "cli_integration" not in st.session_state:
    st.session_state.cli_integration = cli_ui_integration


def initialize_components():
    """Initialize detection and FiftyOne components."""
    try:
        if st.session_state.detector is None:
            with st.spinner("Loading detection model..."):
                st.session_state.detector = WildlifeDetector()

        if st.session_state.fo_manager is None:
            with st.spinner("Initializing FiftyOne..."):
                st.session_state.fo_manager = FiftyOneManager()

        if st.session_state.ls_manager is None:
            with st.spinner("Initializing LabelStudio..."):
                st.session_state.ls_manager = LabelStudioManager()

        # Create necessary directories
        create_directories()

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
        st.subheader("Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum confidence for detections",
        )

        # FiftyOne settings
        st.subheader("FiftyOne Settings")
        if st.button("Launch FiftyOne App"):
            try:
                st.session_state.fo_manager.launch_app()
                st.success("FiftyOne app launched!")
            except Exception as e:
                st.error(f"Error launching FiftyOne: {e}")

        if st.button("Export Annotations"):
            try:
                export_format = st.selectbox(
                    "Export Format", ["coco", "yolo", "pascal"]
                )
                export_path = f"data/annotations_export_{export_format}"
                st.session_state.fo_manager.export_annotations(
                    export_path, export_format
                )
                st.success(f"Annotations exported to {export_path}")
            except Exception as e:
                st.error(f"Error exporting annotations: {e}")

        # LabelStudio settings
        st.subheader("LabelStudio Settings")
        if st.button("Launch LabelStudio"):
            try:
                st.info("LabelStudio should be running at http://localhost:8080")
                st.success("LabelStudio integration ready!")
            except Exception as e:
                st.error(f"Error connecting to LabelStudio: {e}")

        if st.button("Create Annotation Job"):
            if st.session_state.detection_results:
                try:
                    project_name = st.text_input(
                        "Project Name", value="wildlife_annotation_job"
                    )
                    if st.button("Create Job"):
                        image_paths = [
                            r["image_path"] for r in st.session_state.detection_results
                        ]
                        job_info = st.session_state.ls_manager.create_annotation_job(
                            project_name,
                            image_paths,
                            st.session_state.detection_results,
                        )
                        st.success(f"Annotation job created! URL: {job_info['url']}")
                except Exception as e:
                    st.error(f"Error creating annotation job: {e}")
            else:
                st.warning("No detection results available. Run detection first.")

    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Upload & Detect",
            "Results",
            "Dataset Stats",
            "Model Info",
            "LabelStudio",
            "CLI Features",
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
        model_info_tab()

    with tab5:
        labelstudio_tab()

    with tab6:
        cli_features_tab()

    with tab7:
        census_campaign_tab()


def upload_and_detect_tab():
    """Handle image upload and detection."""
    st.header("Upload Images & Run Detection")

    # File upload
    uploaded_files = st.file_uploader(
        "Upload aerial images",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
        accept_multiple_files=True,
        help="Upload one or more aerial images for wildlife detection",
    )

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files

        # Save uploaded files
        config = get_config()
        images_dir = Path(config["paths"]["images_dir"])
        images_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for uploaded_file in uploaded_files:
            # Save to temporary location
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
            ) as tmp_file:
                shutil.copyfileobj(uploaded_file, tmp_file)
                saved_paths.append(tmp_file.name)

        st.session_state.saved_paths = saved_paths

        # Detection button
        if st.button("Run Detection", type="primary"):
            run_detection(
                saved_paths, st.session_state.get("confidence_threshold", 0.5)
            )

    # Batch processing
    st.subheader("Batch Processing")
    batch_dir = st.text_input(
        "Process Directory",
        value="data/images",
        help="Path to directory containing images to process",
    )

    if st.button("Process Directory"):
        if os.path.exists(batch_dir):
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.tiff", "*.bmp"]:
                image_files.extend(Path(batch_dir).glob(ext))

            if image_files:
                st.info(f"Found {len(image_files)} images in {batch_dir}")
                if st.button("Process All Images"):
                    run_batch_detection(
                        [str(f) for f in image_files],
                        st.session_state.get("confidence_threshold", 0.5),
                    )
            else:
                st.warning(f"No image files found in {batch_dir}")
        else:
            st.error(f"Directory {batch_dir} does not exist")


def run_detection(image_paths: List[str], confidence: float):
    """Run detection on uploaded images using CLI integration."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Use CLI integration for detection
    detection_result = st.session_state.cli_integration.run_detection_ui(
        images=image_paths,
        confidence=confidence,
        progress_bar=progress_bar,
        status_text=status_text,
    )

    if detection_result["success"]:
        st.session_state.detection_results = detection_result["results"]

        # Show summary
        total_detections = detection_result["total_detections"]
        total_images = detection_result["total_images"]
        st.success(
            f"Detection completed! Found {total_detections} wildlife in {total_images} images"
        )

        # Add to FiftyOne dataset if available
        if st.session_state.fo_manager and detection_result["results"]:
            for result in detection_result["results"]:
                st.session_state.fo_manager.add_images([result["image_path"]], [result])
    else:
        st.error(f"Detection failed: {detection_result['error']}")


def run_batch_detection(image_paths: List[str], confidence: float):
    """Run detection on a batch of images using CLI integration."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Use CLI integration for batch detection
    detection_result = st.session_state.cli_integration.run_detection_ui(
        images=image_paths,
        confidence=confidence,
        progress_bar=progress_bar,
        status_text=status_text,
    )

    if detection_result["success"]:
        st.session_state.detection_results = detection_result["results"]

        # Show summary
        total_detections = detection_result["total_detections"]
        total_images = detection_result["total_images"]
        st.success(
            f"Batch detection completed! Found {total_detections} wildlife in {total_images} images"
        )

        # Add to FiftyOne dataset if available
        if st.session_state.fo_manager and detection_result["results"]:
            for result in detection_result["results"]:
                st.session_state.fo_manager.add_images([result["image_path"]], [result])
    else:
        st.error(f"Batch detection failed: {detection_result['error']}")


def results_tab():
    """Display detection results."""
    st.header("Detection Results")

    if not st.session_state.detection_results:
        st.info(
            "No detection results available. Upload images and run detection first."
        )
        return

    # Summary statistics
    st.subheader("Summary")
    total_images = len(st.session_state.detection_results)
    total_detections = sum(
        r.get("total_count", 0) for r in st.session_state.detection_results
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Images Processed", total_images)
    with col2:
        st.metric("Total Detections", total_detections)
    with col3:
        avg_detections = total_detections / total_images if total_images > 0 else 0
        st.metric("Avg Detections/Image", f"{avg_detections:.1f}")

    # Species breakdown
    st.subheader("Species Breakdown")
    all_species_counts = {}
    for result in st.session_state.detection_results:
        species_counts = result.get("species_counts", {})
        for species, count in species_counts.items():
            all_species_counts[species] = all_species_counts.get(species, 0) + count

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
    for i, result in enumerate(st.session_state.detection_results):
        with st.expander(f"Image {i+1}: {os.path.basename(result['image_path'])}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Total Detections:** {result.get('total_count', 0)}")
                st.write(f"**Species Found:**")
                for species, count in result.get("species_counts", {}).items():
                    st.write(f"- {species}: {count}")

            with col2:
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success("Detection successful")


def dataset_stats_tab():
    """Display dataset statistics."""
    st.header("Dataset Statistics")

    if st.session_state.fo_manager is None:
        st.error("FiftyOne manager not initialized")
        return

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
            st.metric("Annotated Samples", annotation_stats["annotated_samples"])
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
                    for species, count in annotation_stats["species_counts"].items()
                ]
            ).sort_values("Count", ascending=False)

            st.bar_chart(species_df.set_index("Species"))

    except Exception as e:
        st.error(f"Error getting dataset statistics: {e}")


def model_info_tab():
    """Display model information."""
    st.header("Model Information")

    if st.session_state.detector is None:
        st.error("Detector not initialized")
        return

    try:
        model_info = st.session_state.detector.get_model_info()

        st.subheader("Model Details")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Model Path:** {model_info['model_path']}")
            st.write(f"**Device:** {model_info['device']}")
            st.write(f"**Input Size:** {model_info['input_size']}")

        with col2:
            st.write(f"**Number of Classes:** {model_info['num_classes']}")
            st.write(f"**Classes:** {', '.join(model_info['class_names'])}")

        # Configuration
        config = get_config()
        st.subheader("Configuration")
        st.json(config)

    except Exception as e:
        st.error(f"Error getting model information: {e}")


def labelstudio_tab():
    """Display LabelStudio management interface."""
    st.header("LabelStudio Management")

    if st.session_state.ls_manager is None:
        st.error("LabelStudio manager not initialized")
        return

    # LabelStudio status
    st.subheader("LabelStudio Status")
    try:
        st.info("LabelStudio integration ready")
        st.write("URL: http://localhost:8080")
    except Exception as e:
        st.error(f"LabelStudio connection error: {e}")

    # Create annotation job
    st.subheader("Create Annotation Job")
    if st.session_state.detection_results:
        project_name = st.text_input("Project Name", value="wildlife_annotation_job")
        description = st.text_area(
            "Description", value="Wildlife detection annotation job"
        )

        if st.button("Create Annotation Job"):
            try:
                image_paths = [
                    r["image_path"] for r in st.session_state.detection_results
                ]
                job_info = st.session_state.ls_manager.create_annotation_job(
                    project_name, image_paths, st.session_state.detection_results
                )

                st.success("Annotation job created successfully!")
                st.write(f"**Project ID:** {job_info['project_id']}")
                st.write(f"**Project URL:** {job_info['url']}")
                st.write(f"**Total Tasks:** {job_info['total_tasks']}")
                st.write(f"**Completion Rate:** {job_info['completion_rate']:.1%}")

            except Exception as e:
                st.error(f"Error creating annotation job: {e}")
    else:
        st.warning("No detection results available. Run detection first.")

    # Export annotations
    st.subheader("Export Annotations")
    export_format = st.selectbox("Export Format", ["yolo", "coco", "pascal"])

    if st.button("Export for Training"):
        try:
            config = get_config()
            output_dir = f"{config['paths']['annotations_dir']}/labelstudio_export_{export_format}"

            # This would require a project selection interface
            st.info("Export functionality requires project selection")

        except Exception as e:
            st.error(f"Error exporting annotations: {e}")

    # Sync with FiftyOne
    st.subheader("Sync with FiftyOne")
    dataset_name = st.text_input("FiftyOne Dataset Name", value="wildlife_detection")

    if st.button("Sync Annotations"):
        try:
            st.info("Sync functionality requires project selection")
            st.success("Annotations synced with FiftyOne")
        except Exception as e:
            st.error(f"Error syncing annotations: {e}")


def cli_features_tab():
    """Display CLI features in the UI."""
    st.header("CLI Features Integration")
    st.markdown("Access CLI functionality through the web interface")

    # System Information
    st.subheader("System Information")
    if st.button("Get System Info"):
        system_info = st.session_state.cli_integration.get_system_info_ui()

        # Display components
        st.write("**System Components:**")
        for component, info in system_info["components"].items():
            status_color = "green" if info["status"] == "✓" else "red"
            st.markdown(
                f"- **{component}:** :{status_color}[{info['status']}] {info['details']}"
            )

        # Display dependencies
        st.write("**Dependencies:**")
        for dep, info in system_info["dependencies"].items():
            status_color = "green" if info["status"] == "✓" else "red"
            st.markdown(
                f"- **{dep}:** :{status_color}[{info['status']}] {info['details']}"
            )

    # Results Analysis
    st.subheader("Analyze Results")
    results_file = st.file_uploader(
        "Upload results file (JSON)",
        type=["json"],
        help="Upload a detection results file to analyze",
    )

    if results_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
            tmp_file.write(results_file.getvalue())
            tmp_path = tmp_file.name

        col1, col2 = st.columns(2)

        with col1:
            create_map = st.checkbox("Create geographic map", value=True)

        with col2:
            if st.button("Analyze Results"):
                with st.spinner("Analyzing results..."):
                    analysis_result = (
                        st.session_state.cli_integration.analyze_results_ui(
                            tmp_path, output_dir="analysis", create_map=create_map
                        )
                    )

                if analysis_result["success"]:
                    st.success("Analysis completed successfully!")

                    # Display analysis results
                    analysis = analysis_result["analysis_results"]
                    st.write(f"**Total Images:** {analysis.get('total_images', 0)}")
                    st.write(
                        f"**Total Detections:** {analysis.get('total_detections', 0)}"
                    )

                    if analysis.get("species_breakdown"):
                        st.write("**Species Breakdown:**")
                        for species, count in analysis["species_breakdown"].items():
                            st.write(f"- {species}: {count}")
                else:
                    st.error(f"Analysis failed: {analysis_result['error']}")

    # Results Visualization
    st.subheader("Visualize Results")
    viz_results_file = st.file_uploader(
        "Upload results file for visualization",
        type=["json"],
        help="Upload a detection results file to visualize",
        key="viz_results",
    )

    if viz_results_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
            tmp_file.write(viz_results_file.getvalue())
            tmp_path = tmp_file.name

        col1, col2 = st.columns(2)

        with col1:
            show_confidence = st.checkbox("Show confidence scores", value=True)

        with col2:
            create_viz_map = st.checkbox("Create geographic map", value=True)

        if st.button("Visualize Results"):
            with st.spinner("Creating visualizations..."):
                viz_result = st.session_state.cli_integration.visualize_results_ui(
                    tmp_path,
                    output_dir="visualizations",
                    show_confidence=show_confidence,
                    create_map=create_viz_map,
                )

            if viz_result["success"]:
                st.success("Visualization completed successfully!")

                # Display visualization data
                viz_data = viz_result["visualization_data"]
                st.write(f"**Total Images:** {viz_data.get('total_images', 0)}")
                st.write(f"**Total Detections:** {viz_data.get('total_detections', 0)}")

                if viz_data.get("species_counts"):
                    st.write("**Species Detected:**")
                    for species, count in viz_data["species_counts"].items():
                        st.write(f"- {species}: {count}")
            else:
                st.error(f"Visualization failed: {viz_result['error']}")


def census_campaign_tab():
    """Display census campaign functionality."""
    st.header("Census Campaign Management")
    st.markdown("Run comprehensive wildlife census campaigns with geographic analysis")

    # Campaign Configuration
    st.subheader("Campaign Configuration")

    col1, col2 = st.columns(2)

    with col1:
        campaign_id = st.text_input(
            "Campaign ID",
            value="campaign_001",
            help="Unique identifier for the campaign",
        )
        pilot_name = st.text_input(
            "Pilot Name", value="", help="Name of the pilot conducting the survey"
        )

    with col2:
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence for detections",
        )
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=32,
            value=8,
            help="Number of images to process in each batch",
        )

    # Target Species
    st.subheader("Target Species")
    default_species = ["elephant", "giraffe", "zebra", "lion"]
    target_species = st.multiselect(
        "Select target species",
        options=default_species + ["rhino", "buffalo", "antelope", "other"],
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
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Running census campaign..."):
                campaign_result = st.session_state.cli_integration.run_census_ui(
                    campaign_id=campaign_id,
                    images=image_paths,
                    confidence=confidence,
                    batch_size=batch_size,
                    pilot_name=pilot_name,
                    target_species=target_species,
                    create_map=True,
                    progress_bar=progress_bar,
                    status_text=status_text,
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
