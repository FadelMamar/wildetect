"""Streamlit UI for WildTrain - Modular Computer Vision Framework."""

import streamlit as st
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import json
from datetime import datetime
import subprocess
import threading
import queue
import time

# Add the src directory to the path so we can import wildtrain modules

from wildtrain.cli import (
    train_classifier,
    get_dataset_stats,
    run_detection_pipeline,
    run_classification_pipeline,
    visualize_classifier_predictions,
    visualize_detector_predictions,
    evaluate_detector,
    evaluate_classifier,
)
from wildtrain.utils.logging import ROOT


def setup_page_config():
    """Setup Streamlit page configuration."""
    st.set_page_config(
        page_title="WildTrain - Computer Vision Framework",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def launch_fiftyone_ui():
    """Launch the FiftyOne UI."""
    button = st.button("Launch FiftyOne UI")
    if button:
        subprocess.Popen(["uv", "run", "fiftyone", "app", "launch"],creationflags=subprocess.CREATE_NEW_CONSOLE)

def get_config_files() -> Dict[str, list]:
    """Get available configuration files from the configs directory."""
    configs_dir = ROOT / "configs"
    config_files = {
        "classification": [],
        "detection": [],
        "evaluation": [],
        "visualization": [],
    }
    
    if configs_dir.exists():
        # Classification configs
        classification_dir = configs_dir / "classification"
        if classification_dir.exists():
            config_files["classification"] = [
                str(f) for f in classification_dir.glob("*.yaml")
            ]
        
        # Detection configs
        detection_dir = configs_dir / "detection"
        if detection_dir.exists():
            config_files["detection"] = [
                str(f) for f in detection_dir.glob("*.yaml")
            ]
            # Also check subdirectories
            for subdir in detection_dir.iterdir():
                if subdir.is_dir():
                    config_files["detection"].extend([
                        str(f) for f in subdir.glob("*.yaml")
                    ])
    
    return config_files


def run_command_with_progress(command_func, *args, **kwargs):
    """Run a command with progress tracking."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create a queue for communication between threads
    message_queue = queue.Queue()
    
    def run_command():
        try:
            # Capture stdout/stderr
            import io
            import contextlib
            
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                result = command_func(*args, **kwargs)
            
            message_queue.put(("success", output.getvalue(), result))
        except Exception as e:
            message_queue.put(("error", str(e), None))
    
    # Start the command in a separate thread
    thread = threading.Thread(target=run_command)
    thread.start()
    
    # Update progress while command is running
    progress = 0
    while thread.is_alive():
        time.sleep(0.1)
        progress = min(progress + 0.01, 0.95)
        progress_bar.progress(progress)
        status_text.text("Running command...")
    
    # Wait for completion
    thread.join()
    
    try:
        status, message, result = message_queue.get_nowait()
        progress_bar.progress(1.0)
        
        if status == "success":
            status_text.text("✅ Command completed successfully!")
            return True, message, result
        else:
            status_text.text("❌ Command failed!")
            return False, message, None
    except queue.Empty:
        status_text.text("❌ Command failed!")
        return False, "Command execution failed", None


def home_page():
    """Home page with project overview."""
    st.title("🚀 WildTrain")
    st.subheader("Modular Computer Vision Framework for Detection and Classification")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Training**\n\nTrain classification and detection models with ease.")
    
    with col2:
        st.info("**Evaluation**\n\nEvaluate model performance and generate reports.")
    
    with col3:
        st.info("**Visualization**\n\nVisualize predictions and analyze results.")
    
    st.markdown("---")
    
    # Show available config files
    #config_files = get_config_files()
    
    #st.subheader("📁 Available Configuration Files")
    
    #for category, files in config_files.items():
        #if files:
            #with st.expander(f"{category.title()} Configurations ({len(files)} files)"):
                #for file_path in files:
                    #st.code(file_path, language="text")


def training_page():
    """Training page for model training."""
    st.title("🏋️ Training")
    
    tab1, tab2 = st.tabs(["Detection Pipeline", "Classification Pipeline"])
        
    with tab1:
        st.subheader("Detection Pipeline")
        
        # Get available detection configs
        detection_configs = [f.as_posix() for f in (ROOT / "pipelines").iterdir() if f.is_file() and f.suffix == ".yaml"]
        
        if detection_configs:
            selected_config = st.selectbox(
                "Select Detection Pipeline Config",
                detection_configs,
                key="detection_config",
                help="Choose a YAML configuration file for detection pipeline"
            )
            
            if st.button("🚀 Start Detection Pipeline", type="primary"):
                if selected_config:
                    st.info(f"Starting detection pipeline with config: {selected_config}")
                    
                    success, message, result = run_command_with_progress(
                        run_detection_pipeline, Path(selected_config)
                    )
                    
                    if success:
                        st.success("Detection pipeline completed successfully!")
                        st.text_area("Output", message, height=200)
                    else:
                        st.error(f"Detection pipeline failed: {message}")
        else:
            st.warning("No detection configuration files found in configs/detection/")
    
    with tab2:
        st.subheader("Classification Pipeline")
        
        # Get available classification configs
        classification_configs = [f.as_posix() for f in (ROOT / "pipelines").iterdir() if f.is_file() and f.suffix == ".yaml"]
        
        if classification_configs:
            selected_config = st.selectbox(
                "Select Classification Pipeline Config",
                classification_configs,
                key="classification_pipeline_config",
                help="Choose a YAML configuration file for classification pipeline"
            )
            
            if st.button("🚀 Start Classification Pipeline", type="primary"):
                if selected_config:
                    st.info(f"Starting classification pipeline with config: {selected_config}")
                    
                    success, message, result = run_command_with_progress(
                        run_classification_pipeline, Path(selected_config)
                    )
                    
                    if success:
                        st.success("Classification pipeline completed successfully!")
                        st.text_area("Output", message, height=200)
                    else:
                        st.error(f"Classification pipeline failed: {message}")
        else:
            st.warning("No classification configuration files found in configs/classification/")


def evaluation_page():
    """Evaluation page for model evaluation."""
    st.title("📊 Evaluation")
    
    tab1, tab2 = st.tabs(["Classifier Evaluation", "Detector Evaluation"])
    
    with tab1:
        st.subheader("Classifier Evaluation")
        
        # Get available evaluation configs
        config_files = get_config_files()
        evaluation_configs = config_files.get("evaluation", [])
        classification_configs = config_files.get("classification", [])
        
        # Combine all potential evaluation configs
        all_configs = evaluation_configs + classification_configs
        
        if all_configs:
            selected_config = st.selectbox(
                "Select Evaluation Configuration",
                all_configs,
                help="Choose a YAML configuration file for classifier evaluation"
            )
            
            if st.button("📊 Evaluate Classifier", type="primary"):
                if selected_config:
                    st.info(f"Starting classifier evaluation with config: {selected_config}")
                    
                    success, message, result = run_command_with_progress(
                        evaluate_classifier, Path(selected_config)
                    )
                    
                    if success:
                        st.success("Classifier evaluation completed successfully!")
                        st.text_area("Output", message, height=200)
                    else:
                        st.error(f"Classifier evaluation failed: {message}")
        else:
            st.warning("No evaluation configuration files found")
    
    with tab2:
        st.subheader("Detector Evaluation")
        
        # Get available detection configs
        detection_configs = config_files.get("detection", [])
        
        if detection_configs:
            selected_config = st.selectbox(
                "Select Detector Evaluation Config",
                detection_configs,
                help="Choose a YAML configuration file for detector evaluation"
            )
            
            model_type = st.selectbox(
                "Model Type",
                ["yolo"],
                help="Type of detector to evaluate"
            )
            
            if st.button("📊 Evaluate Detector", type="primary"):
                if selected_config:
                    st.info(f"Starting detector evaluation with config: {selected_config}")
                    
                    success, message, result = run_command_with_progress(
                        evaluate_detector, Path(selected_config), model_type
                    )
                    
                    if success:
                        st.success("Detector evaluation completed successfully!")
                        st.text_area("Output", message, height=200)
                    else:
                        st.error(f"Detector evaluation failed: {message}")
        else:
            st.warning("No detection configuration files found")


def dataset_analysis_page():
    """Dataset analysis page."""
    st.title("📈 Dataset Analysis")
    
    st.subheader("Dataset Statistics")
    
    # Input for data directory
    data_dir = st.text_input(
        "Dataset Directory Path",
        help="Path to the dataset directory"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        split = st.selectbox(
            "Dataset Split",
            ["train", "val", "test"],
            help="Which split to analyze"
        )
    
    with col2:
        output_file = st.text_input(
            "Output File (Optional)",
            help="Path to save statistics JSON file"
        )
    
    if st.button("📊 Analyze Dataset", type="primary"):
        if data_dir and Path(data_dir).exists():
            st.info(f"Analyzing dataset at: {data_dir}")
            
            success, message, result = run_command_with_progress(
                get_dataset_stats,
                Path(data_dir),
                split,
                Path(output_file) if output_file else None
            )
            
            if success:
                st.success("Dataset analysis completed successfully!")
                st.text_area("Output", message, height=200)
            else:
                st.error(f"Dataset analysis failed: {message}")
        else:
            st.error("Please provide a valid dataset directory path")


def visualization_page():
    """Visualization page for predictions."""
    st.title("🎨 Visualization")
    
    tab1, tab2 = st.tabs(["Classifier Predictions", "Detector Predictions"])
    
    with tab1:
        st.subheader("Classifier Predictions")
        
        dataset_name = st.text_input(
            "FiftyOne Dataset Name",
            help="Name of the FiftyOne dataset to use or create"
        )
        
        checkpoint_path = st.text_input(
            "Checkpoint Path",
            help="Path to the classifier checkpoint (.ckpt) file"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            prediction_field = st.text_input(
                "Prediction Field",
                value="classification_predictions",
                help="Field name to store predictions"
            )
            
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=128,
                value=32,
                help="Batch size for prediction inference"
            )
        
        with col2:
            device = st.selectbox(
                "Device",
                ["cpu", "cuda"],
                help="Device to run inference on"
            )
            
            debug = st.checkbox(
                "Debug Mode",
                help="Process only a small number of samples for debugging"
            )
        
        if st.button("🎨 Visualize Classifier Predictions", type="primary"):
            if dataset_name and checkpoint_path:
                st.info(f"Uploading predictions to FiftyOne dataset: {dataset_name}")
                
                success, message, result = run_command_with_progress(
                    visualize_classifier_predictions,
                    dataset_name,
                    Path(checkpoint_path),
                    prediction_field,
                    batch_size,
                    device,
                    debug
                )
                
                if success:
                    st.success("Predictions uploaded successfully!")
                    st.text_area("Output", message, height=200)
                else:
                    st.error(f"Visualization failed: {message}")
            else:
                st.error("Please provide dataset name and checkpoint path")
    
    with tab2:
        st.subheader("Detector Predictions")
        
        # Get available visualization configs
        config_files = get_config_files()
        visualization_configs = config_files.get("visualization", [])
        detection_configs = config_files.get("detection", [])
        
        # Combine all potential visualization configs
        all_configs = visualization_configs + detection_configs
        
        if all_configs:
            selected_config = st.selectbox(
                "Select Visualization Configuration",
                all_configs,
                help="Choose a YAML configuration file for detector visualization"
            )
            
            if st.button("🎨 Visualize Detector Predictions", type="primary"):
                if selected_config:
                    st.info(f"Loading visualization config from: {selected_config}")
                    
                    success, message, result = run_command_with_progress(
                        visualize_detector_predictions,
                        Path(selected_config)
                    )
                    
                    if success:
                        st.success("Detector predictions uploaded successfully!")
                        st.text_area("Output", message, height=200)
                    else:
                        st.error(f"Visualization failed: {message}")
        else:
            st.warning("No visualization configuration files found")


def configuration_page():
    """Configuration management page."""
    st.title("⚙️ Configuration Management")
    
    st.subheader("Available Configuration Files")
    
    config_files = get_config_files()

    pipeline_files = [f.as_posix() for i,f in enumerate((ROOT / "pipelines").iterdir()) if f.is_file() and f.suffix == ".yaml"]
    config_files["pipelines"] = pipeline_files

    for category, files in config_files.items():
        if files:
            with st.expander(f"{category.title()} ({len(files)} files)"):
                for file_path in files:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.code(file_path, language="text")
                    
                    #with col2:
                    if st.button("View", key=f"view_{file_path}"):
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()
                            conf = yaml.safe_load(content)
                            st.text_area("Configuration Content", json.dumps(conf, indent=4), height=300)
                        except Exception as e:
                            st.error(f"Error reading file: {e}")


def main():
    """Main Streamlit application."""
    setup_page_config()
    
    # Sidebar navigation
    st.sidebar.title("🚀 WildTrain")
    
    page = st.sidebar.selectbox(
        "Navigation",
        ["Home", "Training", "Evaluation", "Dataset Analysis", "Visualization", "Configuration"],
        help="Select a page to navigate"
    )


    with st.sidebar:
        st.divider()
        launch_fiftyone_ui()
    
    # Page routing
    if page == "Home":
        home_page()
    elif page == "Training":
        training_page()
    elif page == "Evaluation":
        evaluation_page()
    elif page == "Dataset Analysis":
        st.info("Not implemented yet")
    #    dataset_analysis_page()
    elif page == "Visualization":
        visualization_page()
    elif page == "Configuration":
        configuration_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Version:** 0.1.0")
    st.sidebar.markdown("**Author:** Fadel Seydou")


if __name__ == "__main__":
    main()
