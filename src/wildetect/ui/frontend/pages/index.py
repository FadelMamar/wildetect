"""
Main index page for WildDetect Reflex frontend.
"""

import reflex as rx

from ..state import WildDetectState


def index() -> rx.Component:
    """Main dashboard page."""
    return rx.box(
        rx.vstack(
            # Header
            rx.heading("WildDetect Dashboard", size="lg", color="blue.600", mb=6),
            # System Info Card
            rx.cond(
                WildDetectState.system_info,
                rx.card(
                    rx.vstack(
                        rx.heading("System Information", size="md"),
                        rx.text(
                            f"PyTorch Version: {WildDetectState.system_info.get('pytorch_version', 'Unknown')}"
                        ),
                        rx.text(
                            f"CUDA Available: {WildDetectState.system_info.get('cuda_available', False)}"
                        ),
                        rx.cond(
                            WildDetectState.system_info.get("cuda_device"),
                            rx.text(
                                f"CUDA Device: {WildDetectState.system_info.get('cuda_device')}"
                            ),
                        ),
                        align="start",
                        spacing=2,
                    ),
                    width="100%",
                    mb=4,
                ),
            ),
            # Navigation Tabs
            rx.tabs(
                rx.tab_list(
                    rx.tab("Upload Files", id="upload"),
                    rx.tab("Detection Jobs", id="jobs"),
                    rx.tab("Census Campaigns", id="census"),
                    rx.tab("FiftyOne", id="fiftyone"),
                    rx.tab("System Info", id="system"),
                ),
                rx.tab_panels(
                    rx.tab_panel(upload_panel(), id="upload"),
                    rx.tab_panel(jobs_panel(), id="jobs"),
                    rx.tab_panel(census_panel(), id="census"),
                    rx.tab_panel(fiftyone_panel(), id="fiftyone"),
                    rx.tab_panel(system_panel(), id="system"),
                ),
                width="100%",
                color_scheme="blue",
            ),
            # Error Message
            rx.cond(
                WildDetectState.error_message,
                rx.alert(
                    rx.alert_icon(),
                    rx.alert_title("Error"),
                    rx.alert_description(WildDetectState.error_message),
                    status="error",
                    mb=4,
                ),
            ),
            # Loading Overlay
            rx.cond(
                WildDetectState.loading,
                rx.overlay(
                    rx.spinner(size="lg", color="blue.500"),
                    bg="rgba(0, 0, 0, 0.5)",
                ),
            ),
            spacing=4,
            width="100%",
            max_width="1200px",
            mx="auto",
            p=6,
        ),
        min_height="100vh",
        bg="gray.50",
    )


def upload_panel() -> rx.Component:
    """File upload panel."""
    return rx.vstack(
        rx.heading("Upload Images", size="md", mb=4),
        # File Upload Component
        rx.upload(
            rx.text("Drag and drop images here or click to select"),
            border="2px dashed",
            border_color="blue.300",
            border_radius="lg",
            p=8,
            text_align="center",
            bg="white",
            _hover={"border_color": "blue.500"},
        ),
        # Upload Progress
        rx.cond(
            WildDetectState.upload_progress > 0,
            rx.vstack(
                rx.progress(
                    value=WildDetectState.upload_progress,
                    width="100%",
                    color_scheme="blue",
                ),
                rx.text(WildDetectState.upload_status),
                spacing=2,
                width="100%",
            ),
        ),
        # Uploaded Files List
        rx.cond(
            WildDetectState.uploaded_files,
            rx.vstack(
                rx.heading("Uploaded Files", size="sm"),
                rx.vstack(
                    rx.foreach(
                        WildDetectState.uploaded_files,
                        lambda file: rx.text(file, font_size="sm"),
                    ),
                    align="start",
                    spacing=1,
                ),
                align="start",
                width="100%",
            ),
        ),
        # Detection Configuration
        rx.card(
            rx.vstack(
                rx.heading("Detection Configuration", size="sm"),
                rx.hstack(
                    rx.vstack(
                        rx.text("Model Path:"),
                        rx.input(
                            placeholder="Path to model file",
                            id="model_path",
                        ),
                        align="start",
                        spacing=1,
                    ),
                    rx.vstack(
                        rx.text("Confidence:"),
                        rx.slider(
                            min_=0.0,
                            max_=1.0,
                            step=0.1,
                            default_value=0.2,
                            id="confidence",
                        ),
                        align="start",
                        spacing=1,
                    ),
                    width="100%",
                    spacing=4,
                ),
                rx.button(
                    "Start Detection",
                    color_scheme="blue",
                    on_click=WildDetectState.start_detection,
                    is_loading=WildDetectState.loading,
                ),
                spacing=4,
                width="100%",
            ),
            width="100%",
        ),
        spacing=6,
        width="100%",
    )


def jobs_panel() -> rx.Component:
    """Detection jobs panel."""
    return rx.vstack(
        rx.heading("Detection Jobs", size="md", mb=4),
        # Jobs List
        rx.cond(
            WildDetectState.detection_jobs,
            rx.vstack(
                rx.foreach(
                    WildDetectState.detection_jobs,
                    lambda job: job_card(job),
                ),
                spacing=4,
                width="100%",
            ),
            rx.text("No detection jobs found."),
        ),
        spacing=4,
        width="100%",
    )


def census_panel() -> rx.Component:
    """Census campaigns panel."""
    return rx.vstack(
        rx.heading("Census Campaigns", size="md", mb=4),
        # Census Configuration
        rx.card(
            rx.vstack(
                rx.heading("New Census Campaign", size="sm"),
                rx.hstack(
                    rx.vstack(
                        rx.text("Campaign ID:"),
                        rx.input(
                            placeholder="Enter campaign ID",
                            id="campaign_id",
                        ),
                        align="start",
                        spacing=1,
                    ),
                    rx.vstack(
                        rx.text("Pilot Name:"),
                        rx.input(
                            placeholder="Enter pilot name",
                            id="pilot_name",
                        ),
                        align="start",
                        spacing=1,
                    ),
                    width="100%",
                    spacing=4,
                ),
                rx.button(
                    "Start Census Campaign",
                    color_scheme="green",
                    on_click=WildDetectState.start_census,
                    is_loading=WildDetectState.loading,
                ),
                spacing=4,
                width="100%",
            ),
            width="100%",
        ),
        # Census Jobs List
        rx.cond(
            WildDetectState.census_jobs,
            rx.vstack(
                rx.foreach(
                    WildDetectState.census_jobs,
                    lambda job: census_job_card(job),
                ),
                spacing=4,
                width="100%",
            ),
            rx.text("No census campaigns found."),
        ),
        spacing=6,
        width="100%",
    )


def fiftyone_panel() -> rx.Component:
    """FiftyOne integration panel."""
    return rx.vstack(
        rx.heading("FiftyOne Integration", size="md", mb=4),
        rx.button(
            "Launch FiftyOne",
            color_scheme="purple",
            on_click=WildDetectState.launch_fiftyone,
        ),
        rx.button(
            "Refresh Datasets",
            color_scheme="blue",
            on_click=WildDetectState.get_fiftyone_datasets,
        ),
        # Datasets List
        rx.cond(
            WildDetectState.fiftyone_datasets,
            rx.vstack(
                rx.heading("Available Datasets", size="sm"),
                rx.foreach(
                    WildDetectState.fiftyone_datasets,
                    lambda dataset: rx.text(dataset.get("name", "Unknown")),
                ),
                spacing=2,
                width="100%",
            ),
        ),
        spacing=4,
        width="100%",
    )


def system_panel() -> rx.Component:
    """System information panel."""
    return rx.vstack(
        rx.heading("System Information", size="md", mb=4),
        rx.cond(
            WildDetectState.system_info,
            rx.vstack(
                rx.text(
                    f"PyTorch Version: {WildDetectState.system_info.get('pytorch_version', 'Unknown')}"
                ),
                rx.text(
                    f"CUDA Available: {WildDetectState.system_info.get('cuda_available', False)}"
                ),
                rx.cond(
                    WildDetectState.system_info.get("cuda_device"),
                    rx.text(
                        f"CUDA Device: {WildDetectState.system_info.get('cuda_device')}"
                    ),
                ),
                rx.heading("Dependencies", size="sm", mt=4),
                rx.foreach(
                    WildDetectState.system_info.get("dependencies", {}),
                    lambda dep: rx.text(f"{dep[0]}: {'✓' if dep[1] else '✗'}"),
                ),
                align="start",
                spacing=2,
                width="100%",
            ),
            rx.text("Loading system information..."),
        ),
        spacing=4,
        width="100%",
    )


def job_card(job_data: dict) -> rx.Component:
    """Individual job card component."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.heading(f"Job {job_data.get('job_id', 'Unknown')[:8]}", size="sm"),
                rx.badge(
                    job_data.get("status", "unknown"),
                    color_scheme=get_status_color(job_data.get("status")),
                ),
                justify="space-between",
                width="100%",
            ),
            rx.text(job_data.get("message", "")),
            rx.cond(
                job_data.get("progress", 0) > 0,
                rx.progress(
                    value=job_data.get("progress", 0),
                    width="100%",
                    color_scheme="blue",
                ),
            ),
            align="start",
            spacing=2,
            width="100%",
        ),
        width="100%",
    )


def census_job_card(job_data: dict) -> rx.Component:
    """Individual census job card component."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.heading(
                    f"Census {job_data.get('campaign_id', 'Unknown')}", size="sm"
                ),
                rx.badge(
                    job_data.get("status", "unknown"),
                    color_scheme=get_status_color(job_data.get("status")),
                ),
                justify="space-between",
                width="100%",
            ),
            rx.text(job_data.get("message", "")),
            rx.cond(
                job_data.get("progress", 0) > 0,
                rx.progress(
                    value=job_data.get("progress", 0),
                    width="100%",
                    color_scheme="green",
                ),
            ),
            align="start",
            spacing=2,
            width="100%",
        ),
        width="100%",
    )


def get_status_color(status: str) -> str:
    """Get color scheme for job status."""
    status_colors = {
        "running": "blue",
        "completed": "green",
        "failed": "red",
        "pending": "yellow",
    }
    return status_colors.get(status, "gray")
