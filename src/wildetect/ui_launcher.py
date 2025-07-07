#!/usr/bin/env python3
"""
WildDetect UI Launcher with CLI Integration

This script launches the Streamlit UI with integrated CLI functionality.
"""

import os
import sys
from pathlib import Path


def main():
    """Launch the WildDetect UI with CLI integration."""
    try:
        import streamlit as st

        # Set up the page configuration
        st.set_page_config(
            page_title="WildDetect - Wildlife Detection System",
            page_icon="🦁",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Import and run the main UI
        from wildetect.ui.main import main as ui_main

        ui_main()

    except ImportError as e:
        print(f"Error: Missing required dependencies. {e}")
        print("Please install the required packages:")
        print("pip install streamlit pandas rich typer")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching UI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
