"""
WildDetect Reflex Frontend

A modern web interface for the WildDetect wildlife detection system.
"""

import reflex as rx

from .pages.index import index
from .state import WildDetectState

# Create the Reflex app
app = rx.App()

# Add the main page
app.add_page(index, route="/")

# Add API transformer to integrate with existing FastAPI backend
app.api_transformer = lambda app: app
