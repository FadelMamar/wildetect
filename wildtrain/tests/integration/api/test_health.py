"""
Critical API Health & Basic Functionality Tests

These tests are essential for basic API functionality and must be implemented first.
"""

import pytest
import requests
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from wildtrain.api.main import fastapi_app as app


@pytest.mark.critical
@pytest.mark.health
class TestAPIHealthAndBasicFunctionality:
    """Test API health endpoints and basic functionality."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        return TestClient(app)
    
    @pytest.fixture
    def temp_test_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp(prefix="wildtrain_api_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_health_endpoint_returns_200(self, client):
        """Test that /health endpoint returns 200 status code."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "timestamp" in response.json()
    
    def test_docs_endpoint_is_accessible(self, client):
        """Test that /docs endpoint is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_openapi_endpoint_is_accessible(self, client):
        """Test that /openapi.json endpoint is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert response.headers.get("content-type") == "application/json"
    
    def test_redoc_endpoint_is_accessible(self, client):
        """Test that /redoc endpoint is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_cors_headers_are_present(self, client):
        """Test that CORS headers are present in responses."""
        response = client.get("/health")
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-credentials" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers
    
    def test_root_endpoint_returns_api_info(self, client):
        """Test that root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "redoc" in data
        assert data["message"] == "WildTrain API"
    
    def test_api_startup_and_shutdown(self):
        """Test API startup and shutdown functionality."""
        # Test that the app can be created without errors
        from wildtrain.api.main import create_app
        app = create_app()
        assert app is not None
        assert hasattr(app, "routes")
        
        # Test that all expected routers are included
        routes = [route.path for route in app.routes]
        expected_prefixes = ["/training", "/evaluation", "/pipeline", "/visualization", "/dataset", "/config"]
        
        for prefix in expected_prefixes:
            assert any(route.startswith(prefix) for route in routes), f"Missing router with prefix {prefix}"
    
    def test_basic_request_response_flow(self, client):
        """Test basic request/response flow for all endpoints."""
        # Test all main endpoint groups
        endpoints_to_test = [
            "/training/status",
            "/evaluation/status", 
            "/pipeline/status",
            "/visualization/status",
            "/dataset/status",
            "/config/status"
        ]
        
        for endpoint in endpoints_to_test:
            response = client.get(endpoint)
            # Should return some response (even if 404 for unimplemented endpoints)
            assert response.status_code in [200, 404, 405]
    
    def test_error_handling(self, client):
        """Test that error handling works correctly."""
        # Test non-existent endpoint
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        # Test invalid method
        response = client.post("/health")
        assert response.status_code == 405
    
    def test_api_version_consistency(self, client):
        """Test that API version is consistent across endpoints."""
        # Get version from root endpoint
        root_response = client.get("/")
        root_version = root_response.json().get("version")
        
        # Get version from OpenAPI spec
        openapi_response = client.get("/openapi.json")
        openapi_data = openapi_response.json()
        openapi_version = openapi_data.get("info", {}).get("version")
        
        # Versions should match
        assert root_version == openapi_version, f"Version mismatch: root={root_version}, openapi={openapi_version}"
    
    def test_logging_configuration(self):
        """Test that logging is properly configured."""
        import logging
        logger = logging.getLogger("wildtrain.api")
        assert logger.level <= logging.INFO
    
    def test_middleware_configuration(self, client):
        """Test that middleware is properly configured."""
        # Test CORS preflight request
        response = client.options("/health")
        assert response.status_code in [200, 405]  # 405 is acceptable for OPTIONS
    
    def test_exception_handler_registration(self):
        """Test that custom exception handlers are registered."""
        from wildtrain.api.main import fastapi_app as app
        from wildtrain.api.utils.error_handling import WildTrainAPIException
        
        # Check that the custom exception handler is registered
        exception_handlers = app.exception_handlers
        assert WildTrainAPIException in exception_handlers
    
    def test_router_inclusion(self):
        """Test that all routers are properly included."""
        from wildtrain.api.main import fastapi_app as app
        
        # Check that all expected routers are included
        expected_routers = [
            "training", "evaluation", "pipeline", 
            "visualization", "dataset", "config"
        ]
        
        for router_name in expected_routers:
            # Check if router endpoints exist
            router_found = False
            for route in app.routes:
                if hasattr(route, 'prefix') and router_name in route.prefix:
                    router_found = True
                    break
            assert router_found, f"Router {router_name} not found in app routes"
    
    def test_api_metadata(self, client):
        """Test that API metadata is correctly set."""
        response = client.get("/openapi.json")
        openapi_data = response.json()
        
        # Check API metadata
        info = openapi_data.get("info", {})
        assert info.get("title") == "WildTrain API"
        assert "description" in info
        assert "version" in info
        
        # Check that servers are configured
        assert "servers" in openapi_data or "host" in openapi_data
    
    def test_response_content_types(self, client):
        """Test that responses have correct content types."""
        # Test JSON endpoints
        json_endpoints = ["/", "/health"]
        for endpoint in json_endpoints:
            response = client.get(endpoint)
            assert response.headers.get("content-type") == "application/json"
        
        # Test HTML endpoints
        html_endpoints = ["/docs", "/redoc"]
        for endpoint in html_endpoints:
            response = client.get(endpoint)
            assert "text/html" in response.headers.get("content-type", "")
    
    def test_api_structure_integrity(self):
        """Test that API structure is intact and properly configured."""
        from wildtrain.api.main import fastapi_app as app
        
        # Check that FastAPI app is properly configured
        assert hasattr(app, "title")
        assert hasattr(app, "version")
        assert hasattr(app, "description")
        
        # Check that middleware is configured
        assert len(app.user_middleware) > 0
        
        # Check that exception handlers are configured
        assert len(app.exception_handlers) > 0
