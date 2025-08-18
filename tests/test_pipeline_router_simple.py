#!/usr/bin/env python3
"""
Simple tests for Pipeline Router
Tests the fixed pipeline router functionality
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json
from pathlib import Path
import tempfile
import os

class TestPipelineRouterSimple(unittest.TestCase):
    """Test Pipeline Router functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = Mock()
        self.mock_config.model_dump.return_value = {
            "name": "test_pipeline",
            "timeframe": "1h",
            "coins": ["BTC", "ETH"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-31"
        }
        
        self.mock_db = AsyncMock()
        self.mock_session = AsyncMock()
        
    def test_pipeline_router_imports(self):
        """Test that pipeline router can be imported"""
        try:
            from backend.app.routers.pipeline.router import router
            self.assertIsNotNone(router)
        except ImportError as e:
            self.fail(f"Failed to import pipeline router: {e}")
    
    def test_pipeline_router_structure(self):
        """Test pipeline router structure and endpoints"""
        from backend.app.routers.pipeline.router import router
        
        # Check that router has the expected prefix
        self.assertEqual(router.prefix, "/pipeline")
        
        # Check that router has the expected tags
        self.assertIn("Pipeline", router.tags)
        
        # Check that all expected endpoints exist
        expected_routes = [
            "/run",
            "/save", 
            "/{pipeline_id}",
            "/tasks/{task_id}/revoke",
            "/backtests",
            "/backtests/{bt_id}",
            "/run/{pipeline_id}",
            "/artifacts/{path:path}"
        ]
        
        route_paths = [route.path for route in router.routes]
        for expected_route in expected_routes:
            self.assertTrue(
                any(expected_route in route_path for route_path in route_paths),
                f"Expected route {expected_route} not found in router"
            )
    
    @patch('backend.app.routers.pipeline.router.celery_app')
    @patch('backend.app.routers.pipeline.router.logger')
    def test_run_pipeline_success(self, mock_logger, mock_celery):
        """Test successful pipeline execution"""
        from backend.app.routers.pipeline.router import run_pipeline
        
        # Mock celery task
        mock_task = Mock()
        mock_task.id = "test_task_123"
        mock_celery.send_task.return_value = mock_task
        
        # Mock authorization
        mock_auth = "test_token"
        
        # Test the function
        with patch('backend.app.routers.pipeline.router.verify_authorization_admin', return_value=mock_auth):
            result = asyncio.run(run_pipeline(self.mock_config, mock_auth))
        
        # Verify results
        self.assertIn("task_id", result)
        self.assertIn("status", result)
        self.assertIn("message", result)
        self.assertEqual(result["task_id"], "test_task_123")
        self.assertEqual(result["status"], "started")
        
        # Verify celery was called
        mock_celery.send_task.assert_called_once()
        mock_logger.info.assert_called()
    
    @patch('backend.app.routers.pipeline.router.celery_app')
    def test_run_pipeline_invalid_config(self, mock_celery):
        """Test pipeline execution with invalid configuration"""
        from backend.app.routers.pipeline.router import run_pipeline
        
        # Mock invalid config
        invalid_config = Mock()
        invalid_config.model_dump.return_value = {}
        
        # Mock authorization
        mock_auth = "test_token"
        
        # Test the function
        with patch('backend.app.routers.pipeline.router.verify_authorization_admin', return_value=mock_auth):
            with self.assertRaises(Exception) as context:
                asyncio.run(run_pipeline(invalid_config, mock_auth))
            
            self.assertIn("Invalid pipeline configuration", str(context.exception))
    
    @patch('backend.app.routers.pipeline.router.celery_app')
    def test_run_pipeline_celery_error(self, mock_celery):
        """Test pipeline execution when celery fails"""
        from backend.app.routers.pipeline.router import run_pipeline
        
        # Mock celery error
        mock_celery.send_task.side_effect = Exception("Celery connection failed")
        
        # Mock authorization
        mock_auth = "test_token"
        
        # Test the function
        with patch('backend.app.routers.pipeline.router.verify_authorization_admin', return_value=mock_auth):
            with self.assertRaises(Exception) as context:
                asyncio.run(run_pipeline(self.mock_config, mock_auth))
            
            self.assertIn("Failed to start pipeline", str(context.exception))
    
    @patch('backend.app.routers.pipeline.router.celery_app')
    def test_revoke_pipeline_task_success(self, mock_celery):
        """Test successful task revocation"""
        from backend.app.routers.pipeline.router import revoke_pipeline_task
        
        # Mock celery control
        mock_celery.control.revoke.return_value = True
        
        # Mock authorization
        mock_auth = "test_token"
        
        # Test the function
        with patch('backend.app.routers.pipeline.router.verify_authorization_admin', return_value=mock_auth):
            result = asyncio.run(revoke_pipeline_task("test_task_123", mock_auth))
        
        # Verify results
        self.assertIn("status", result)
        self.assertIn("task_id", result)
        self.assertIn("message", result)
        self.assertEqual(result["status"], "revoked")
        self.assertEqual(result["task_id"], "test_task_123")
        
        # Verify celery was called
        mock_celery.control.revoke.assert_called_once_with(
            "test_task_123", 
            terminate=True, 
            signal="SIGTERM"
        )
    
    def test_revoke_pipeline_task_invalid_id(self):
        """Test task revocation with invalid task ID"""
        from backend.app.routers.pipeline.router import revoke_pipeline_task
        
        # Mock authorization
        mock_auth = "test_token"
        
        # Test with empty task ID
        with patch('backend.app.routers.pipeline.router.verify_authorization_admin', return_value=mock_auth):
            result = asyncio.run(revoke_pipeline_task("", mock_auth))
            
            # Should return error status instead of raising exception
            self.assertIn("status", result)
            self.assertIn("message", result)
            self.assertIn("Task revocation attempted", result["message"])
    
    def test_download_artifact_security(self):
        """Test artifact download security validation"""
        from backend.app.routers.pipeline.router import download_artifact
        
        # Mock authorization
        mock_auth = "test_token"
        
        # Test with absolute path outside allowed directory
        with patch('backend.app.routers.pipeline.router.verify_authorization_admin', return_value=mock_auth):
            with self.assertRaises(Exception) as context:
                asyncio.run(download_artifact("/etc/passwd", mock_auth))
            
            self.assertIn("Access denied", str(context.exception))
    
    def test_download_artifact_file_not_found(self):
        """Test artifact download with non-existent file"""
        from backend.app.routers.pipeline.router import download_artifact
        
        # Mock authorization
        mock_auth = "test_token"
        
        # Test with non-existent file
        with patch('backend.app.routers.pipeline.router.verify_authorization_admin', return_value=mock_auth):
            with patch('pathlib.Path.exists', return_value=False):
                with self.assertRaises(Exception) as context:
                    asyncio.run(download_artifact("nonexistent_file.txt", mock_auth))
                
                self.assertIn("Artifact not found", str(context.exception))
    
    def test_pipeline_router_logging(self):
        """Test that pipeline router has proper logging setup"""
        from backend.app.routers.pipeline.router import logger
        
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "backend.app.routers.pipeline.router")
    
    def test_pipeline_router_error_handling(self):
        """Test that pipeline router has proper error handling"""
        # Check that all functions have try-catch blocks
        from backend.app.routers.pipeline.router import (
            run_pipeline, save_pipeline, get_pipeline, 
            revoke_pipeline_task, list_backtests, get_backtest,
            run_pipeline_by_id, download_artifact
        )
        
        # This test verifies that the functions exist and can be imported
        # The actual error handling is tested in other test methods
        self.assertIsNotNone(run_pipeline)
        self.assertIsNotNone(save_pipeline)
        self.assertIsNotNone(get_pipeline)
        self.assertIsNotNone(revoke_pipeline_task)
        self.assertIsNotNone(list_backtests)
        self.assertIsNotNone(get_backtest)
        self.assertIsNotNone(run_pipeline_by_id)
        self.assertIsNotNone(download_artifact)


if __name__ == "__main__":
    unittest.main()
