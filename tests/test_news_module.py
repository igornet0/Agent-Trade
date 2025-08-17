#!/usr/bin/env python3
"""
Tests for News module functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json


class TestNewsBackgroundORM:
    """Test NewsBackground ORM functions"""
    
    @pytest.mark.asyncio
    async def test_orm_create_news_background(self):
        """Test creating news background record"""
        from core.database.orm.news import orm_create_news_background
        
        # Mock session
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()
        
        # Test data
        coin_id = 1
        timestamp = datetime.utcnow()
        score = 25.5
        source_count = 5
        sources_breakdown = {"twitter": 3, "telegram": 2}
        
        # Mock the NewsBackground model
        with patch('core.database.orm.news.NewsBackground') as mock_model:
            mock_instance = Mock()
            mock_model.return_value = mock_instance
            
            result = await orm_create_news_background(
                mock_session, coin_id, timestamp, score, source_count, sources_breakdown
            )
            
            # Verify model was created with correct parameters
            mock_model.assert_called_once_with(
                coin_id=coin_id,
                timestamp=timestamp,
                score=score,
                source_count=source_count,
                sources_breakdown=sources_breakdown,
                window_hours=24,
                decay_factor=0.95
            )
            
            # Verify session operations
            mock_session.add.assert_called_once_with(mock_instance)
            mock_session.commit.assert_called_once()
            mock_session.refresh.assert_called_once_with(mock_instance)
            
            assert result == mock_instance
    
    @pytest.mark.asyncio
    async def test_orm_get_news_background(self):
        """Test getting news background records"""
        from core.database.orm.news import orm_get_news_background
        
        # Mock session and result
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_session.execute.return_value = mock_result
        
        # Mock data
        mock_backgrounds = [Mock(), Mock()]
        mock_result.scalars.return_value.all.return_value = mock_backgrounds
        
        # Test query
        coin_id = 1
        start_time = datetime.utcnow() - timedelta(hours=24)
        end_time = datetime.utcnow()
        
        result = await orm_get_news_background(
            mock_session, coin_id, start_time, end_time, 100
        )
        
        assert result == mock_backgrounds
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_orm_get_latest_news_background(self):
        """Test getting latest news background"""
        from core.database.orm.news import orm_get_latest_news_background
        
        # Mock session and result
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_session.execute.return_value = mock_result
        
        # Mock data
        mock_background = Mock()
        mock_result.scalar_one_or_none.return_value = mock_background
        
        # Test query
        coin_id = 1
        result = await orm_get_latest_news_background(mock_session, coin_id)
        
        assert result == mock_background
        mock_session.execute.assert_called_once()


class TestNewsBackgroundService:
    """Test NewsBackgroundService functionality"""
    
    def test_cache_key_generation(self):
        """Test Redis cache key generation"""
        from core.services.news_background_service import NewsBackgroundService
        
        service = NewsBackgroundService()
        
        # Test different coin IDs and window hours
        key1 = service._get_cache_key(1, 24)
        key2 = service._get_cache_key(2, 48)
        key3 = service._get_cache_key(1, 12)
        
        assert key1 == "news_bg:1:24h"
        assert key2 == "news_bg:2:48h"
        assert key3 == "news_bg:1:12h"
        assert key1 != key2
        assert key1 != key3
    
    @pytest.mark.asyncio
    async def test_calculate_news_background_no_data(self):
        """Test background calculation with no news data"""
        from core.services.news_background_service import NewsBackgroundService
        
        # Mock session
        mock_session = AsyncMock()
        
        # Mock ORM function to return no data
        with patch('core.services.news_background_service.orm_get_news_for_background_calculation') as mock_get:
            mock_get.return_value = []
            
            # Mock ORM create function
            with patch('core.services.news_background_service.orm_create_news_background') as mock_create:
                mock_create.return_value = Mock()
                
                service = NewsBackgroundService()
                result = await service.calculate_news_background(mock_session, 1)
                
                # Verify result structure
                assert result['coin_id'] == 1
                assert result['score'] == 0.0
                assert result['source_count'] == 0
                assert result['news_count'] == 0
                assert result['weighted_score'] == 0.0
                assert 'timestamp' in result
                assert result['window_hours'] == 24
                assert result['decay_factor'] == 0.95
    
    @pytest.mark.asyncio
    async def test_calculate_news_background_with_data(self):
        """Test background calculation with news data"""
        from core.services.news_background_service import NewsBackgroundService
        
        # Mock session
        mock_session = AsyncMock()
        
        # Mock news data
        mock_news_data = [
            {
                'date': datetime.utcnow() - timedelta(hours=1),
                'score': 0.5,
                'source': 'twitter',
                'title': 'Test news 1'
            },
            {
                'date': datetime.utcnow() - timedelta(hours=2),
                'score': -0.3,
                'source': 'telegram',
                'title': 'Test news 2'
            }
        ]
        
        with patch('core.services.news_background_service.orm_get_news_for_background_calculation') as mock_get:
            mock_get.return_value = mock_news_data
            
            # Mock ORM create function
            with patch('core.services.news_background_service.orm_create_news_background') as mock_create:
                mock_create.return_value = Mock()
                
                service = NewsBackgroundService()
                result = await service.calculate_news_background(mock_session, 1)
                
                # Verify result structure
                assert result['coin_id'] == 1
                assert result['source_count'] == 2
                assert result['news_count'] == 2
                assert 'sources_breakdown' in result
                assert result['sources_breakdown']['twitter'] == 1
                assert result['sources_breakdown']['telegram'] == 1
                assert 'timestamp' in result
    
    @pytest.mark.asyncio
    async def test_get_cached_background(self):
        """Test getting cached background from Redis"""
        from core.services.news_background_service import NewsBackgroundService
        
        # Mock Redis client
        mock_redis = Mock()
        mock_redis.get.return_value = json.dumps({
            'coin_id': 1,
            'score': 25.5,
            'source_count': 5
        })
        
        service = NewsBackgroundService(mock_redis)
        result = await service.get_cached_background(1, 24)
        
        assert result is not None
        assert result['coin_id'] == 1
        assert result['score'] == 25.5
        assert result['source_count'] == 5
        
        # Test cache miss
        mock_redis.get.return_value = None
        result = await service.get_cached_background(1, 24)
        assert result is None
    
    def test_clear_cache(self):
        """Test clearing Redis cache"""
        from core.services.news_background_service import NewsBackgroundService
        
        # Mock Redis client
        mock_redis = Mock()
        mock_redis.keys.return_value = [b"news_bg:1:24h", b"news_bg:2:24h"]
        mock_redis.delete.return_value = 2
        
        service = NewsBackgroundService(mock_redis)
        result = service.clear_cache()
        
        assert result is True
        mock_redis.keys.assert_called_once_with("news_bg:*")
        mock_redis.delete.assert_called_once_with(b"news_bg:1:24h", b"news_bg:2:24h")


class TestNewsAPIEndpoints:
    """Test News API endpoints"""
    
    @pytest.mark.asyncio
    async def test_recalc_news_background_endpoint(self):
        """Test news background recalculation endpoint"""
        from fastapi import HTTPException
        from backend.app.routers.apidb_agent.router import recalc_news_background
        
        # Mock dependencies
        mock_auth = "admin_token"
        
        # Mock service and Celery
        with patch('backend.app.routers.apidb_agent.router.NewsBackgroundService') as mock_service_class:
            with patch('backend.app.routers.apidb_agent.router.celery_app') as mock_celery:
                mock_task = Mock()
                mock_task.id = "test_task_id"
                mock_celery.send_task.return_value = mock_task
                
                # Test with specific coins
                result = await recalc_news_background(
                    coins="1,2,3",
                    window_hours=48,
                    decay_factor=0.9,
                    force_recalculate=True,
                    _=mock_auth
                )
                
                assert result["status"] == "started"
                assert result["task_id"] == "test_task_id"
                assert len(result["config"]["coin_ids"]) == 3
                assert result["config"]["window_hours"] == 48
                assert result["config"]["decay_factor"] == 0.9
                assert result["config"]["force_recalculate"] is True
    
    @pytest.mark.asyncio
    async def test_get_news_background_endpoint(self):
        """Test getting news background endpoint"""
        from backend.app.routers.apidb_agent.router import get_news_background
        
        # Mock dependencies
        mock_auth = "admin_token"
        coin_id = 1
        
        # Mock service
        with patch('backend.app.routers.apidb_agent.router.NewsBackgroundService') as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            
            # Mock database session
            with patch('backend.app.routers.apidb_agent.router.db_helper.get_session') as mock_db:
                mock_session = AsyncMock()
                mock_db.return_value.__aenter__.return_value = mock_session
                
                # Mock service methods
                mock_service.get_cached_background.return_value = None
                mock_service.get_background_history.return_value = []
                mock_service.calculate_news_background.return_value = {
                    'coin_id': 1,
                    'score': 25.5,
                    'source_count': 5
                }
                
                result = await get_news_background(
                    coin_id=coin_id,
                    start_time=None,
                    end_time=None,
                    limit=1000,
                    _=mock_auth
                )
                
                assert result["status"] == "success"
                assert result["source"] == "calculated"
                assert "data" in result
    
    @pytest.mark.asyncio
    async def test_get_coins_with_news_endpoint(self):
        """Test getting coins with news data endpoint"""
        from backend.app.routers.apidb_agent.router import get_coins_with_news
        
        # Mock dependencies
        mock_auth = "admin_token"
        
        # Mock service
        with patch('backend.app.routers.apidb_agent.router.NewsBackgroundService') as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            
            # Mock database session
            with patch('backend.app.routers.apidb_agent.router.db_helper.get_session') as mock_db:
                mock_session = AsyncMock()
                mock_db.return_value.__aenter__.return_value = mock_session
                
                # Mock service method
                mock_service.get_coins_with_news_data.return_value = [
                    {'id': 1, 'name': 'Bitcoin', 'symbol': 'BTC', 'has_news_background': True},
                    {'id': 2, 'name': 'Ethereum', 'symbol': 'ETH', 'has_news_background': False}
                ]
                
                result = await get_coins_with_news(_=mock_auth)
                
                assert result["status"] == "success"
                assert result["count"] == 2
                assert len(result["coins"]) == 2
                assert result["coins"][0]["name"] == "Bitcoin"
                assert result["coins"][1]["name"] == "Ethereum"


class TestNewsCeleryTasks:
    """Test News Celery tasks"""
    
    def test_train_news_task_structure(self):
        """Test train_news_task structure and imports"""
        from backend.celery_app.tasks import train_news_task
        
        # Verify task exists and is callable
        assert callable(train_news_task)
        assert hasattr(train_news_task, 'delay')
        assert hasattr(train_news_task, 'apply_async')
    
    def test_evaluate_news_task_structure(self):
        """Test evaluate_news_task structure and imports"""
        from backend.celery_app.tasks import evaluate_news_task
        
        # Verify task exists and is callable
        assert callable(evaluate_news_task)
        assert hasattr(evaluate_news_task, 'delay')
        assert hasattr(evaluate_news_task, 'apply_async')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
