"""
News Background Service for calculating and managing aggregated news sentiment
"""

import redis
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.orm.news import (
    orm_get_news_for_background_calculation,
    orm_create_news_background,
    orm_get_latest_news_background,
    orm_get_news_background,
    orm_get_coins_with_news
)

logger = logging.getLogger(__name__)


class NewsBackgroundService:
    """Service for managing news background calculations"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.cache_ttl = 3600  # 1 hour cache TTL
    
    def _get_cache_key(self, coin_id: int, window_hours: int = 24) -> str:
        """Generate Redis cache key for news background"""
        return f"news_bg:{coin_id}:{window_hours}h"
    
    def _get_aggregated_cache_key(self, coin_id: int) -> str:
        """Generate Redis cache key for aggregated background"""
        return f"news_bg_agg:{coin_id}"
    
    async def calculate_news_background(
        self,
        session: AsyncSession,
        coin_id: int,
        window_hours: int = 24,
        decay_factor: float = 0.95,
        force_recalculate: bool = False
    ) -> Dict[str, Any]:
        """Calculate news background for a specific coin"""
        
        # Check cache first
        if not force_recalculate and self.redis_client:
            cache_key = self._get_cache_key(coin_id, window_hours)
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                try:
                    return json.loads(cached_data)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Invalid cached data for {cache_key}")
        
        # Calculate time window
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=window_hours)
        
        # Get news data for the time window
        news_data = await orm_get_news_for_background_calculation(
            session, coin_id, start_time, end_time
        )
        
        if not news_data:
            # No news data, return neutral background
            background = {
                'coin_id': coin_id,
                'timestamp': end_time,
                'score': 0.0,
                'source_count': 0,
                'sources_breakdown': {},
                'window_hours': window_hours,
                'decay_factor': decay_factor,
                'news_count': 0,
                'weighted_score': 0.0
            }
        else:
            # Calculate weighted score with exponential decay
            weighted_scores = []
            source_counts = {}
            total_weight = 0.0
            
            for news in news_data:
                # Calculate time decay weight
                time_diff = (end_time - news['date']).total_seconds() / 3600  # hours
                weight = decay_factor ** time_diff
                
                weighted_scores.append(news['score'] * weight)
                total_weight += weight
                
                # Count sources
                source = news['source'] or 'unknown'
                source_counts[source] = source_counts.get(source, 0) + 1
            
            # Calculate final weighted score
            if total_weight > 0:
                weighted_score = sum(weighted_scores) / total_weight
            else:
                weighted_score = 0.0
            
            # Normalize score to [-100, 100] range
            normalized_score = max(-100.0, min(100.0, weighted_score * 100))
            
            background = {
                'coin_id': coin_id,
                'timestamp': end_time,
                'score': normalized_score,
                'source_count': len(news_data),
                'sources_breakdown': source_counts,
                'window_hours': window_hours,
                'decay_factor': decay_factor,
                'news_count': len(news_data),
                'weighted_score': weighted_score
            }
        
        # Save to database
        try:
            await orm_create_news_background(
                session=session,
                coin_id=coin_id,
                timestamp=background['timestamp'],
                score=background['score'],
                source_count=background['source_count'],
                sources_breakdown=background['sources_breakdown'],
                window_hours=window_hours,
                decay_factor=decay_factor
            )
        except Exception as e:
            logger.error(f"Failed to save news background for coin {coin_id}: {e}")
        
        # Cache the result
        if self.redis_client:
            try:
                cache_key = self._get_cache_key(coin_id, window_hours)
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(background, default=str)
                )
            except Exception as e:
                logger.warning(f"Failed to cache news background for coin {coin_id}: {e}")
        
        return background
    
    async def calculate_multiple_coins_background(
        self,
        session: AsyncSession,
        coin_ids: List[int],
        window_hours: int = 24,
        decay_factor: float = 0.95
    ) -> Dict[int, Dict[str, Any]]:
        """Calculate news background for multiple coins"""
        results = {}
        
        for coin_id in coin_ids:
            try:
                background = await self.calculate_news_background(
                    session, coin_id, window_hours, decay_factor
                )
                results[coin_id] = background
            except Exception as e:
                logger.error(f"Failed to calculate background for coin {coin_id}: {e}")
                results[coin_id] = {
                    'coin_id': coin_id,
                    'error': str(e),
                    'score': 0.0,
                    'source_count': 0
                }
        
        return results
    
    async def get_cached_background(
        self,
        coin_id: int,
        window_hours: int = 24
    ) -> Optional[Dict[str, Any]]:
        """Get cached news background from Redis"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._get_cache_key(coin_id, window_hours)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Failed to get cached background for coin {coin_id}: {e}")
        
        return None
    
    async def get_background_history(
        self,
        session: AsyncSession,
        coin_id: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get news background history for a coin"""
        try:
            backgrounds = await orm_get_news_background(
                session, coin_id, start_time, end_time, limit
            )
            
            return [
                {
                    'id': bg.id,
                    'timestamp': bg.timestamp.isoformat(),
                    'score': bg.score,
                    'source_count': bg.source_count,
                    'sources_breakdown': bg.sources_breakdown,
                    'window_hours': bg.window_hours,
                    'decay_factor': bg.decay_factor
                }
                for bg in backgrounds
            ]
        except Exception as e:
            logger.error(f"Failed to get background history for coin {coin_id}: {e}")
            return []
    
    async def get_latest_background(
        self,
        session: AsyncSession,
        coin_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get the latest news background for a coin"""
        try:
            background = await orm_get_latest_news_background(session, coin_id)
            
            if background:
                return {
                    'id': background.id,
                    'timestamp': background.timestamp.isoformat(),
                    'score': background.score,
                    'source_count': background.source_count,
                    'sources_breakdown': background.sources_breakdown,
                    'window_hours': background.window_hours,
                    'decay_factor': background.decay_factor
                }
        except Exception as e:
            logger.error(f"Failed to get latest background for coin {coin_id}: {e}")
        
        return None
    
    async def get_coins_with_news_data(
        self,
        session: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Get all coins that have news data"""
        try:
            coins = await orm_get_coins_with_news(session)
            
            return [
                {
                    'id': coin.id,
                    'name': coin.name,
                    'symbol': coin.symbol,
                    'has_news_background': len(coin.news_background) > 0
                }
                for coin in coins
            ]
        except Exception as e:
            logger.error(f"Failed to get coins with news data: {e}")
            return []
    
    def clear_cache(self, coin_id: Optional[int] = None) -> bool:
        """Clear Redis cache for news background"""
        if not self.redis_client:
            return False
        
        try:
            if coin_id:
                # Clear specific coin cache
                pattern = f"news_bg:{coin_id}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            else:
                # Clear all news background cache
                pattern = "news_bg:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    async def recalculate_all_backgrounds(
        self,
        session: AsyncSession,
        window_hours: int = 24,
        decay_factor: float = 0.95
    ) -> Dict[str, Any]:
        """Recalculate news background for all coins with news data"""
        try:
            coins = await self.get_coins_with_news_data(session)
            coin_ids = [coin['id'] for coin in coins]
            
            results = await self.calculate_multiple_coins_background(
                session, coin_ids, window_hours, decay_factor
            )
            
            # Clear cache after recalculation
            self.clear_cache()
            
            return {
                'status': 'success',
                'coins_processed': len(coin_ids),
                'results': results,
                'window_hours': window_hours,
                'decay_factor': decay_factor,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to recalculate all backgrounds: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
