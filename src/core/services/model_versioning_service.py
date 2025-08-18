"""
Model Versioning Service - управление версиями моделей и их продвижением
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import shutil

from ..database.orm.artifacts import (
    orm_create_artifact, orm_get_artifacts_by_agent, orm_get_latest_artifact,
    orm_get_artifacts_by_version, orm_delete_artifact, orm_get_artifact_stats
)
from ..database.orm.agents import orm_get_agent_by_id, orm_update_agent
from ..database.engine import get_db

logger = logging.getLogger(__name__)


class ModelVersioningService:
    """Сервис для управления версиями моделей"""
    
    def __init__(self):
        self.models_dir = "models/models_pth"
        self.production_dir = "models/production"
        os.makedirs(self.production_dir, exist_ok=True)
    
    def create_version(
        self,
        agent_id: int,
        version: str,
        model_path: str,
        config_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Создание новой версии модели
        
        Args:
            agent_id: ID агента
            version: Версия модели
            model_path: Путь к файлу модели
            config_path: Путь к конфигурации
            scaler_path: Путь к scaler'у
            metadata: Дополнительные метаданные
        
        Returns:
            Информация о созданной версии
        """
        try:
            db = next(get_db())
            
            # Проверяем существование агента
            agent = orm_get_agent_by_id(db, agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Создаем артефакты для каждого файла
            artifacts = {}
            
            # Модель
            if os.path.exists(model_path):
                model_artifact = orm_create_artifact(
                    db=db,
                    agent_id=agent_id,
                    version=version,
                    path=model_path,
                    artifact_type="model",
                    size_bytes=os.path.getsize(model_path)
                )
                artifacts["model"] = model_artifact
            
            # Конфигурация
            if config_path and os.path.exists(config_path):
                config_artifact = orm_create_artifact(
                    db=db,
                    agent_id=agent_id,
                    version=version,
                    path=config_path,
                    artifact_type="config",
                    size_bytes=os.path.getsize(config_path)
                )
                artifacts["config"] = config_artifact
            
            # Scaler
            if scaler_path and os.path.exists(scaler_path):
                scaler_artifact = orm_create_artifact(
                    db=db,
                    agent_id=agent_id,
                    version=version,
                    path=scaler_path,
                    artifact_type="scaler",
                    size_bytes=os.path.getsize(scaler_path)
                )
                artifacts["scaler"] = scaler_artifact
            
            # Метаданные
            if metadata:
                metadata_path = f"{self.models_dir}/{agent.agent_type}/{version}_metadata.json"
                os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                metadata_artifact = orm_create_artifact(
                    db=db,
                    agent_id=agent_id,
                    version=version,
                    path=metadata_path,
                    artifact_type="metadata",
                    size_bytes=os.path.getsize(metadata_path)
                )
                artifacts["metadata"] = metadata_artifact
            
            logger.info(f"Created version {version} for agent {agent_id}")
            
            return {
                "version": version,
                "agent_id": agent_id,
                "artifacts": artifacts,
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating version {version} for agent {agent_id}: {e}")
            raise
    
    def promote_version(
        self,
        agent_id: int,
        version: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Продвижение версии в продакшн
        
        Args:
            agent_id: ID агента
            version: Версия для продвижения
            force: Принудительное продвижение
        
        Returns:
            Информация о продвижении
        """
        try:
            db = next(get_db())
            
            # Проверяем существование агента
            agent = orm_get_agent_by_id(db, agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Получаем артефакты версии
            artifacts = orm_get_artifacts_by_version(db, agent_id, version)
            if not artifacts:
                raise ValueError(f"No artifacts found for version {version}")
            
            # Проверяем, что все необходимые артефакты есть
            required_types = ["model"]
            artifact_types = [a.type for a in artifacts]
            
            missing_types = [t for t in required_types if t not in artifact_types]
            if missing_types:
                raise ValueError(f"Missing required artifacts: {missing_types}")
            
            # Создаем продакшн директорию для агента
            agent_prod_dir = f"{self.production_dir}/{agent.agent_type}"
            os.makedirs(agent_prod_dir, exist_ok=True)
            
            # Копируем артефакты в продакшн
            promoted_artifacts = {}
            
            for artifact in artifacts:
                if os.path.exists(artifact.path):
                    # Создаем продакшн путь
                    prod_path = f"{agent_prod_dir}/{artifact.type}_{version}{os.path.splitext(artifact.path)[1]}"
                    
                    # Копируем файл
                    shutil.copy2(artifact.path, prod_path)
                    
                    # Создаем симлинк для последней версии
                    latest_link = f"{agent_prod_dir}/{artifact.type}_latest{os.path.splitext(artifact.path)[1]}"
                    if os.path.exists(latest_link):
                        os.remove(latest_link)
                    os.symlink(prod_path, latest_link)
                    
                    promoted_artifacts[artifact.type] = {
                        "original_path": artifact.path,
                        "production_path": prod_path,
                        "latest_link": latest_link
                    }
            
            # Обновляем агента
            orm_update_agent(
                db=db,
                agent_id=agent_id,
                production_version=version,
                production_artifacts=promoted_artifacts
            )
            
            logger.info(f"Promoted version {version} for agent {agent_id} to production")
            
            return {
                "agent_id": agent_id,
                "version": version,
                "promoted_artifacts": promoted_artifacts,
                "promoted_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error promoting version {version} for agent {agent_id}: {e}")
            raise
    
    def rollback_version(
        self,
        agent_id: int,
        target_version: str
    ) -> Dict[str, Any]:
        """
        Откат к предыдущей версии
        
        Args:
            agent_id: ID агента
            target_version: Версия для отката
        
        Returns:
            Информация об откате
        """
        try:
            db = next(get_db())
            
            # Проверяем существование агента
            agent = orm_get_agent_by_id(db, agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Проверяем существование целевой версии
            target_artifacts = orm_get_artifacts_by_version(db, agent_id, target_version)
            if not target_artifacts:
                raise ValueError(f"No artifacts found for version {target_version}")
            
            # Выполняем откат
            result = self.promote_version(agent_id, target_version, force=True)
            result["rollback_from"] = agent.production_version
            result["rollback_to"] = target_version
            
            logger.info(f"Rolled back agent {agent_id} from {agent.production_version} to {target_version}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error rolling back agent {agent_id} to version {target_version}: {e}")
            raise
    
    def list_versions(
        self,
        agent_id: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Получение списка версий агента
        
        Args:
            agent_id: ID агента
            limit: Ограничение количества версий
        
        Returns:
            Список версий с информацией
        """
        try:
            db = next(get_db())
            
            # Получаем агента
            agent = orm_get_agent_by_id(db, agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Получаем все артефакты агента
            all_artifacts = orm_get_artifacts_by_agent(db, agent_id, limit=limit)
            
            # Группируем по версиям
            versions = {}
            for artifact in all_artifacts:
                if artifact.version not in versions:
                    versions[artifact.version] = {
                        "version": artifact.version,
                        "artifacts": [],
                        "is_production": artifact.version == agent.production_version,
                        "total_size": 0
                    }
                
                versions[artifact.version]["artifacts"].append({
                    "type": artifact.type,
                    "path": artifact.path,
                    "size_bytes": artifact.size_bytes
                })
                versions[artifact.version]["total_size"] += artifact.size_bytes or 0
            
            # Сортируем по версиям (новые сначала)
            sorted_versions = sorted(
                versions.values(),
                key=lambda x: x["version"],
                reverse=True
            )
            
            return sorted_versions
            
        except Exception as e:
            logger.error(f"Error listing versions for agent {agent_id}: {e}")
            raise
    
    def get_version_info(
        self,
        agent_id: int,
        version: str
    ) -> Dict[str, Any]:
        """
        Получение детальной информации о версии
        
        Args:
            agent_id: ID агента
            version: Версия
        
        Returns:
            Детальная информация о версии
        """
        try:
            db = next(get_db())
            
            # Получаем агента
            agent = orm_get_agent_by_id(db, agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Получаем артефакты версии
            artifacts = orm_get_artifacts_by_version(db, agent_id, version)
            if not artifacts:
                raise ValueError(f"No artifacts found for version {version}")
            
            # Читаем метаданные
            metadata = {}
            metadata_artifact = next((a for a in artifacts if a.type == "metadata"), None)
            if metadata_artifact and os.path.exists(metadata_artifact.path):
                try:
                    with open(metadata_artifact.path, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading metadata for version {version}: {e}")
            
            # Проверяем статус файлов
            artifacts_status = []
            for artifact in artifacts:
                file_exists = os.path.exists(artifact.path)
                artifacts_status.append({
                    "type": artifact.type,
                    "path": artifact.path,
                    "size_bytes": artifact.size_bytes,
                    "exists": file_exists,
                    "checksum": artifact.checksum
                })
            
            return {
                "version": version,
                "agent_id": agent_id,
                "agent_type": agent.agent_type,
                "is_production": version == agent.production_version,
                "artifacts": artifacts_status,
                "metadata": metadata,
                "total_size": sum(a.size_bytes or 0 for a in artifacts),
                "created_at": min(a.id for a in artifacts) if artifacts else None
            }
            
        except Exception as e:
            logger.error(f"Error getting version info for agent {agent_id}, version {version}: {e}")
            raise
    
    def delete_version(
        self,
        agent_id: int,
        version: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Удаление версии
        
        Args:
            agent_id: ID агента
            version: Версия для удаления
            force: Принудительное удаление (даже продакшн)
        
        Returns:
            Информация об удалении
        """
        try:
            db = next(get_db())
            
            # Получаем агента
            agent = orm_get_agent_by_id(db, agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Проверяем, не является ли версия продакшн
            if version == agent.production_version and not force:
                raise ValueError(f"Cannot delete production version {version} without force=True")
            
            # Получаем артефакты версии
            artifacts = orm_get_artifacts_by_version(db, agent_id, version)
            if not artifacts:
                raise ValueError(f"No artifacts found for version {version}")
            
            # Удаляем файлы
            deleted_files = []
            for artifact in artifacts:
                if os.path.exists(artifact.path):
                    try:
                        os.remove(artifact.path)
                        deleted_files.append(artifact.path)
                    except Exception as e:
                        logger.warning(f"Error deleting file {artifact.path}: {e}")
            
            # Удаляем записи из БД
            deleted_count = 0
            for artifact in artifacts:
                try:
                    orm_delete_artifact(db, artifact.id)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Error deleting artifact record {artifact.id}: {e}")
            
            logger.info(f"Deleted version {version} for agent {agent_id}: {deleted_count} artifacts, {len(deleted_files)} files")
            
            return {
                "agent_id": agent_id,
                "version": version,
                "deleted_artifacts": deleted_count,
                "deleted_files": deleted_files
            }
            
        except Exception as e:
            logger.error(f"Error deleting version {version} for agent {agent_id}: {e}")
            raise
    
    def get_production_status(self, agent_id: int) -> Dict[str, Any]:
        """
        Получение статуса продакшн версии
        
        Args:
            agent_id: ID агента
        
        Returns:
            Статус продакшн версии
        """
        try:
            db = next(get_db())
            
            # Получаем агента
            agent = orm_get_agent_by_id(db, agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            if not agent.production_version:
                return {
                    "agent_id": agent_id,
                    "has_production": False,
                    "production_version": None,
                    "production_artifacts": None
                }
            
            # Получаем информацию о продакшн версии
            version_info = self.get_version_info(agent_id, agent.production_version)
            
            return {
                "agent_id": agent_id,
                "has_production": True,
                "production_version": agent.production_version,
                "production_artifacts": agent.production_artifacts,
                "version_info": version_info
            }
            
        except Exception as e:
            logger.error(f"Error getting production status for agent {agent_id}: {e}")
            raise
    
    def cleanup_old_versions(
        self,
        agent_id: int,
        keep_versions: int = 5
    ) -> Dict[str, Any]:
        """
        Очистка старых версий
        
        Args:
            agent_id: ID агента
            keep_versions: Количество версий для сохранения
        
        Returns:
            Информация об очистке
        """
        try:
            # Получаем список версий
            versions = self.list_versions(agent_id)
            
            if len(versions) <= keep_versions:
                return {
                    "agent_id": agent_id,
                    "deleted_versions": [],
                    "kept_versions": len(versions),
                    "message": "No cleanup needed"
                }
            
            # Определяем версии для удаления (исключаем продакшн)
            agent = orm_get_agent_by_id(next(get_db()), agent_id)
            versions_to_delete = []
            
            for version_info in versions[keep_versions:]:
                if version_info["version"] != agent.production_version:
                    versions_to_delete.append(version_info["version"])
            
            # Удаляем версии
            deleted_versions = []
            for version in versions_to_delete:
                try:
                    result = self.delete_version(agent_id, version)
                    deleted_versions.append({
                        "version": version,
                        "result": result
                    })
                except Exception as e:
                    logger.warning(f"Error deleting version {version}: {e}")
            
            logger.info(f"Cleaned up {len(deleted_versions)} old versions for agent {agent_id}")
            
            return {
                "agent_id": agent_id,
                "deleted_versions": deleted_versions,
                "kept_versions": keep_versions,
                "total_versions_before": len(versions)
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up old versions for agent {agent_id}: {e}")
            raise
