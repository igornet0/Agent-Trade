"""
ORM методы для работы с артефактами моделей
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from core.database.models.main_models import Artifact
from core.database.models.ML_models import Agent


def orm_create_artifact(
    db: Session,
    agent_id: int,
    version: str,
    path: str,
    artifact_type: str,
    size_bytes: Optional[int] = None,
    checksum: Optional[str] = None
) -> Artifact:
    """Создание нового артефакта"""
    artifact = Artifact(
        agent_id=agent_id,
        version=version,
        path=path,
        type=artifact_type,
        size_bytes=size_bytes,
        checksum=checksum
    )
    
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    
    return artifact


def orm_get_artifact_by_id(db: Session, artifact_id: int) -> Optional[Artifact]:
    """Получение артефакта по ID"""
    return db.query(Artifact).filter(Artifact.id == artifact_id).first()


def orm_get_artifacts_by_agent(
    db: Session, 
    agent_id: int, 
    artifact_type: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Artifact]:
    """Получение артефактов агента"""
    query = db.query(Artifact).filter(Artifact.agent_id == agent_id)
    
    if artifact_type:
        query = query.filter(Artifact.type == artifact_type)
    
    query = query.order_by(desc(Artifact.id))  # Используем ID вместо created_at
    
    if limit:
        query = query.limit(limit)
    
    return query.all()


def orm_get_latest_artifact(
    db: Session, 
    agent_id: int, 
    artifact_type: str
) -> Optional[Artifact]:
    """Получение последнего артефакта определенного типа для агента"""
    return db.query(Artifact).filter(
        and_(
            Artifact.agent_id == agent_id,
            Artifact.type == artifact_type
        )
    ).order_by(desc(Artifact.id)).first()  # Используем ID вместо created_at


def orm_get_artifacts_by_version(
    db: Session, 
    agent_id: int, 
    version: str
) -> List[Artifact]:
    """Получение всех артефактов определенной версии агента"""
    return db.query(Artifact).filter(
        and_(
            Artifact.agent_id == agent_id,
            Artifact.version == version
        )
    ).all()


def orm_delete_artifact(db: Session, artifact_id: int) -> bool:
    """Удаление артефакта"""
    artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
    if artifact:
        db.delete(artifact)
        db.commit()
        return True
    return False


def orm_delete_artifacts_by_agent(
    db: Session, 
    agent_id: int, 
    artifact_type: Optional[str] = None
) -> int:
    """Удаление всех артефактов агента (или определенного типа)"""
    query = db.query(Artifact).filter(Artifact.agent_id == agent_id)
    
    if artifact_type:
        query = query.filter(Artifact.type == artifact_type)
    
    count = query.count()
    query.delete()
    db.commit()
    
    return count


def orm_get_artifact_stats(db: Session, agent_id: int) -> Dict[str, Any]:
    """Получение статистики артефактов агента"""
    artifacts = db.query(Artifact).filter(Artifact.agent_id == agent_id).all()
    
    stats = {
        "total_count": len(artifacts),
        "total_size_bytes": sum(a.size_bytes or 0 for a in artifacts),
        "by_type": {},
        "versions": list(set(a.version for a in artifacts))
    }
    
    # Группировка по типам
    for artifact in artifacts:
        artifact_type = artifact.type
        if artifact_type not in stats["by_type"]:
            stats["by_type"][artifact_type] = {
                "count": 0,
                "size_bytes": 0,
                "latest_version": None
            }
        
        stats["by_type"][artifact_type]["count"] += 1
        stats["by_type"][artifact_type]["size_bytes"] += artifact.size_bytes or 0
        
        # Используем ID для определения последней версии
        if (stats["by_type"][artifact_type]["latest_version"] is None or 
            artifact.id > max(a.id for a in artifacts if a.type == artifact_type)):
            stats["by_type"][artifact_type]["latest_version"] = artifact.version
    
    return stats


def orm_cleanup_old_artifacts(
    db: Session, 
    agent_id: int, 
    keep_versions: int = 3
) -> int:
    """Очистка старых версий артефактов, оставляя только последние N версий"""
    # Получаем все версии агента, отсортированные по ID
    versions = db.query(Artifact.version, Artifact.id).filter(
        Artifact.agent_id == agent_id
    ).distinct().order_by(desc(Artifact.id)).all()
    
    if len(versions) <= keep_versions:
        return 0
    
    # Определяем версии для удаления
    versions_to_delete = [v.version for v in versions[keep_versions:]]
    
    # Удаляем артефакты старых версий
    deleted_count = db.query(Artifact).filter(
        and_(
            Artifact.agent_id == agent_id,
            Artifact.version.in_(versions_to_delete)
        )
    ).delete()
    
    db.commit()
    return deleted_count
