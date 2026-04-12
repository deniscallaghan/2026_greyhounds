"""Feature definition CRUD + preview + materialization endpoints."""

import logging
from threading import Thread
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.models.feature_definition import FeatureDefinition
from app.schemas.feature import (
    FeatureDefinitionCreate,
    FeatureDefinitionResponse,
    FeatureDefinitionUpdate,
    FeaturePreviewRequest,
    FeaturePreviewResponse,
)
from app.services.feature_engine import get_dog_history, compute_visual_feature
from app.services.feature_sandbox import execute_feature_code, validate_feature_code

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/features", tags=["features"])


class MaterializeRequest(BaseModel):
    feature_ids: list[int] | None = None  # None = all enabled
    force: bool = False
    # When set, computed features are saved under a named version snapshot.
    # Create the version first via POST /features/versions, then pass its id.
    version_id: int | None = None


class MaterializeResponse(BaseModel):
    message: str
    results: dict[str, Any] | None = None


class CreateVersionRequest(BaseModel):
    name: str
    description: str | None = None


# Maximum number of feature versions to keep.  When a new version is created,
# the oldest versions beyond this limit are automatically deleted along with
# their computed features.
MAX_FEATURE_VERSIONS = 3


class VersionResponse(BaseModel):
    id: int
    name: str
    description: str | None
    created_at: Any
    coverage_snapshot: Any | None
    feature_count: int = 0
    model_config = {"from_attributes": True}


class FeatureCoverageItem(BaseModel):
    feature_id: int
    name: str
    display_name: str | None
    feature_type: str
    enabled: bool
    computed_count: int
    incomplete_count: int = 0
    total_entries: int
    coverage_pct: float


class DataIntegrityRequest(BaseModel):
    start_date: str | None = None
    end_date: str | None = None
    max_gap_days: int = 14


@router.get("/", response_model=list[FeatureDefinitionResponse])
def list_features(enabled_only: bool = False, db: Session = Depends(get_db)):
    query = db.query(FeatureDefinition)
    if enabled_only:
        query = query.filter(FeatureDefinition.enabled.is_(True))
    return query.order_by(FeatureDefinition.name).all()


@router.get("/coverage", response_model=list[FeatureCoverageItem])
def get_coverage(version_id: int | None = None, db: Session = Depends(get_db)):
    """Get computation coverage stats for all features, optionally filtered by version."""
    from ml.feature_store import get_feature_coverage
    return get_feature_coverage(db, version_id=version_id)


@router.get("/data-integrity")
def get_data_integrity(
    start_date: str | None = None,
    end_date: str | None = None,
    max_gap_days: int = 14,
    db: Session = Depends(get_db),
):
    """
    Check data completeness before materializing features.

    Reports scrape coverage gaps that could cause features to be computed
    with incomplete dog histories — e.g. if a dog raced at Dublin and
    Limerick but only Limerick data has been scraped, rolling features
    like 'mean last 5 race times' would silently be wrong.

    Returns a recommendation of "safe", "warning", or "incomplete".
    """
    from datetime import date as date_type
    from ml.data_integrity import assess_materialization_readiness

    sd = date_type.fromisoformat(start_date) if start_date else None
    ed = date_type.fromisoformat(end_date) if end_date else None

    return assess_materialization_readiness(db, sd, ed, max_gap_days)


@router.post("/data-integrity")
def post_data_integrity(req: DataIntegrityRequest, db: Session = Depends(get_db)):
    """POST variant of data-integrity check."""
    from datetime import date as date_type
    from ml.data_integrity import assess_materialization_readiness

    sd = date_type.fromisoformat(req.start_date) if req.start_date else None
    ed = date_type.fromisoformat(req.end_date) if req.end_date else None

    return assess_materialization_readiness(db, sd, ed, req.max_gap_days)


@router.get("/versions")
def list_versions(db: Session = Depends(get_db)):
    """List all feature versions, newest first."""
    from sqlalchemy import func as sqlfunc
    from app.models.feature_version import FeatureVersion
    from app.models.computed_feature import ComputedFeature

    count_sub = (
        db.query(
            ComputedFeature.version_id,
            sqlfunc.count(ComputedFeature.id).label("feature_count"),
        )
        .group_by(ComputedFeature.version_id)
        .subquery()
    )

    rows = (
        db.query(FeatureVersion, sqlfunc.coalesce(count_sub.c.feature_count, 0))
        .outerjoin(count_sub, FeatureVersion.id == count_sub.c.version_id)
        .order_by(FeatureVersion.created_at.desc())
        .all()
    )

    return [
        {
            "id": v.id,
            "name": v.name,
            "description": v.description,
            "created_at": v.created_at,
            "coverage_snapshot": v.coverage_snapshot,
            "feature_count": count,
        }
        for v, count in rows
    ]


def _enforce_version_retention(db: Session) -> None:
    """Delete the oldest feature versions beyond MAX_FEATURE_VERSIONS."""
    from app.models.feature_version import FeatureVersion
    from app.models.computed_feature import ComputedFeature

    all_versions = (
        db.query(FeatureVersion)
        .order_by(FeatureVersion.created_at.desc())
        .all()
    )
    if len(all_versions) <= MAX_FEATURE_VERSIONS:
        return

    to_delete = all_versions[MAX_FEATURE_VERSIONS:]
    for v in to_delete:
        db.query(ComputedFeature).filter(ComputedFeature.version_id == v.id).delete()
        db.delete(v)
        logger.info("Retention policy: deleted feature version '%s' (id=%d)", v.name, v.id)
    db.commit()


@router.post("/versions", status_code=201)
def create_version(req: CreateVersionRequest, db: Session = Depends(get_db)):
    """
    Create a named feature version (snapshot).

    This captures a data-integrity snapshot at creation time so you can
    see the scrape coverage that was in effect when features were computed.
    After creating, pass the returned id to POST /features/materialize
    to compute features into this version.
    """
    from app.models.feature_version import FeatureVersion
    from ml.data_integrity import assess_materialization_readiness

    existing = db.query(FeatureVersion).filter(FeatureVersion.name == req.name).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Version '{req.name}' already exists")

    snapshot = assess_materialization_readiness(db)

    version = FeatureVersion(
        name=req.name,
        description=req.description,
        coverage_snapshot=snapshot,
    )
    db.add(version)
    db.commit()
    db.refresh(version)

    # Enforce retention policy — delete oldest versions beyond the limit
    _enforce_version_retention(db)

    return {
        "id": version.id,
        "name": version.name,
        "description": version.description,
        "created_at": version.created_at,
        "coverage_snapshot": version.coverage_snapshot,
        "feature_count": 0,
    }


@router.get("/versions/{version_id}")
def get_version(version_id: int, db: Session = Depends(get_db)):
    """Get details for a specific feature version."""
    from sqlalchemy import func as sqlfunc
    from app.models.feature_version import FeatureVersion
    from app.models.computed_feature import ComputedFeature

    version = db.query(FeatureVersion).filter(FeatureVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    try:
        count = (
            db.query(sqlfunc.count(ComputedFeature.id))
            .filter(ComputedFeature.version_id == version.id)
            .scalar() or 0
        )
        incomplete = (
            db.query(sqlfunc.count(ComputedFeature.id))
            .filter(
                ComputedFeature.version_id == version.id,
                ComputedFeature.data_complete.is_(False),
            )
            .scalar() or 0
        )
    except Exception:
        db.rollback()
        count = 0
        incomplete = 0

    return {
        "id": version.id,
        "name": version.name,
        "description": version.description,
        "created_at": version.created_at,
        "coverage_snapshot": version.coverage_snapshot,
        "feature_count": count,
        "incomplete_count": incomplete,
    }


@router.delete("/versions/{version_id}", status_code=204)
def delete_version(version_id: int, db: Session = Depends(get_db)):
    """Delete a feature version and all its computed features."""
    from app.models.feature_version import FeatureVersion
    from app.models.computed_feature import ComputedFeature

    version = db.query(FeatureVersion).filter(FeatureVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    db.query(ComputedFeature).filter(ComputedFeature.version_id == version.id).delete()
    db.delete(version)
    db.commit()


@router.get("/start-materialize")
def start_materialize_get(force: bool = False, db: Session = Depends(get_db)):
    """GET endpoint to trigger materialization from browser URL bar."""
    req = MaterializeRequest(force=force)
    return trigger_materialization(req, db)


@router.get("/{feature_id}", response_model=FeatureDefinitionResponse)
def get_feature(feature_id: int, db: Session = Depends(get_db)):
    feature = db.query(FeatureDefinition).filter(FeatureDefinition.id == feature_id).first()
    if not feature:
        raise HTTPException(status_code=404, detail="Feature not found")
    return feature


@router.post("/", response_model=FeatureDefinitionResponse, status_code=201)
def create_feature(feature: FeatureDefinitionCreate, db: Session = Depends(get_db)):
    # Validate code features
    if feature.feature_type == "code" and feature.code:
        error = validate_feature_code(feature.code)
        if error:
            raise HTTPException(status_code=400, detail=f"Invalid code: {error}")

    db_feature = FeatureDefinition(**feature.model_dump())
    db.add(db_feature)
    db.commit()
    db.refresh(db_feature)
    return db_feature


@router.patch("/{feature_id}", response_model=FeatureDefinitionResponse)
def update_feature(feature_id: int, update: FeatureDefinitionUpdate, db: Session = Depends(get_db)):
    db_feature = db.query(FeatureDefinition).filter(FeatureDefinition.id == feature_id).first()
    if not db_feature:
        raise HTTPException(status_code=404, detail="Feature not found")
    for key, value in update.model_dump(exclude_unset=True).items():
        setattr(db_feature, key, value)
    db.commit()
    db.refresh(db_feature)
    return db_feature


@router.delete("/{feature_id}", status_code=204)
def delete_feature(feature_id: int, db: Session = Depends(get_db)):
    db_feature = db.query(FeatureDefinition).filter(FeatureDefinition.id == feature_id).first()
    if not db_feature:
        raise HTTPException(status_code=404, detail="Feature not found")
    db.delete(db_feature)
    db.commit()


@router.post("/preview", response_model=FeaturePreviewResponse)
def preview_feature(req: FeaturePreviewRequest, db: Session = Depends(get_db)):
    """
    Preview a feature value for a specific dog without saving.
    Uses the dog's most recent race as the race context.
    """
    from app.models.race import Race
    from app.models.race_entry import RaceEntry

    # Get the dog's most recent race entry for context
    latest_entry = (
        db.query(RaceEntry)
        .join(Race)
        .filter(RaceEntry.dog_id == req.dog_id, Race.status == "resulted")
        .order_by(Race.race_date.desc())
        .first()
    )

    if not latest_entry:
        return FeaturePreviewResponse(error="No race history found for this dog")

    from app.services.feature_engine import get_race_context
    ctx = get_race_context(db, latest_entry.id)
    if not ctx:
        return FeaturePreviewResponse(error="Could not build race context")

    history = get_dog_history(db, req.dog_id, ctx["race_date"])
    if history.empty:
        return FeaturePreviewResponse(error="No prior race history for this dog")

    if req.feature_type == "visual":
        config = req.config_json or {}
        value = compute_visual_feature(history, config, ctx)
        return FeaturePreviewResponse(value=value)

    elif req.feature_type == "code":
        if not req.code:
            return FeaturePreviewResponse(error="No code provided")

        error = validate_feature_code(req.code)
        if error:
            return FeaturePreviewResponse(error=error)

        value, error = execute_feature_code(req.code, history, ctx)
        return FeaturePreviewResponse(value=value, error=error)

    return FeaturePreviewResponse(error=f"Unknown feature_type: {req.feature_type}")


@router.post("/materialize", response_model=MaterializeResponse)
def trigger_materialization(req: MaterializeRequest, db: Session = Depends(get_db)):
    """
    Trigger feature materialization in the background.

    If version_id is provided, features are saved into that named snapshot.
    Otherwise they are saved unversioned (upserted in place).
    """
    from ml.feature_store import materialize_feature

    if req.version_id is not None:
        from app.models.feature_version import FeatureVersion
        version = db.query(FeatureVersion).filter(FeatureVersion.id == req.version_id).first()
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")

    if req.feature_ids:
        features = db.query(FeatureDefinition).filter(FeatureDefinition.id.in_(req.feature_ids)).all()
        if not features:
            raise HTTPException(status_code=404, detail="No features found")
    else:
        features = db.query(FeatureDefinition).filter(FeatureDefinition.enabled.is_(True)).all()

    if not features:
        return MaterializeResponse(message="No enabled features to materialize")

    version_id = req.version_id

    def _run():
        db2 = SessionLocal()
        try:
            for f in features:
                feat = db2.query(FeatureDefinition).filter(FeatureDefinition.id == f.id).first()
                if feat:
                    try:
                        materialize_feature(
                            db2, feat, force=req.force, version_id=version_id,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to materialize feature '%s' (id=%d), "
                            "continuing with remaining features",
                            f.name, f.id,
                        )
                        db2.rollback()
        finally:
            db2.close()

    thread = Thread(target=_run, daemon=True)
    thread.start()

    version_msg = f" into version {version_id}" if version_id else ""
    return MaterializeResponse(
        message=f"Materialization started for {len(features)} features{version_msg} in background",
    )

