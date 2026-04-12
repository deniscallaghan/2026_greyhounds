"""Training API: create experiments, trigger training, view results."""

import logging
import os
from threading import Thread
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db, SessionLocal
from app.models.experiment import Experiment
from app.models.feature_definition import FeatureDefinition
from app.schemas.experiment import ExperimentCreate, ExperimentResponse
from ml.autoresearch import OBJECTIVE_DIRECTIONS
from ml.trainers.base import BaseTrainer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["training"])


class DefaultParamsResponse(BaseModel):
    params: dict


@router.get("/experiments", response_model=list[ExperimentResponse])
def list_experiments(
    status: str | None = None,
    target: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    query = db.query(Experiment)
    if status:
        query = query.filter(Experiment.status == status)
    if target:
        query = query.filter(Experiment.target == target)
    return query.order_by(Experiment.created_at.desc()).offset(offset).limit(limit).all()


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.get("/default-params/{algorithm}", response_model=DefaultParamsResponse)
def get_default_params(algorithm: str):
    """Get default hyperparameters for an algorithm."""
    params = BaseTrainer.get_default_params(algorithm)
    if not params:
        raise HTTPException(status_code=404, detail=f"Unknown algorithm: {algorithm}")
    return DefaultParamsResponse(params=params)


@router.post("/experiments", response_model=ExperimentResponse, status_code=201)
def create_experiment(experiment: ExperimentCreate, db: Session = Depends(get_db)):
    """Create an experiment and start training in the background."""
    # Use default params if none provided
    hyperparams = experiment.hyperparameters
    if not hyperparams:
        hyperparams = BaseTrainer.get_default_params(experiment.algorithm)

    db_experiment = Experiment(
        name=experiment.name,
        description=experiment.description,
        algorithm=experiment.algorithm,
        target=experiment.target,
        hyperparameters=hyperparams,
        feature_set=experiment.feature_set,
        split_config=experiment.split_config,
        status="pending",
    )
    db.add(db_experiment)
    db.commit()
    db.refresh(db_experiment)

    exp_id = db_experiment.id
    use_optuna = experiment.auto_tune
    optuna_trials = experiment.optuna_trials

    # Start training in background
    def _train():
        db2 = SessionLocal()
        try:
            from app.services.training_service import run_training, run_optuna_optimization
            if use_optuna:
                run_optuna_optimization(db2, exp_id, n_trials=optuna_trials)
            else:
                run_training(db2, exp_id)
        finally:
            db2.close()

    Thread(target=_train, daemon=True).start()

    return db_experiment


@router.delete("/experiments/{experiment_id}", status_code=204)
def delete_experiment(experiment_id: int, db: Session = Depends(get_db)):
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    db.delete(experiment)
    db.commit()

    # Clean up model artifact on disk
    model_path = os.path.join(
        settings.model_artifacts_dir, f"experiment_{experiment_id}.joblib"
    )
    if os.path.isfile(model_path):
        try:
            os.remove(model_path)
            logger.info("Deleted model artifact: %s", model_path)
        except OSError as e:
            logger.warning("Failed to delete model artifact %s: %s", model_path, e)


# ---------------------------------------------------------------------------
# Autoresearch endpoints
# ---------------------------------------------------------------------------


class AutoResearchRequest(BaseModel):
    objective: str = "betting_kelly_roi"
    algorithm: str = "lightgbm"
    target: str = "win_prob"
    feature_ids: list[int] | None = None  # None = all enabled features
    max_experiments: int = 100
    patience: int = 20
    split_config: dict[str, Any] | None = None


class AutoResearchStatusResponse(BaseModel):
    running: bool
    total_experiments: int | None = None
    total_improvements: int | None = None
    best_score: float | None = None
    best_algorithm: str | None = None
    objective: str | None = None


# In-memory state for the running autoresearch job
_autoresearch_state: dict[str, Any] = {
    "running": False,
    "summary": None,
}


@router.get("/autoresearch/objectives")
def list_autoresearch_objectives():
    """List available optimization objectives."""
    return {
        obj: "maximize" if higher else "minimize"
        for obj, higher in OBJECTIVE_DIRECTIONS.items()
    }


@router.post("/autoresearch/start", status_code=202)
def start_autoresearch(req: AutoResearchRequest):
    """Start an autoresearch loop in the background."""
    if _autoresearch_state["running"]:
        raise HTTPException(
            status_code=409,
            detail="Autoresearch is already running",
        )

    if req.objective not in OBJECTIVE_DIRECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown objective '{req.objective}'. Options: {list(OBJECTIVE_DIRECTIONS.keys())}",
        )

    _autoresearch_state["running"] = True
    _autoresearch_state["summary"] = None

    def _run():
        from ml.autoresearch import AutoResearchLoop

        db = SessionLocal()
        try:
            # Resolve features
            feature_ids = req.feature_ids
            if not feature_ids:
                features = (
                    db.query(FeatureDefinition)
                    .filter(FeatureDefinition.enabled.is_(True))
                    .all()
                )
                feature_ids = [f.id for f in features]

            all_features = (
                db.query(FeatureDefinition)
                .filter(FeatureDefinition.enabled.is_(True))
                .all()
            )
            all_feature_ids = [f.id for f in all_features]

            loop = AutoResearchLoop(
                db=db,
                feature_ids=feature_ids,
                objective=req.objective,
                algorithm=req.algorithm,
                target=req.target,
                split_config=req.split_config,
                all_feature_ids=all_feature_ids,
            )
            summary = loop.run(
                max_experiments=req.max_experiments,
                patience=req.patience,
            )
            _autoresearch_state["summary"] = summary
        except Exception as e:
            _autoresearch_state["summary"] = {"error": str(e)}
        finally:
            _autoresearch_state["running"] = False
            db.close()

    Thread(target=_run, daemon=True).start()

    return {"message": "Autoresearch started", "objective": req.objective}


@router.get("/autoresearch/status", response_model=AutoResearchStatusResponse)
def autoresearch_status():
    """Check autoresearch status and results."""
    summary = _autoresearch_state.get("summary")
    if summary and "error" not in summary:
        return AutoResearchStatusResponse(
            running=_autoresearch_state["running"],
            total_experiments=summary.get("total_experiments"),
            total_improvements=summary.get("total_improvements"),
            best_score=summary.get("best_score"),
            best_algorithm=summary.get("best_algorithm"),
            objective=summary.get("objective"),
        )
    return AutoResearchStatusResponse(running=_autoresearch_state["running"])


@router.get("/autoresearch/results")
def autoresearch_results():
    """Get full autoresearch results (after completion)."""
    if _autoresearch_state["running"]:
        return {"status": "running", "message": "Autoresearch is still running"}
    summary = _autoresearch_state.get("summary")
    if not summary:
        return {"status": "idle", "message": "No autoresearch has been run"}
    return {"status": "completed", **summary}
