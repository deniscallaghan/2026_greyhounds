"""Purge all computed features and feature versions.

This does NOT delete original scraped data (races, dogs, race_entries,
tracks, odds_snapshots, etc.) — only the derived computed_features and
feature_versions tables are cleared.  Features can be recomputed from
the scraped data at any time via the materialize endpoint.

Also truncates the SQLite WAL file to reclaim disk space.

Usage:
    python scripts/cleanup_features.py          # interactive confirmation
    python scripts/cleanup_features.py --yes    # skip confirmation
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import engine
from sqlalchemy.orm import Session
from sqlalchemy import func, text


def cleanup_features(skip_confirm: bool = False) -> None:
    from app.models.computed_feature import ComputedFeature
    from app.models.feature_version import FeatureVersion

    with Session(engine) as db:
        feature_count = db.query(func.count(ComputedFeature.id)).scalar() or 0
        version_count = db.query(func.count(FeatureVersion.id)).scalar() or 0

        print(f"Found {feature_count:,} computed features across {version_count} versions.")

        if feature_count == 0 and version_count == 0:
            print("Nothing to clean up.")
            return

        if not skip_confirm:
            answer = input("Delete ALL computed features and versions? (y/N): ")
            if answer.lower() != "y":
                print("Aborted.")
                return

        # Delete computed features first (FK constraint)
        deleted = db.query(ComputedFeature).delete()
        print(f"Deleted {deleted:,} computed features.")

        deleted_v = db.query(FeatureVersion).delete()
        print(f"Deleted {deleted_v} feature versions.")

        db.commit()

    # Truncate WAL to reclaim disk space
    try:
        raw_conn = engine.raw_connection()
        raw_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        raw_conn.close()
        print("WAL file truncated.")
    except Exception as e:
        print(f"WAL checkpoint skipped: {e}")

    # Also clean up orphaned model artifacts
    _cleanup_orphaned_models()

    print("Done. Scraped data (races, dogs, entries, tracks) is untouched.")


def _cleanup_orphaned_models() -> None:
    """Delete .joblib files that don't correspond to any existing experiment."""
    from app.models.experiment import Experiment
    from app.config import settings

    model_dir = settings.model_artifacts_dir
    if not os.path.isdir(model_dir):
        return

    with Session(engine) as db:
        experiment_ids = {
            row[0] for row in db.query(Experiment.id).all()
        }

    removed = 0
    for fname in os.listdir(model_dir):
        if not fname.endswith(".joblib"):
            continue
        # Expected format: experiment_{id}.joblib
        try:
            exp_id = int(fname.replace("experiment_", "").replace(".joblib", ""))
        except ValueError:
            continue
        if exp_id not in experiment_ids:
            path = os.path.join(model_dir, fname)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            os.remove(path)
            removed += 1
            print(f"  Removed orphaned model: {fname} ({size_mb:.1f} MB)")

    if removed:
        print(f"Cleaned up {removed} orphaned model artifact(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Purge computed features and versions")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    args = parser.parse_args()
    cleanup_features(skip_confirm=args.yes)
