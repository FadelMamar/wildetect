"""MLflow setup for an Ultralytics detection run."""


def configure_detection_tracking(tracking_uri: str) -> None:
    """Point MLflow and Ultralytics at the same tracking URI.

    Registration metadata is unchanged. This only selects where the training
    run is recorded.
    """
    import mlflow
    from ultralytics import settings

    mlflow.set_tracking_uri(tracking_uri)
    settings.update({"mlflow": True})
