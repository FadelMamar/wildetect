"""Visualization-related CLI commands."""

import traceback
import typer
from pathlib import Path
from omegaconf import OmegaConf

from ..config_loader import ConfigLoader
from ...visualization import Visualizer
from ...models.detector import Detector
from ...models.localizer import UltralyticsLocalizer
from ...models.classifier import GenericClassifier
from ...shared.config_types import ConfigType
from ...shared.validation import ConfigFileNotFoundError, ConfigParseError, ConfigValidationError
from .utils import console, setup_logging, log_file_path

visualize_app = typer.Typer(name="visualize", help="Visualization commands")


@visualize_app.command()
def classifier_predictions(
    config: Path = typer.Option("", "--config", "-c",help="Path to classification visualization configuration YAML file"),
    template: bool = typer.Option(False, "--template", "-t", help="Show default configuration template instead of visualization")
) -> None:
    """Upload classifier predictions to a FiftyOne dataset for visualization using YAML configuration."""
    if template:
        console.print(f"[bold green]Showing default classification visualization configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config(ConfigType.CLASSIFICATION_VISUALIZATION)
            console.print(f"\n[bold blue]Default classification visualization configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Loading classification visualization config from:[/bold green] {config}")
    
    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_classification_visualization_config(config)
        console.print(f"[bold green]✓[/bold green] Classification visualization configuration validated successfully")
        
        # Convert validated config back to DictConfig for backward compatibility
        cfg = OmegaConf.create(validated_config.model_dump())
        console.print("cfg:",cfg)
        
        # Extract configuration values
        dataset_name = cfg.dataset_name
        prediction_field = cfg.prediction_field
        batch_size = cfg.batch_size
        debug = cfg.debug
        
        console.print(f"[bold green]Uploading classifier predictions to FiftyOne dataset:[/bold green] {dataset_name}")
        
        log_file = log_file_path("visualize_classifier_predictions")
        setup_logging(log_file=log_file)

        if cfg.mlflow.alias is not None and cfg.mlflow.name is not None:
            assert cfg.weights is None, "weights should not be provided when using mlflow"
            console.print(f"[bold green]Loading classifier from MLflow:[/bold green] {cfg.mlflow.name} {cfg.mlflow.alias}")
            model = GenericClassifier.from_mlflow(name=cfg.mlflow.name,
                                                    alias=cfg.mlflow.alias,
                                                    dwnd_location=cfg.mlflow.dwnd_location,
                                                    mlflow_tracking_uri=cfg.mlflow.tracking_uri
                                                )
        else:
            console.print(f"[bold green]Loading classifier from config:[/bold green] {cfg.weights}")
            model = GenericClassifier.load_from_checkpoint(cfg.weights,map_location=cfg.device,label_to_class_map=cfg.label_to_class_map)

        visualizer = Visualizer(fiftyone_dataset_name=dataset_name)
        visualizer.add_predictions_from_classifier(model=model,prediction_field=prediction_field,batch_size=batch_size,debug=debug)
        console.print(f"[bold blue]Classifier predictions uploaded to FiftyOne dataset:[/bold blue] {dataset_name}")
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {traceback.format_exc()}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Visualization failed: {traceback.format_exc()}")
        raise typer.Exit(1)


@visualize_app.command()
def detector_predictions(
    config: Path = typer.Option("", "--config", "-c",help="Path to visualization configuration YAML file"),
    template: bool = typer.Option(False, "--template", "-t", help="Show default configuration template instead of visualization")
) -> None:
    """Upload detector predictions to a FiftyOne dataset for visualization using YAML configuration."""
    if template:
        console.print(f"[bold green]Showing default detection visualization configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config(ConfigType.DETECTION_VISUALIZATION)
            console.print(f"\n[bold blue]Default detection visualization configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Loading visualization config from:[/bold green] {config}")
    
    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_detection_visualization_config(config)
        console.print(f"[bold green]✓[/bold green] Detection visualization configuration validated successfully")
        
        # Convert validated config back to DictConfig for backward compatibility
        cfg = OmegaConf.create(validated_config.model_dump())
        console.print("cfg:",cfg)
        
        # Extract configuration values
        dataset_name = cfg.fiftyone.dataset_name
        prediction_field = cfg.fiftyone.prediction_field
                
        console.print(f"[bold green]Uploading detector predictions to FiftyOne dataset:[/bold green] {dataset_name}")
                
        # Create detector
        if cfg.mlflow.alias is not None and cfg.mlflow.name is not None:
            console.print(f"[bold green]Loading detector from MLflow:[/bold green] {cfg.mlflow.name} {cfg.mlflow.alias}")
            detector = Detector.from_mlflow(name=cfg.mlflow.name,
                                            alias=cfg.mlflow.alias,
                                            dwnd_location=cfg.mlflow.dwnd_location,
                                            mlflow_tracking_uri=cfg.mlflow.tracking_uri
                                        )
        else:
            console.print(f"[bold green]Loading detector from config:[/bold green] {cfg.localizer} {cfg.classifier_weights}")
            detector = Detector.from_config(localizer_config=cfg.localizer,
                                            classifier_ckpt=cfg.classifier_weights
                                        )
        
        log_file = log_file_path("visualize_detector_predictions")
        setup_logging(log_file=log_file)

        visualizer = Visualizer(fiftyone_dataset_name=dataset_name,
                                label_studio_url=cfg.label_studio.url,
                                label_studio_api_key=cfg.label_studio.api_key,
                                label_studio_project_id=cfg.label_studio.project_id
                                )
        visualizer.add_predictions_from_detector(detector=detector,
                                                imgsz=cfg.localizer.imgsz,
                                                prediction_field=prediction_field,
                                                batch_size=cfg.batch_size,
                                                debug=cfg.debug,
                                                model_tag=cfg.label_studio.model_tag
                                                )
        console.print(f"[bold blue]Detector predictions uploaded to FiftyOne dataset:[/bold blue] {dataset_name}")
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {traceback.format_exc()}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Visualization failed: {traceback.format_exc()}")
        raise typer.Exit(1)
