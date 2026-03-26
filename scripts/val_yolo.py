from ultralytics import YOLO
import fire
import yaml


def main(cfg:str):

    with open(cfg) as f:
        config = yaml.safe_load(f)
    
    # Load a YOLO model from a pretrained weights file
    model = YOLO(config.pop('model'))

    # Run the model in MODE using custom ARGS
    MODE = "val"    
    getattr(model, MODE)(**config)

if __name__ == "__main__":
    fire.Fire(main)