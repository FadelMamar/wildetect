from wildtrain.evaluators.ultralytics import UltralyticsEvaluator
from wildtrain.shared.models import DetectionEvalConfig

if __name__ == "__main__":
    config = r"wildtrain\configs\detection\yolo_configs\yolo_eval.yaml"
    # Instantiate the evaluator
    evaluator = UltralyticsEvaluator(config=DetectionEvalConfig.from_yaml(config))

    # Run evaluation
    print("Running YOLO evaluation...")
    results = evaluator.evaluate(debug=False)

    # Print results
    # print("End.:")

    print("Results:", results)
    print("Confusion Matrix:", evaluator.get_confusion_matrix())
