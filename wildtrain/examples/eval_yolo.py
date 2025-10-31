import yaml
import sys
from wildtrain.evaluators.ultralytics import UltralyticsEvaluator


config = "D:/workspace/repos/wildtrain/configs/detection/yolo_eval.yaml"
# Instantiate the evaluator
evaluator = UltralyticsEvaluator(config=config)

# Run evaluation
print("Running YOLO evaluation...")
results = evaluator.evaluate(debug=True)

# Print results
print("End.:")
# print(results)



