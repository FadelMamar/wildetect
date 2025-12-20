import sys
from wildtrain.evaluators.classification import ClassificationEvaluator
from omegaconf import OmegaConf

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a classification model.")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation config YAML file.")
    parser.add_argument("--save-path", type=str, default=None, help="Optional path to save evaluation results as JSON.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (evaluate only a few batches).")
    args = parser.parse_args()

    evaluator = ClassificationEvaluator(args.config)
    results = evaluator.evaluate(debug=args.debug, save_path=args.save_path)
    print("Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v}") 