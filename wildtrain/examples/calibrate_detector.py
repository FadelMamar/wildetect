from wildtrain.evaluators.calibrator import DetectionCalibrator


calibration_config_path = r"D:\PhD\workspace\wildetect\wildtrain\configs\detection\yolo_configs\calibration_example.yaml"
def main(calibration_config_path:str=calibration_config_path):
    calibrator = DetectionCalibrator(calibration_config_path=calibration_config_path,debug=False)
    calibrator.run()

if __name__ == "__main__":
    main()