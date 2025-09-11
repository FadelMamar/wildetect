from wildetect.core.visualization import LabelStudioManager
from wildetect.core.config import FlightSpecs

client = LabelStudioManager(url="http://localhost:8080", 
                            api_key="edc3c4539d7b0708a2c189be47ff94708d1de3cc",
                            )

annotations = client.get_all_project_detections(project_id=7,load_as_drone_image=True,
                                                flight_specs=FlightSpecs(sensor_height=24,focal_length=35,flight_height=180))

annot = annotations[1]


