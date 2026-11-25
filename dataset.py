from roboflow import Roboflow
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use the API key from the environment variable
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

project = rf.workspace("sammit").project(
    "face-detection-mik1i-twori"
)

dataset = project.version(1).download("yolov8")

print(dataset.location)