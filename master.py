import subprocess

# Run person detection script
person_process = subprocess.Popen(['python', 'person_detection.py'])

# Run helmet detection script
helmet_process = subprocess.Popen(['python', 'helmet_detection.py'])

# Wait for both processes to complete
person_process.wait()
helmet_process.wait()
import torch
print(torch.__version__)
