import os
import requests
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
)

def download_file(url, weight_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024

    with open(weight_path, 'wb') as file, Progress(
        SpinnerColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[progress.description]{task.description}", justify="right"),
    ) as progress:
        task = progress.add_task(f"{weight_path} is Downloading...", total=total_size)
        for data in response.iter_content(chunk_size=chunk_size):
            file.write(data)
            progress.update(task, advance=len(data))
            
def download_weights_goldyolo():
    weight_path = "./gold_yolo_l_head_post_0277_0.5353_1x3x480x640.onnx"
    url = "https://github.com/TadaLabInfo/DrivingEvaluationPrograms/releases/download/onxx/gold_yolo_l_head_post_0277_0.5353_1x3x480x640.onnx"
    download_file(url, weight_path)
    
def download_weights_6drepnet():
    weight_path = "./sixdrepnet360_1x3x224x224_full.onnx"
    url = "https://github.com/TadaLabInfo/DrivingEvaluationPrograms/releases/download/onxx/sixdrepnet360_1x3x224x224_full.onnx"
    download_file(url, weight_path)
    
if __name__ == "__main__":
    download_weights_goldyolo()
    download_weights_6drepnet()
    print("Weights download completed.")