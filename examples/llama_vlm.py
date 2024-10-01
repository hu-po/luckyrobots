"""
Uses Llama 3.2 11B Vision model to zero-shot the robot from images.

pip3 install accelerate
pip3 install git+https://github.com/huggingface/transformers.git@main
huggingface-cli login

sim takes 4.5GB of GPU memory
llama takes 22GB of GPU memory
"""

import luckyrobots as lr
import cv2
import numpy as np
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

prompt = """<|image|><|begin_of_text|>You are a robot, type W to go forward, S to go backwards"""


@lr.on("robot_output")
def handle_file_created(robot_images: list):
    if robot_images:
        
        if isinstance(robot_images, dict) and 'rgb_cam1' in robot_images:
            image_path = robot_images['rgb_cam1'].get('file_path')
            if image_path:
                print(f"Processing image: {image_path}")
                
                results = model(image_path)
                image = results[0].plot()
                
                if isinstance(image, np.ndarray):
                    # If image is already a numpy array, use it directly
                    img = image
                else:
                    try:
                        # Convert bytes to numpy array if necessary
                        nparr = np.frombuffer(image, np.uint8)
                        # Decode the image
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    except Exception as e:
                        print(f"Error decoding image: {str(e)}")
                        return

                if img is None:
                    print("Failed to decode image")
                    return
                
                # Display the image
                cv2.imshow('Robot Image', img)

                inputs = processor(image, prompt, return_tensors="pt").to(model.device)

                output = model.generate(**inputs, max_new_tokens=2)
                print(processor.decode(output[0]))
                
                cv2.waitKey(1)  # Wait for 1ms to allow the image to be displayed
            else:
                print("No file_path found in rgb_cam1")
        else:
            print("Unexpected structure in robot_images")
    else:
        print("No robot_images received")                


@lr.on("start")
def start():
    print("Starting")
    commands = [
        ["RESET"],
        {"commands":[{"id":123456, "code":"w 5650 1"}, {"id":123457, "code":"a 30 1"}], "batchID": "123456"},
        ["A 0 1", "W 1800 1"],
        ["w 2500 1", "d 30 1", "EX1 10", "EX2 10", "G 100 1", {"id":123458, "code":"g -100 1"}],
        ["w 3000 1", "a 0 1", "u 100"],
        ["u -200"]
    ]
    lr.send_message(commands)



@lr.on("task_complete")
def handle_tasks_complete(id):
    print(f"Task complete: {id}")

@lr.on("firehose")
def handle_firehose(data):
    # print(f"Firehose: {data}")
    pass

if __name__ == "__main__":
    lr.start()
