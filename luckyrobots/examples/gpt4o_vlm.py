"""
Uses gpt4o-mini model to zero-shot the robot from images.

pip3 install openai
pip3 install requests
"""
import luckyrobots as lr
import requests
import os
import base64

# Set your OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')

@lr.on("robot_output")
def handle_file_created(robot_images: list):
    if robot_images:
        if isinstance(robot_images, dict) and 'rgb_cam1' in robot_images:
            image_path = robot_images['rgb_cam1'].get('file_path')
            if image_path:
                print(f"Processing image: {image_path}")

                # Read and encode the image to base64
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')

                # Create data URI for the image
                data_uri = f"data:image/jpeg;base64,{base64_image}"

                # Prepare the messages payload
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """
You are the vision system for a robot, output a single token with one of the following commands:
W to go forward, S to go backwards, A to turn left, D to turn right
"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_uri
                                }
                            }
                        ]
                    }
                ]

                # Prepare headers and payload for the API request
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }

                payload = {
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "max_tokens": 10,
                }

                try:
                    # Make the API request to OpenAI
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=payload
                    )

                    response_json = response.json()
                    if response.status_code == 200:
                        output_text = response_json['choices'][0]['message']['content']
                        print(f"GPT-4 Response: {output_text}")
                        lr.send_message([[f"{output_text} 3600 1"]])
                    else:
                        print(f"Error {response.status_code}: {response_json}")
                except Exception as e:
                    print(f"Error communicating with OpenAI API: {str(e)}")
            else:
                print("No file_path found in rgb_cam1")
        else:
            print("Unexpected structure in robot_images")
    else:
        print("No robot_images received")

@lr.on("start")
def start():
    print("Starting")
    lr.send_message(["RESET"])

@lr.on("task_complete")
def handle_tasks_complete(id):
    print(f"Task complete: {id}")

@lr.on("firehose")
def handle_firehose(data):
    pass  # You can process firehose data here if needed

if __name__ == "__main__":
    lr.start()
