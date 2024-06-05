import requests
import threading

def send_request(task_id, image_url):
    preprocessing_service_url = "http://localhost:8000/preprocess/"
    payload = {
        "taskId": task_id,
        "imageUrl": image_url
    }
    response = requests.post(preprocessing_service_url, json=payload)
    print(f"Task {task_id} response: {response.json()}")

image_url = "https://drive.google.com/uc?export=download&id=1ur-XxKqALeRTtGDOH0flI22il_ca68nm"

threads = []
for i in range(0, 100):
    thread = threading.Thread(target=send_request, args=(i, image_url))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()