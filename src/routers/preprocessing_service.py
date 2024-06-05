from fastapi import APIRouter, HTTPException, status
import requests
import httpx
from PIL import Image
from io import BytesIO
import base64
import torchvision.transforms as transforms
from src.schemas import IncomingRequestPreprocessing, OutputRequest
from src.configuration.config import PREDICT_URL

preprocessing_router = APIRouter(
    tags=["preprocess"],
    prefix="/preprocess"
)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@preprocessing_router.post("/", response_model=OutputRequest, status_code=status.HTTP_200_OK)
def preprocess_image(request: IncomingRequestPreprocessing):
    try:
        response = requests.get(request.imageUrl)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")

        input_tensor = preprocess(image)
        input_tensor = input_tensor.unsqueeze(0)

        tensor_bytes = input_tensor.numpy().tobytes()
        tensor_base64 = base64.b64encode(tensor_bytes).decode('utf-8')
        
        payload = {
            "taskId": request.taskId,
            "image_tensor": tensor_base64
        }
        print("good")
        model_response = requests.post(PREDICT_URL, json=payload)
        model_response.raise_for_status()
        result = model_response.json()

        return result

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    