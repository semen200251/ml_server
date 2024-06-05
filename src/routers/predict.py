import base64
import torch
import numpy as np
from torchvision import models
from fastapi import APIRouter, HTTPException, status
from src.schemas import IncomingRequestPredict, OutputRequest
from src.configuration.config import PATH_TO_MODEL

predict_router = APIRouter(
    tags=["predict"],
    prefix="/predict"
)

print("gsdag")

model = models.resnet101()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 27)
model.load_state_dict(torch.load(PATH_TO_MODEL, map_location=torch.device('cpu')))
model.eval()


@predict_router.post("/", response_model=OutputRequest, status_code=status.HTTP_200_OK)
def predict(request: IncomingRequestPredict):
    print("predict")
    try:
        tensor_bytes = base64.b64decode(request.image_tensor)
        np_array = np.frombuffer(tensor_bytes, dtype=np.float32).copy()
        input_tensor = torch.from_numpy(np_array).reshape((1, 3, 224, 224))
        with torch.no_grad():
            output = model(input_tensor)
        
        _, predicted_idx = torch.max(output, 1)
        predicted_class_id = predicted_idx.item()

        return {"taskId": request.taskId, "classId": predicted_class_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")