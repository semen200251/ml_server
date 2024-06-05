from pydantic import BaseModel

__all__ = ["ImageRequest", "IncomingRequestPreprocessing", "IncomingRequestPredict", "OutputRequest"]

class ImageRequest(BaseModel):
    taskId: int

class IncomingRequestPreprocessing(ImageRequest):
    imageUrl: str

class IncomingRequestPredict(ImageRequest):
    image_tensor: str

class OutputRequest(ImageRequest):
    classId: int