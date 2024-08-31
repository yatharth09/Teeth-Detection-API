from typing import Any, Dict, List, Optional
import base64
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model_path = 'best_yatarth.pt'  # Update this path
model = YOLO(model_path)


# DTO for the image request
class TeethRequestDto(BaseModel):
    teeth_image: str  # Base64 encoded image
    confidence: float = 0.2  # Default confidence threshold


# DTO for the response of the process-image endpoint
class TeethResponseDto(BaseModel):
    x_min: List[float]
    y_min: List[float]
    width: List[float]
    height: List[float]
    teeth_number: List[int]
    success: bool
    error: Optional[str]
    results: Optional[List[Dict[str, Any]]] = []


# Initialize router
router = APIRouter()


@router.post("/process-image/", response_model=TeethResponseDto)
async def process_image(dto: TeethRequestDto) -> TeethResponseDto:
    try:
        # Decode the base64 image
        with open(dto.teeth_image, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        img_data = base64.b64decode(encoded_image)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model(img, conf=dto.confidence)

        # Extract predictions
        x_min = []
        y_min = []
        width = []
        height = []
        teeth_number = []
        results_list = []

        # Iterate through detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
                cls = box.cls[0].item()  # Class label

                x_min.append(xyxy[0])
                y_min.append(xyxy[1])
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                width.append(w)
                height.append(h)
                teeth_number.append(int(cls))
                results_list.append(
                    {
                        "x_min": float(xyxy[0]),
                        "y_min": float(xyxy[1]),
                        "width": float(w),
                        "height": float(h),
                    }
                )

        return TeethResponseDto(
            x_min=x_min,
            y_min=y_min,
            width=width,
            height=height,
            teeth_number=teeth_number,
            success=True,
            error="",
            results=results_list,
        )
    except Exception as e:
        print(e)
        return TeethResponseDto(
            x_min=[],
            y_min=[],
            width=[],
            height=[],
            teeth_number=[],
            success=False,
            error=str(e),
            results=[],
        )


# Include the router
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8111)
