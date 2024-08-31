from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import APIRouter, FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from inference import get_model
from pydantic import BaseModel
import base64
import numpy as np
import cv2

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
model = get_model(model_id="teeth_annotation_sl_techno/1", api_key="tlU5DJCpixYaQTqSGhMx")


# DTO for the image request
class TeethRequestDto(BaseModel):
    teeth_image: str
    confidence: float = 0.2


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
         # Decode the base64 image string to binary data
        img_data = base64.b64decode(dto.teeth_image)
        # Convert binary data to a NumPy array
        nparr = np.frombuffer(img_data, np.uint8)
        # Decode the NumPy array into an image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Image decoding failed")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Perform inference
        result = model.infer(img, confidence= dto.confidence)

        # Extract predictions
        predictions = result[0].predictions
        x_min = []
        y_min = []
        width = []
        height = []
        teeth_number = []
        results = []

        # Iterate through detections
        for pred in predictions:
            x_min.append(pred.x - pred.width / 2)
            y_min.append(pred.y - pred.height / 2)
            width.append(pred.width)
            height.append(pred.height)
            teeth_number.append(pred.class_id)
            results.append(
                {
                    "x": float(pred.x - float(pred.width / 2)),
                    "y": float(pred.y - float(pred.height / 2)),
                    "width": float(pred.width),
                    "height": float(pred.height),
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
            results=results,
        )
    except Exception as e:
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
    uvicorn.run(app, host="0.0.0.0")


#tlU5DJCpixYaQTqSGhMx