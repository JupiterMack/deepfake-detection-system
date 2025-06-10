# src/deepfake_detector/api/schemas.py

"""
Defines the Pydantic schemas for the API request and response models.

These schemas are used by FastAPI for data validation, serialization,
and documentation. They ensure that the data flowing in and out of the
API conforms to a predefined structure.
"""

from pydantic import BaseModel, Field
from typing import List, Optional

# ----------------------------------------------------------------------------
# Generic Schemas
# ----------------------------------------------------------------------------

class Message(BaseModel):
    """
    A generic schema for returning simple messages or error details.
    """
    detail: str = Field(
        ...,
        description="A detail message for the client.",
        example="Operation completed successfully."
    )


# ----------------------------------------------------------------------------
# Detection Endpoint Schemas
# ----------------------------------------------------------------------------

class DetectionResponse(BaseModel):
    """
    Schema for the response of a successful deepfake detection request.
    """
    filename: str = Field(
        ...,
        description="The name of the processed file.",
        example="test_image_01.jpg"
    )

    is_fake: bool = Field(
        ...,
        description="The prediction result. True if the media is classified as fake, False otherwise.",
        example=True
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The model's confidence in the 'is_fake' prediction. "
                    "A value closer to 1.0 indicates higher confidence.",
        example=0.987
    )

    explanation_map: Optional[List[List[float]]] = Field(
        None,
        description="A 2D list (heatmap) from the explainable AI module. "
                    "This highlights the regions of the media that most influenced the prediction. "
                    "This field may be null if explanation generation was not requested or failed.",
        example=[
            [0.1, 0.8, 0.2],
            [0.2, 0.9, 0.3],
            [0.1, 0.85, 0.25]
        ]
    )

    class Config:
        """
        Pydantic model configuration.
        Provides an example for the API documentation.
        """
        json_schema_extra = {
            "example": {
                "filename": "user_upload.png",
                "is_fake": True,
                "confidence": 0.991,
                "explanation_map": [
                    [0.01, 0.02, 0.89, 0.11],
                    [0.03, 0.05, 0.91, 0.15],
                    [0.02, 0.04, 0.95, 0.12],
                    [0.01, 0.01, 0.85, 0.09]
                ]
            }
        }
```