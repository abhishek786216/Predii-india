from typing import Optional
from pydantic import BaseModel, Field


class VehicleSpecification(BaseModel):
    specification_name: str
    value: str
    unit: Optional[str] = None
    vehicle_model: Optional[str] = None
    part_number: Optional[str] = None
    source_page: Optional[int] = None
    source_text: Optional[str] = None
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0
    )


class ExtractionResult(BaseModel):
    specifications: list[VehicleSpecification]