from pydantic import BaseModel, Field


class IrisInput(BaseModel):
    sepal_length: float = Field(ge=0, le=10)
    sepal_width: float = Field(ge=0, le=10)
    petal_length: float = Field(ge=0, le=10)
    petal_width: float = Field(ge=0, le=10)


class IrisOutput(BaseModel):
    species: str
    probabilities: dict[str, float]
