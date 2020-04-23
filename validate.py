from pydantic import BaseModel, validator, Field
from typing import List


class NestedSchema(BaseModel):
    pclass: int = Field(title="Pclass", description="Cabin Class", ge=1, le=3)
    name: str = Field(title="Name", description="Passenger name", max_length=100, default="Test")
    sex: str = Field(title="Sex", description="Passenger sex")
    sibsp: int = Field(title="SibSp", description="Number of Siblings/Spouses Aboard")
    parch: int = Field(title="Parch", description="Number of Parents/Children Aboard")
    embarked: str = Field(title="Embarked", description="Port of embarkation")
    fare: int = Field(title="Fare", description="Ticket cost", ge=1, le=200)
    age: int = Field(title="Age", description="Passenger Age", ge=1, le=100)

    @validator("sex")
    def name_validator(cls, sex):
        if sex not in {"male", "female"}:
            raise ValueError('Valid values for sex are: male, female.')
        return sex

    @validator("embarked")
    def embarked_validator(cls, embarked):
        if embarked not in {"S", "Q", "C"}:
            raise ValueError('Valid values for embarked are: S, Q, C.')
        return embarked


class ModelSchema(BaseModel):
    models: List = Field(title="Model", description="List of models")
    data: NestedSchema
