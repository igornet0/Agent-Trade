from typing import Literal, Optional, List
from pydantic import BaseModel, Field, field_validator


class StrategyCreate(BaseModel):
    name: str
    type: str = Field(default="test")  # test | trade
    risk: float = Field(default=0.05)
    reward: float = Field(default=0.05)

    coins: List[int] = Field(default_factory=list)
    agents: List[int] = Field(default_factory=list)

    model_risk_id: Optional[int] = None
    model_order_id: Optional[int] = None

    @field_validator("risk", "reward")
    @classmethod
    def _ratio_bounds(cls, v: float):
        if not 0 <= v <= 1:
            raise ValueError("must be between 0 and 1")
        return v

    @field_validator("coins", "agents")
    @classmethod
    def _non_negative_ids(cls, v: List[int]):
        if any((i is None or i <= 0) for i in v):
            raise ValueError("ids must be positive integers")
        return v


class StrategyModelsUpdate(BaseModel):
    model_risk_id: Optional[int] = None
    model_order_id: Optional[int] = None


class StrategyResponse(BaseModel):
    id: int
    type: Literal["train", "test", "trade"]

class CreateStrategyResponse(StrategyResponse):
    pass
