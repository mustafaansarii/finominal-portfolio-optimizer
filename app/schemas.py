from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from enum import Enum

class StrategyEnum(str, Enum):
    equal_weights = "equal_weights"
    risk_parity = "risk_parity"
    minimize_drawdown = "minimize_drawdown"
    minimize_volatility = "minimize_volatility"
    maximize_sharpe_ratio = "maximize_sharpe_ratio"
    optimize_factor_exposure = "optimize_factor_exposure"

class Constraints(BaseModel):
    min_weight: Optional[float] = Field(None, description="Minimum weight per security, e.g., 0.05 for 5%")
    max_weight: Optional[float] = Field(None, description="Maximum weight per security, e.g., 0.40 for 40%")
    min_dividend_yield: Optional[float] = Field(None, description="Portfolio-level minimum dividend yield, e.g., 0.025 for 2.5%")

class OptimizationRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of ticker symbols to optimize")
    current_weights: Optional[Dict[str, float]] = Field(None, description="Optional starting weights (0.0 to 1.0) for the tickers")
    strategy: StrategyEnum
    constraints: Optional[Constraints] = None
    
    # For strategy 6 (bonus)
    factor_to_optimize: Optional[str] = Field("Momentum Factor", description="E.g., Momentum Factor, Value Factor, Size Factor")
    maximize_factor: Optional[bool] = Field(True, description="True to maximize, False to minimize exposure")
    
    @validator('tickers')
    def validate_tickers(cls, v):
        if not v or len(v) < 2:
            raise ValueError("At least two tickers are required for optimization.")
        return [ticker.upper() for ticker in v]

class AllocationChange(BaseModel):
    ticker: str
    security_name: str
    current_weight: float
    optimized_weight: float
    change: float

class FactorBetas(BaseModel):
    value: float
    momentum: float
    size: float

class FactorBetasComparison(BaseModel):
    current_portfolio: FactorBetas
    optimized_portfolio: FactorBetas

class OptimizationResponse(BaseModel):
    optimization_strategy: str
    allocation_changes: List[AllocationChange]
    factor_betas: Optional[FactorBetasComparison] = None
