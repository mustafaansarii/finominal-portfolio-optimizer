from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import numpy as np

from .data_loader import data_store
from .schemas import OptimizationRequest, OptimizationResponse, AllocationChange, FactorBetasComparison, FactorBetas
from .optimizer import PortfolioOptimizer, calculate_factor_betas

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup: Data loaded successfully.")
    yield

app = FastAPI(title="Finominal Portfolio Optimizer API", lifespan=lifespan)

@app.post("/optimize", response_model=OptimizationResponse)
def optimize_portfolio(request: OptimizationRequest):
    try:
        tickers = request.tickers
        
        missing = [t for t in tickers if t not in data_store.fund_info]
        if missing:
            raise HTTPException(status_code=400, detail=f"Data missing for tickers: {missing}")

        fund_info = data_store.get_fund_info(tickers)
        returns_df = data_store.get_returns_data(tickers)
        
        if len(returns_df) < 30:
            raise HTTPException(status_code=400, detail="Not enough overlapping historical return data for the requested tickers.")

        factor_df = data_store.get_factor_data(returns_df.index)

        optimizer = PortfolioOptimizer(
            tickers=tickers, 
            returns_df=returns_df, 
            fund_info=fund_info,
            factor_df=factor_df
        )

        optimized_weights = optimizer.run_optimization(
            strategy=request.strategy.value,
            constraints_req=request.constraints,
            factor_to_opt=request.factor_to_optimize,
            maximize=request.maximize_factor
        )

        if request.current_weights:
            current_weights = np.array([request.current_weights.get(t, 1.0/len(tickers)) for t in tickers])
        else:
            current_weights = np.array([1.0 / len(tickers)] * len(tickers))

        allocation_changes = []
        for i, ticker in enumerate(tickers):
            cur_w_pct = round(current_weights[i] * 100, 2)
            opt_w_pct = round(optimized_weights[i] * 100, 2)
            
            allocation_changes.append(
                AllocationChange(
                    ticker=ticker,
                    security_name=fund_info[ticker]['fund_name'],
                    current_weight=cur_w_pct,
                    optimized_weight=opt_w_pct,
                    change=round(opt_w_pct - cur_w_pct, 2)
                )
            )

        cur_port_returns = returns_df.dot(current_weights)
        opt_port_returns = returns_df.dot(optimized_weights)
        
        cur_betas = calculate_factor_betas(cur_port_returns, factor_df)
        opt_betas = calculate_factor_betas(opt_port_returns, factor_df)
        
        factor_betas_cmp = FactorBetasComparison(
            current_portfolio=FactorBetas(**{k: round(v, 2) for k, v in cur_betas.items()}),
            optimized_portfolio=FactorBetas(**{k: round(v, 2) for k, v in opt_betas.items()})
        )

        return OptimizationResponse(
            optimization_strategy=request.strategy.value,
            allocation_changes=allocation_changes,
            factor_betas=factor_betas_cmp
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

