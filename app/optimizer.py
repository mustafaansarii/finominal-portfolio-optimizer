import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

def calculate_volatility(weights, cov_matrix):
    """Calculate annualized volatility (assuming 252 trading days)"""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

def calculate_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    """Calculate annualized Sharpe Ratio"""
    ret = np.sum(mean_returns * weights) * 252
    vol = calculate_volatility(weights, cov_matrix)
    return (ret - risk_free_rate) / vol

def calculate_max_drawdown(weights, returns_df):
    """Calculate maximum drawdown for a given set of weights and historical returns"""
    portfolio_returns = returns_df.dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min() # This is a negative number, so minimizing its negative means making it closer to 0

def risk_contribution(weights, cov_matrix):
    """Calculate risk contribution of each asset"""
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
    risk_contrib = weights * marginal_contrib
    return risk_contrib / portfolio_vol # normalize to sum to 1

def risk_parity_objective(weights, cov_matrix):
    """Objective function for risk parity: minimize variance of risk contributions"""
    rc = risk_contribution(weights, cov_matrix)
    target = 1.0 / len(weights)
    return np.sum(np.square(rc - target))

def calculate_factor_betas(portfolio_returns, factor_returns_df):
    """Calculate betas against Momentum, Value, Size factors using linear regression"""
    # Ensure indices align
    common_dates = portfolio_returns.index.intersection(factor_returns_df.index)
    if len(common_dates) == 0:
        return {"value": 0.0, "momentum": 0.0, "size": 0.0}
        
    y = portfolio_returns.loc[common_dates].values
    X = factor_returns_df.loc[common_dates][['Momentum Factor', 'Value Factor', 'Size Factor']].values
    
    # Linear Regression
    model = LinearRegression().fit(X, y)
    
    return {
        "momentum": float(model.coef_[0]),
        "value": float(model.coef_[1]),
        "size": float(model.coef_[2])
    }

class PortfolioOptimizer:
    def __init__(self, tickers: list[str], returns_df: pd.DataFrame, fund_info: dict, factor_df: pd.DataFrame = None):
        self.tickers = tickers
        self.returns_df = returns_df
        self.fund_info = fund_info
        self.factor_df = factor_df
        
        self.n = len(tickers)
        self.mean_returns = returns_df.mean().values
        self.cov_matrix = returns_df.cov().values
        
        self.dividend_yields = np.array([self.fund_info[t].get('dividend_yield', 0.0) for t in tickers])

    def run_optimization(self, strategy: str, constraints_req=None, factor_to_opt=None, maximize=True):
        # 1. Base constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # 2. Bounds: 0 <= w <= 1 (default) or apply custom min/max
        min_w = 0.0
        max_w = 1.0
        if constraints_req:
            if constraints_req.min_weight is not None:
                min_w = constraints_req.min_weight
            if constraints_req.max_weight is not None:
                max_w = constraints_req.max_weight
                
        bounds = tuple((min_w, max_w) for _ in range(self.n))
        
        # 3. Portfolio-level Constraints (e.g. min dividend yield)
        if constraints_req and constraints_req.min_dividend_yield is not None:
            min_div = constraints_req.min_dividend_yield
            # weight_1 * div_1 + weight_2 * div_2 >= min_div
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: np.sum(w * self.dividend_yields) - min_div
            })

        # Initial guess (Equal weights)
        init_guess = np.array(self.n * [1.0 / self.n])
        
        # Optimization strategies
        if strategy == "equal_weights":
            return init_guess # No optimization needed, just equal splits
            
        elif strategy == "minimize_volatility":
            res = minimize(
                calculate_volatility, 
                init_guess, 
                args=(self.cov_matrix,), 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )
            
        elif strategy == "maximize_sharpe_ratio":
            def neg_sharpe(w):
                return -calculate_sharpe_ratio(w, self.mean_returns, self.cov_matrix)
            res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            
        elif strategy == "minimize_drawdown":
            def neg_max_dd(w):
                # max drawdown is negative, e.g. -0.20. Minimizing -0.20 means making it closer to 0?
                # actually we want to minimize the depth of the drawdown, so we minimize the negative of the max drawdown (which is a positive number, depth)
                # or maximize the max drawdown (make it less negative).
                # calculate_max_drawdown returns a negative float like -0.45 (meaning 45% drop).
                # To minimize the drop (make it closer to 0), we minimize the absolute value, or minimize -calculate_max_drawdown
                return -calculate_max_drawdown(w, self.returns_df)
            res = minimize(neg_max_dd, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            
        elif strategy == "risk_parity":
            res = minimize(
                risk_parity_objective,
                init_guess,
                args=(self.cov_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
        elif strategy == "optimize_factor_exposure":
            if self.factor_df is None:
                raise ValueError("Factor returns data required for factor optimization")
                
            def factor_objective(w):
                port_ret = self.returns_df.dot(w)
                betas = calculate_factor_betas(port_ret, self.factor_df)
                
                # Map standard factor names to our keys
                key_map = {
                    "Momentum Factor": "momentum",
                    "Value Factor": "value",
                    "Size Factor": "size"
                }
                beta_val = betas.get(key_map.get(factor_to_opt, "momentum"), 0.0)
                
                # Minimize -beta to maximize it, or minimize beta to minimize it
                return -beta_val if maximize else beta_val
                
            res = minimize(factor_objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        if not res.success:
            raise ValueError(f"Optimization failed: {res.message}. Constraints might be infeasible.")
            
        return res.x
