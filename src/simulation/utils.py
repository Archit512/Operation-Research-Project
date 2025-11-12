"""
Utility Functions Module
=========================
Helper functions for calculating expected values, thresholds, and other utilities.
"""

import numpy as np
from scipy.stats import poisson
from functools import lru_cache


@lru_cache(maxsize=None)
def calculate_expected_contribution_next_day(inventory, mu_demand, p_price, c_holding, max_demand=30):
    """
    Calculates the simplified expected contribution (profit) from in-store sales
    for the next day (myopic view).
    
    EC_j = p * (Expected_Sales) - c_h * I_j
    
    Args:
        inventory (int): Current inventory level
        mu_demand (float): Average daily demand (Poisson parameter)
        p_price (float): Product selling price
        c_holding (float): Holding cost per unit per day
        max_demand (int): Upper limit for demand summation
        
    Returns:
        float: Expected contribution value
    """
    if inventory <= 0:
        return 0.0
        
    expected_sales = 0.0
    for d in range(max_demand + 1):
        prob = poisson.pmf(d, mu_demand)
        sales = min(inventory, d)
        expected_sales += prob * sales
        
    revenue = p_price * expected_sales
    holding_cost = c_holding * inventory
    
    return revenue - holding_cost


@lru_cache(maxsize=None)
def get_days_until_replenishment(t, L, R):
    """
    Calculates the number of days from sub-period t until the next
    replenishment arrives.
    
    Args:
        t (int): Current time period (1-indexed)
        L (int): Lead time (days)
        R (int): Review period length (days)
        
    Returns:
        int: Number of days until replenishment
    """
    if t <= L:
        # Need to cover demand for days t, t+1, ... L
        return L - t + 1
    else:
        # t > L. Replenishment has arrived.
        # Need to cover demand until next week's replenishment
        return (R - t + 1) + L


@lru_cache(maxsize=None)
def calculate_threshold(t, mu_demand, L, R, p_price, c_holding):
    """
    Calculates the threshold inventory w_j(t) to hold back for future demand.
    
    Based on the TMFP policy - stores should hold back inventory to meet
    expected future demand until the next replenishment.
    
    Args:
        t (int): Current time period
        mu_demand (float): Average daily demand
        L (int): Lead time
        R (int): Review period
        p_price (float): Product price
        c_holding (float): Holding cost
        
    Returns:
        int: Threshold inventory level
    """
    days_to_cover = get_days_until_replenishment(t, L, R)
    
    if days_to_cover <= 0:
        return 0
        
    # Demand from t up until replenishment
    mu_future_demand = mu_demand * days_to_cover
    
    # Maximum holding cost
    holding_cost_future = c_holding * days_to_cover
    
    # Target probability
    target_prob = p_price / (holding_cost_future + p_price)
    
    # Inverse of the cumulative distribution
    threshold = poisson.ppf(target_prob, mu_future_demand)
    return int(threshold)


@lru_cache(maxsize=None)
def dummy_value_function_vj(inventory, mu_demand, t, q, R, p_price, c_holding, max_demand=30):
    """
    Placeholder value function for OSIP policy.
    
    The real OSIP policy requires a pre-computed value function obtained
    from value iteration. This is a simple proxy that estimates the value
    as expected contribution for the rest of the period.
    
    Args:
        inventory (int): Current inventory
        mu_demand (float): Average demand
        t (int): Current time period
        q (int): Outstanding order quantity (unused in this simple version)
        R (int): Review period length
        p_price (float): Product price
        c_holding (float): Holding cost
        max_demand (int): Upper limit for demand summation
        
    Returns:
        float: Estimated value of the state
    """
    days_left_in_period = R - t + 1
    val = calculate_expected_contribution_next_day(inventory, mu_demand, p_price, c_holding, max_demand)
    return val * max(1, days_left_in_period)
