#!/usr/bin/env python

import numpy as np
from scipy.stats import poisson
import math
from functools import lru_cache

# ---
# 1. PARAMETERS (Corrected from preprocessed_product_parameters.csv)
# ---
# These parameters are for the product "Milk" and are
# used to set the simulation's constants.
params = {
  "product_simulated": "Milk",
  "mu_0_online_demand": 39.86094584,  # Network-wide online demand
  "total_instore_demand": 738.0,       # TOTAL in-store demand across ALL stores
  "p_price": 7.6,
  "c_p_product_cost": 5.32,
  "c_h_holding_cost": 1.0,
  "c_s_shipping_cost": 10.0,
  "c_l_penalty_cost": 10.0,
  "R_period_days": 7,
  "L_lead_time_days": 2,
  
  # Store configuration
  "num_stores_small": 10,
  "num_stores_medium": 10,
  "num_stores_large": 10,
  
  # Demand distribution ratios (based on typical retail patterns)
  # These determine how the total 738 units/day is distributed
  "demand_ratio": {
    "small": 1.0,   # Small stores get baseline demand
    "medium": 4.0,  # Medium stores get 4x small store demand
    "large": 6.0    # Large stores get 6x small store demand
  }
}

# Calculate per-store demand based on distribution
def calculate_per_store_demand():
    """
    Distribute the total in-store demand (738 units/day) across all stores
    based on their relative sizes.
    """
    total_demand = params['total_instore_demand']
    
    # Calculate total demand units (weighted sum)
    small_count = params['num_stores_small']
    medium_count = params['num_stores_medium']
    large_count = params['num_stores_large']
    
    small_ratio = params['demand_ratio']['small']
    medium_ratio = params['demand_ratio']['medium']
    large_ratio = params['demand_ratio']['large']
    
    total_demand_units = (small_count * small_ratio + 
                          medium_count * medium_ratio + 
                          large_count * large_ratio)
    
    # Calculate per-store demand for each type
    demand_per_unit = total_demand / total_demand_units
    
    return {
        "small": demand_per_unit * small_ratio,
        "medium": demand_per_unit * medium_ratio,
        "large": demand_per_unit * large_ratio
    }

per_store_demand = calculate_per_store_demand()

params['store_types'] = {
    "small": {"mu_j": per_store_demand['small']},
    "medium": {"mu_j": per_store_demand['medium']},
    "large": {"mu_j": per_store_demand['large']}
}

print("\n" + "="*70)
print("CORRECTED DEMAND DISTRIBUTION")
print("="*70)
print(f"Total in-store demand for Milk: {params['total_instore_demand']} units/day")
print(f"Network configuration:")
print(f"  - {params['num_stores_small']} small stores × {per_store_demand['small']:.2f} units/day = {params['num_stores_small'] * per_store_demand['small']:.2f} units")
print(f"  - {params['num_stores_medium']} medium stores × {per_store_demand['medium']:.2f} units/day = {params['num_stores_medium'] * per_store_demand['medium']:.2f} units")
print(f"  - {params['num_stores_large']} large stores × {per_store_demand['large']:.2f} units/day = {params['num_stores_large'] * per_store_demand['large']:.2f} units")
print(f"  TOTAL: {params['num_stores_small'] * per_store_demand['small'] + params['num_stores_medium'] * per_store_demand['medium'] + params['num_stores_large'] * per_store_demand['large']:.2f} units/day")
print("="*70 + "\n")

# ---
# 2. GLOBAL CONSTANTS
# ---
P_PRICE = params['p_price']
C_HOLDING = params['c_h_holding_cost']
C_PRODUCT = params['c_p_product_cost']
C_SHIPPING = params['c_s_shipping_cost']
C_PENALTY = params['c_l_penalty_cost']

R_PERIOD_DAYS = params['R_period_days']
L_LEAD_TIME_DAYS = params['L_lead_time_days']

MU_0_ONLINE = params['mu_0_online_demand']

# Practical upper bound for demand summation
MAX_DEMAND_STORE = int(params['store_types']['large']['mu_j'] * 3)

# ---
# 3. STORE CLASS
# ---
class Store:
    """Represents a single retail store."""
    def __init__(self, id, type, mu_demand, initial_inventory):
        self.id = id
        self.type = type
        self.mu_demand = mu_demand
        self.inventory = initial_inventory
        self.outstanding_order = 0
    
    def __repr__(self):
        return f"Store {self.id} ({self.type}): mu_j={self.mu_demand:.2f}, Inv={self.inventory}, Q={self.outstanding_order}"

# ---
# 4. HELPER FUNCTIONS
# ---

@lru_cache(maxsize=None)
def calculate_expected_contribution_next_day(inventory, mu_demand):
    """
    Calculates the simplified expected contribution (profit) from in-store sales
    for the *next day* (myopic view).
    """
    if inventory <= 0:
        return 0.0
        
    expected_sales = 0.0
    upper_demand_limit = max(50, int(mu_demand + 4 * np.sqrt(mu_demand)))
    
    for d in range(upper_demand_limit + 1):
        prob = poisson.pmf(d, mu_demand)
        sales = min(inventory, d)
        expected_sales += prob * sales
        
    revenue = P_PRICE * expected_sales
    holding_cost = C_HOLDING * inventory
    
    return revenue - holding_cost

@lru_cache(maxsize=None)
def get_days_until_replenishment(t, L, R):
    """Calculates the number of days from sub-period t until replenishment."""
    if t <= L:
        return L - t + 1
    else:
        return (R - t + 1) + L

@lru_cache(maxsize=None)
def calculate_threshold(t, mu_demand):
    """Calculates the threshold inventory w_j(t) to hold back."""
    days_to_cover = get_days_until_replenishment(t, L_LEAD_TIME_DAYS, R_PERIOD_DAYS)
    
    if days_to_cover <= 0:
        return 0
        
    mu_future_demand = mu_demand * days_to_cover
    holding_cost_future = C_HOLDING * days_to_cover
    
    if (holding_cost_future + P_PRICE) == 0:
        return 0
        
    target_prob = P_PRICE / (holding_cost_future + P_PRICE)
    threshold = poisson.ppf(target_prob, mu_future_demand)
    return int(threshold)

# ---
# 5. FULFILLMENT POLICIES
# ---

def mfp_fulfillment_policy(stores, F_online_orders):
    """Myopic Fulfillment Policy (MFP)"""
    current_inventories = {s.id: s.inventory for s in stores}
    allocations = {s.id: 0 for s in stores}
    
    for _ in range(F_online_orders):
        best_store_id = None
        best_marginal_gain = -float('inf')
        
        for s in stores:
            inv = current_inventories[s.id]
            if inv > 0:
                ec_with_fulfill = calculate_expected_contribution_next_day(inv - 1, s.mu_demand)
                ec_without_fulfill = calculate_expected_contribution_next_day(inv, s.mu_demand)
                marginal_oc = ec_with_fulfill - ec_without_fulfill
                gain = (P_PRICE - C_SHIPPING) + C_PENALTY + marginal_oc
                
                if gain > best_marginal_gain:
                    best_marginal_gain = gain
                    best_store_id = s.id
        
        if best_marginal_gain > 0:
            allocations[best_store_id] += 1
            current_inventories[best_store_id] -= 1
        else:
            break
            
    return allocations

def tmfp_fulfillment_policy(stores, F_online_orders, t):
    """Threshold-based Myopic Policy (TMFP)"""
    current_inventories = {s.id: s.inventory for s in stores}
    allocations = {s.id: 0 for s in stores}
    
    thresholds = {}
    for s in stores:
        if s.mu_demand not in thresholds:
            thresholds[s.mu_demand] = calculate_threshold(t, s.mu_demand)
    
    for _ in range(F_online_orders):
        best_store_id = None
        best_marginal_gain = -float('inf')
        
        for s in stores:
            inv = current_inventories[s.id]
            threshold = thresholds[s.mu_demand]
            
            if inv > threshold:
                ec_with_fulfill = calculate_expected_contribution_next_day(inv - 1, s.mu_demand)
                ec_without_fulfill = calculate_expected_contribution_next_day(inv, s.mu_demand)
                marginal_oc = ec_with_fulfill - ec_without_fulfill
                gain = (P_PRICE - C_SHIPPING) + C_PENALTY + marginal_oc
                
                if gain > best_marginal_gain:
                    best_marginal_gain = gain
                    best_store_id = s.id
        
        if best_marginal_gain > 0:
            allocations[best_store_id] += 1
            current_inventories[best_store_id] -= 1
        else:
            break
            
    return allocations

@lru_cache(maxsize=None)
def dummy_value_function_vj(inventory, mu_demand, t, q):
    """Placeholder value function for OSIP"""
    days_left_in_period = R_PERIOD_DAYS - t + 1
    val = calculate_expected_contribution_next_day(inventory, mu_demand)
    return val * days_left_in_period

def osip_fulfillment_policy(stores, F_online_orders, t, value_function):
    """One-Step Policy Improvement (OSIP)"""
    dp_cache = {}

    def solve_dp(store_idx, orders_left):
        if store_idx == len(stores) or orders_left == 0:
            return (0.0, [])
        
        state = (store_idx, orders_left)
        if state in dp_cache:
            return dp_cache[state]

        s = stores[store_idx]
        best_value = -float('inf')
        best_allocation = []
        max_f_j = min(orders_left, s.inventory)
        
        for f_j in range(max_f_j + 1):
            immediate_profit_fj = (P_PRICE - C_SHIPPING) * f_j
            replenishment = s.outstanding_order if t == L_LEAD_TIME_DAYS else 0
            future_inv = s.inventory - f_j + replenishment
            
            future_value_vj = value_function(
                future_inv, s.mu_demand, 
                (t % R_PERIOD_DAYS) + 1, s.outstanding_order
            )
            
            value_from_rest, alloc_from_rest = solve_dp(store_idx + 1, orders_left - f_j)
            current_total_value = immediate_profit_fj + future_value_vj + value_from_rest
            
            if current_total_value > best_value:
                best_value = current_total_value
                best_allocation = [f_j] + alloc_from_rest
        
        dp_cache[state] = (best_value, best_allocation)
        return best_value, best_allocation

    best_total_profit = -float('inf')
    final_allocations = {}
    
    for F_to_fulfill in range(F_online_orders + 1):
        penalty_cost = C_PENALTY * (F_online_orders - F_to_fulfill)
        dp_cache = {}
        total_value_from_dp, alloc_list = solve_dp(0, F_to_fulfill)
        total_profit = total_value_from_dp - penalty_cost
        
        if total_profit > best_total_profit:
            best_total_profit = total_profit
            padded_alloc = alloc_list + [0] * (len(stores) - len(alloc_list))
            final_allocations = {stores[i].id: padded_alloc[i] for i in range(len(stores))}

    return final_allocations

# ---
# 6. SIMULATION HARNESS
# ---

def run_simulation():
    """
    Runs a simulation for one R-day period using corrected demand distribution.
    """
    print("--- 🚚 Omni-Channel Simulation Start (CORRECTED) ---")
    print(f"Running simulation for: {params['product_simulated']}")
    print(f"  Online Demand (mu_0): {MU_0_ONLINE:.2f}")
    print(f"  Price (p): {P_PRICE}, Cost (c_p): {C_PRODUCT}")

    # Create stores with corrected demand
    # For simulation, we'll use a subset: 1 small, 2 medium, 1 large
    store_types = params['store_types']
    store_list = [
        Store(id=1, type="small", mu_demand=store_types['small']['mu_j'], initial_inventory=30),
        Store(id=2, type="medium", mu_demand=store_types['medium']['mu_j'], initial_inventory=120),
        Store(id=3, type="medium", mu_demand=store_types['medium']['mu_j'], initial_inventory=120),
        Store(id=4, type="large", mu_demand=store_types['large']['mu_j'], initial_inventory=180)
    ]
    
    print("\nInitial State (Sample of 4 stores):")
    for s in store_list: print(f"  {s}")

    # Set initial orders proportional to demand
    store_list[0].outstanding_order = int(store_types['small']['mu_j'] * 7)      # 7 days worth
    store_list[1].outstanding_order = int(store_types['medium']['mu_j'] * 7)
    store_list[2].outstanding_order = int(store_types['medium']['mu_j'] * 7)
    store_list[3].outstanding_order = int(store_types['large']['mu_j'] * 7)
    
    np.random.seed(42)
    
    # Loop through the 7-day period
    for t in range(1, R_PERIOD_DAYS + 1):
        print(f"\n--- Day t={t} (Epoch 1) ---")
        
        if t == 1:
            print("Ordering decision made (using preset Q).")

        # In-store demand
        print("In-store demand occurs:")
        for s in store_list:
            demand = poisson.rvs(s.mu_demand)
            sold = min(s.inventory, demand)
            s.inventory -= sold
            print(f"  {s.id} ({s.type}): mu={s.mu_demand:.2f}, Demand={demand}, Sold={sold}, Inv={s.inventory}")

        # Online orders
        F_online_orders = poisson.rvs(MU_0_ONLINE)
        print(f"Online orders received: F = {F_online_orders} (from mu_0={MU_0_ONLINE:.2f})")

        print(f"\n--- Day t={t} (Epoch 2) ---")
        print("Making online fulfillment decisions...")

        # Create copies for policies
        stores_mfp = [Store(s.id, s.type, s.mu_demand, s.inventory) for s in store_list]
        stores_tmfp = [Store(s.id, s.type, s.mu_demand, s.inventory) for s in store_list]
        stores_osip = [Store(s.id, s.type, s.mu_demand, s.inventory) for s in store_list]
        for i in range(len(store_list)):
            stores_osip[i].outstanding_order = store_list[i].outstanding_order

        # Run policies
        alloc_mfp = mfp_fulfillment_policy(stores_mfp, F_online_orders)
        alloc_tmfp = tmfp_fulfillment_policy(stores_tmfp, F_online_orders, t)
        alloc_osip = osip_fulfillment_policy(stores_osip, F_online_orders, t, dummy_value_function_vj)
        
        print(f"  MFP Policy:  {alloc_mfp}")
        print(f"  TMFP Policy: {alloc_tmfp}")
        print(f"  OSIP Policy: {alloc_osip} (Using dummy v_j)")

        # Apply OSIP policy
        print("Applying OSIP policy for state transition...")
        for s in store_list:
            f_j = alloc_osip[s.id]
            s.inventory -= f_j
            
            if t == L_LEAD_TIME_DAYS:
                print(f"  Store {s.id}: Replenishment {s.outstanding_order} arrived!")
                s.inventory += s.outstanding_order
                s.outstanding_order = 0
        
        print("End of day state (I_j''):")
        for s in store_list: print(f"  {s}")

    print("\n--- 🚚 Simulation End ---")

if __name__ == "__main__":
    run_simulation()
