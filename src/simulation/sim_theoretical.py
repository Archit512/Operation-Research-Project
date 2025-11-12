#!/usr/bin/env python

import numpy as np
from scipy.stats import poisson
import math
from functools import lru_cache

# ---
# 1. PARAMETERS (Generated from preprocessed_product_parameters.csv)
# ---
# These parameters are for the product "Milk" and are
# used to set the simulation's constants.
params = {
  "product_simulated": "Milk",
  "mu_0_online_demand": 39.86094584,
  "mu_j_medium_store": 738.0,
  "p_price": 7.6,
  "c_p_product_cost": 5.32,
  "c_h_holding_cost": 1.0,
  "c_s_shipping_cost": 10.0,
  "c_l_penalty_cost": 10.0,
  "R_period_days": 7,
  "L_lead_time_days": 2,
  "store_types": {
    "small": {"mu_j": 2.0},  # Using paper's assumption
    "medium": {"mu_j": 738.0}, # From data
    "large": {"mu_j": 6.0}   # Using paper's assumption
  }
}

# ---
# 2. GLOBAL CONSTANTS
# ---
# Extracting parameters into global constants for the functions
P_PRICE = params['p_price']
C_HOLDING = params['c_h_holding_cost']
C_PRODUCT = params['c_p_product_cost']
C_SHIPPING = params['c_s_shipping_cost']
C_PENALTY = params['c_l_penalty_cost']

R_PERIOD_DAYS = params['R_period_days']
L_LEAD_TIME_DAYS = params['L_lead_time_days']

MU_0_ONLINE = params['mu_0_online_demand'] # Network-wide online demand

# Practical upper bound for demand summation in helper functions
MAX_DEMAND_STORE = int(params['store_types']['medium']['mu_j'] * 1.5)

# ---
# 3. STORE CLASS
# ---
class Store:
    """Represents a single retail store."""
    def __init__(self, id, type, mu_demand, initial_inventory):
        self.id = id
        self.type = type
        self.mu_demand = mu_demand  # mu_j, average in-store demand per day
        self.inventory = initial_inventory # I_j
        self.outstanding_order = 0 # Q_j
    
    def __repr__(self):
        return f"Store {self.id} ({self.type}): mu_j={self.mu_demand:.2f}, Inv={self.inventory}, Q={self.outstanding_order}"

# ---
# 4. HELPER FUNCTIONS
# ---

@lru_cache(maxsize=None)
def calculate_expected_contribution_next_day(inventory, mu_demand):
    """
    Calculates the simplified expected contribution (profit) from in-store sales
    for the *next day* (myopic view). Based on Eq. 5.1 and used in Eq. 11.
    """
    if inventory <= 0:
        return 0.0
        
    expected_sales = 0.0
    # Sum over all possible demands d_j
    # We use poisson.pmf for P_j(d_j)
    
    # Use a practical upper limit for summation
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
    """
    Calculates the threshold inventory w_j(t) to hold back.
    Based on Eq. 13.
    """
    days_to_cover = get_days_until_replenishment(t, L_LEAD_TIME_DAYS, R_PERIOD_DAYS)
    
    if days_to_cover <= 0:
        return 0
        
    mu_future_demand = mu_demand * days_to_cover
    holding_cost_future = C_HOLDING * days_to_cover
    
    if (holding_cost_future + P_PRICE) == 0:
        return 0 # Avoid division by zero
        
    target_prob = P_PRICE / (holding_cost_future + P_PRICE)
    
    # Inverse of the cumulative distribution
    threshold = poisson.ppf(target_prob, mu_future_demand)
    return int(threshold)

# ---
# 5. FULFILLMENT POLICIES
# ---

def mfp_fulfillment_policy(stores, F_online_orders):
    """
    Allocates online orders based on the Myopic Fulfilment Policy (MFP).
    (Algorithm 2)
    """
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
    """
    Allocates online orders based on the Threshold-based Myopic Policy (TMFP).
    (Section 4.2.2)
    """
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
            
            # TMFP constraint (Eq. 14)
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
    """
    *** PLACEHOLDER VALUE FUNCTION ***
    The real OSIP policy requires a pre-computed value function v_n,j(s_j)
    obtained from solving the decomposed MDP with value iteration (Algorithm 1).
    
    For demonstration, we use a simple proxy: the value of a state is
    the expected contribution for the *rest of the period*.
    """
    days_left_in_period = R_PERIOD_DAYS - t + 1
    val = calculate_expected_contribution_next_day(inventory, mu_demand)
    return val * days_left_in_period

def osip_fulfillment_policy(stores, F_online_orders, t, value_function):
    """
    Allocates online orders based on the One-Step Policy Improvement (OSIP).
    (Equation 10)
    """
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
            # Pad the allocation list with zeros if it's shorter than the number of stores
            padded_alloc = alloc_list + [0] * (len(stores) - len(alloc_list))
            final_allocations = {stores[i].id: padded_alloc[i] for i in range(len(stores))}

    return final_allocations

# ---
# 6. SIMULATION HARNESS
# ---

def run_simulation():
    """
    Runs a simple simulation for one R-day period using
    the parameters we generated from the dataset.
    """
    print("--- 🚚 Omni-Channel Simulation Start ---")
    print(f"Running simulation for: {params['product_simulated']}")
    print(f"  Online Demand (mu_0): {MU_0_ONLINE:.2f}")
    print(f"  Price (p): {P_PRICE}, Cost (c_p): {C_PRODUCT}")

    # --- Create stores based on the 'params' dictionary ---
    # For this demo, we'll create 1 of each store type.
    store_types = params['store_types']
    store_list = [
        Store(id=1, type="small", mu_demand=store_types['small']['mu_j'], initial_inventory=20),
        Store(id=2, type="medium", mu_demand=store_types['medium']['mu_j'], initial_inventory=3000), # Increased initial inventory for high demand
        Store(id=3, type="large", mu_demand=store_types['large']['mu_j'], initial_inventory=40)
    ]
    
    print("\nInitial State:")
    for s in store_list: print(s)

    # Simulate outstanding orders (Q_j)
    # Assume orders were placed at t=1, to arrive at t=3
    store_list[0].outstanding_order = 15
    store_list[1].outstanding_order = 1500 # Increased order for high demand
    store_list[2].outstanding_order = 35
    
    np.random.seed(42)
    
    # --- Loop through the 7-day period (R=7 sub-periods) ---
    for t in range(1, R_PERIOD_DAYS + 1):
        print(f"\n--- Day t={t} (Epoch 1) ---")
        
        if t == 1:
            print("Ordering decision made (using preset Q).")

        # In-store demand (d_j) realizes and is fulfilled
        print("In-store demand occurs:")
        for s in store_list:
            # Generate demand from this store's specific mu_j
            demand = poisson.rvs(s.mu_demand)
            sold = min(s.inventory, demand)
            s.inventory -= sold # This is I_j'
            print(f"  {s.id} ({s.type}): mu={s.mu_demand:.2f}, Demand={demand}, Sold={sold}, Inv={s.inventory}")

        # Online orders (F) are known, generated from network-wide mu_0
        F_online_orders = poisson.rvs(MU_0_ONLINE)
        print(f"Online orders received: F = {F_online_orders} (from mu_0={MU_0_ONLINE:.2f})")

        print(f"\n--- Day t={t} (Epoch 2) ---")
        print("Making online fulfillment decisions...")

        # Create copies of stores for each policy to make decisions on
        stores_mfp = [Store(s.id, s.type, s.mu_demand, s.inventory) for s in store_list]
        stores_tmfp = [Store(s.id, s.type, s.mu_demand, s.inventory) for s in store_list]
        stores_osip = [Store(s.id, s.type, s.mu_demand, s.inventory) for s in store_list]
        for i in range(len(store_list)):
            stores_osip[i].outstanding_order = store_list[i].outstanding_order

        # Run policies to get allocation decisions (f_j)
        alloc_mfp = mfp_fulfillment_policy(stores_mfp, F_online_orders)
        alloc_tmfp = tmfp_fulfillment_policy(stores_tmfp, F_online_orders, t)
        alloc_osip = osip_fulfillment_policy(stores_osip, F_online_orders, t, dummy_value_function_vj)
        
        print(f"  MFP Policy:  {alloc_mfp}")
        print(f"  TMFP Policy: {alloc_tmfp}")
        print(f"  OSIP Policy: {alloc_osip} (Using dummy v_j)")

        # Apply one policy (e.g., OSIP) to update state for next sub-period
        print("Applying OSIP policy for state transition...")
        for s in store_list:
            f_j = alloc_osip[s.id]
            s.inventory -= f_j # I_j' - f_j
            
            # Check for replenishment (Eq. 4)
            if t == L_LEAD_TIME_DAYS:
                print(f"  Store {s.id}: Replenishment {s.outstanding_order} arrived!")
                s.inventory += s.outstanding_order
                s.outstanding_order = 0 # Q_j'' = 0
        
        print("End of day state (I_j''):")
        for s in store_list: print(s)

    print("\n--- 🚚 Simulation End ---")

if __name__ == "__main__":
    run_simulation()