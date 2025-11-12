"""
Real Data Simulation (Base Test Case)
======================================
Simulates omni-channel fulfillment using theoretical base test case parameters.
Demonstrates MFP, TMFP, and OSIP policies in action.

Based on Section 5.1 of the research paper.
"""

import numpy as np
from scipy.stats import poisson

from .models import Store
from .policies import mfp_fulfillment_policy, tmfp_fulfillment_policy, osip_fulfillment_policy
from .utils import calculate_threshold


# --- Base Test Case Parameters (from Section 5.1) ---

# Economic parameters
P_PRICE = 100.0     # p, price of the product
C_HOLDING = 1.0     # c_h, holding cost per day
C_PRODUCT = 70.0    # c_p, cost of the product
C_SHIPPING = 10.0   # c_s, shipping cost of an online order
C_PENALTY = 10.0    # c_l, penalty of a lost sale (unfulfilled online order)

# Time parameters
R_PERIOD_DAYS = 7   # R, review period (sub-periods)
L_LEAD_TIME_DAYS = 2 # L, fixed lead time

# Demand parameters
MAX_DEMAND_STORE = 30 # Practical upper bound for D_j (Poisson truncation)



# --- Simulation Harness ---

def run_simulation():
    """
    1. Daily in-store demand (Poisson distributed)
    2. Online order arrivals
    3. Fulfillment decisions using MFP, TMFP, and OSIP
    4. Inventory replenishment
    """
    print("--- Omni-Channel Simulation Start ---")
    print(f"Base Parameters: p={P_PRICE}, c_h={C_HOLDING}, c_s={C_SHIPPING}, c_l={C_PENALTY}")
    print(f"Time: R={R_PERIOD_DAYS} days, L={L_LEAD_TIME_DAYS} days")
    
    # Create stores based on base test case
    # 1 small, 1 medium, 1 large store for demonstration
    store_list = [
        Store(id=1, type="small", mu_demand=2.0, initial_inventory=20),
        Store(id=2, type="medium", mu_demand=4.0, initial_inventory=30),
        Store(id=3, type="large", mu_demand=6.0, initial_inventory=40)
    ]
    print("\nInitial State:")
    for s in store_list: 
        print(s)

    # Set outstanding orders (arrive at t=L_LEAD_TIME_DAYS)
    store_list[0].outstanding_order = 15
    store_list[1].outstanding_order = 25
    store_list[2].outstanding_order = 35
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # --- Simulate the 7-day period ---
    for t in range(1, R_PERIOD_DAYS + 1):
        print(f"\n--- Day t={t} (Epoch 1: In-store Demand) ---")
        
        # Display ordering decision on day 1
        if t == 1:
            print("Ordering decision made (using preset Q).")
            for s in store_list:
                print(f"  Store {s.id}: Q = {s.outstanding_order}")

        # In-store demand realizes and is fulfilled
        print("In-store demand occurs:")
        for s in store_list:
            demand = poisson.rvs(s.mu_demand)
            sold = min(s.inventory, demand)
            s.inventory -= sold  # Update to I_j'
            print(f"  Store {s.id}: Demand={demand}, Sold={sold}, Inv={s.inventory}")

        # Online orders arrive
        F_online_orders = poisson.rvs(10)  # Demo: assume mu_0 = 10
        print(f"Online orders received: F = {F_online_orders}")

        print(f"\n--- Day t={t} (Epoch 2: Online Fulfillment) ---")
        print("Making online fulfillment decisions...")

        # Create store copies for each policy
        stores_mfp = [s.copy() for s in store_list]
        stores_tmfp = [s.copy() for s in store_list]
        stores_osip = [s.copy() for s in store_list]

        # Run all three policies
        alloc_mfp = mfp_fulfillment_policy(
            stores_mfp, F_online_orders, P_PRICE, C_SHIPPING, C_PENALTY, C_HOLDING, MAX_DEMAND_STORE
        )
        alloc_tmfp = tmfp_fulfillment_policy(
            stores_tmfp, F_online_orders, t, L_LEAD_TIME_DAYS, R_PERIOD_DAYS,
            P_PRICE, C_SHIPPING, C_PENALTY, C_HOLDING, MAX_DEMAND_STORE
        )
        alloc_osip = osip_fulfillment_policy(
            stores_osip, F_online_orders, t, L_LEAD_TIME_DAYS, R_PERIOD_DAYS,
            P_PRICE, C_SHIPPING, C_PENALTY, C_HOLDING, MAX_DEMAND_STORE
        )
        
        # Display policy decisions
        print(f"  MFP Policy:  {alloc_mfp}")
        print(f"  TMFP Policy: {alloc_tmfp} (Thresholds: {[calculate_threshold(t, s.mu_demand, L_LEAD_TIME_DAYS, R_PERIOD_DAYS, P_PRICE, C_HOLDING) for s in store_list]})")
        print(f"  OSIP Policy: {alloc_osip}")

        # Apply OSIP policy to update state for next period
        print("Applying OSIP policy for state transition...")
        for s in store_list:
            f_j = alloc_osip[s.id]
            s.inventory -= f_j  # Update to I_j''
            
            # Check for replenishment arrival
            if t == L_LEAD_TIME_DAYS:
                print(f"  Store {s.id}: Replenishment of {s.outstanding_order} units arrived!")
                s.inventory += s.outstanding_order
                s.outstanding_order = 0
        
        print("End of day state (I_j''):")
        for s in store_list: 
            print(s)

    print("\n--- 🚚 Simulation End ---")

if __name__ == "__main__":
    run_simulation()