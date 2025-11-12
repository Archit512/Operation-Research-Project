#!/usr/bin/env python

import numpy as np
from scipy.stats import poisson
import math
from functools import lru_cache
import matplotlib.pyplot as plt

# ---
# PARAMETERS
# ---
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
    "small": {"mu_j": 2.0},
    "medium": {"mu_j": 738.0},
    "large": {"mu_j": 6.0}
  }
}

P_PRICE = params['p_price']
C_HOLDING = params['c_h_holding_cost']
C_PRODUCT = params['c_p_product_cost']
C_SHIPPING = params['c_s_shipping_cost']
C_PENALTY = params['c_l_penalty_cost']
R_PERIOD_DAYS = params['R_period_days']
L_LEAD_TIME_DAYS = params['L_lead_time_days']
MU_0_ONLINE = params['mu_0_online_demand']

# ---
# STORE CLASS
# ---
class Store:
    def __init__(self, id, type, mu_demand, initial_inventory):
        self.id = id
        self.type = type
        self.mu_demand = mu_demand
        self.inventory = initial_inventory
        self.outstanding_order = 0
        self.inventory_history = []
        self.sales_history = []
        self.allocations_osip = []

    def __repr__(self):
        return f"Store {self.id} ({self.type}): mu_j={self.mu_demand:.2f}, Inv={self.inventory}, Q={self.outstanding_order}"

# ---
# HELPER FUNCTIONS
# ---
@lru_cache(maxsize=None)
def calculate_expected_contribution_next_day(inventory, mu_demand):
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
    if t <= L:
        return L - t + 1
    else:
        return (R - t + 1) + L

@lru_cache(maxsize=None)
def calculate_threshold(t, mu_demand):
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
# POLICIES
# ---
def mfp_fulfillment_policy(stores, F_online_orders):
    current_inv = {s.id: s.inventory for s in stores}
    alloc = {s.id: 0 for s in stores}
    for _ in range(F_online_orders):
        best_store_id, best_gain = None, -float('inf')
        for s in stores:
            inv = current_inv[s.id]
            if inv > 0:
                ec_with = calculate_expected_contribution_next_day(inv - 1, s.mu_demand)
                ec_without = calculate_expected_contribution_next_day(inv, s.mu_demand)
                gain = (P_PRICE - C_SHIPPING) + C_PENALTY + (ec_with - ec_without)
                if gain > best_gain:
                    best_gain = gain
                    best_store_id = s.id
        if best_gain > 0:
            alloc[best_store_id] += 1
            current_inv[best_store_id] -= 1
        else:
            break
    return alloc

def tmfp_fulfillment_policy(stores, F_online_orders, t):
    current_inv = {s.id: s.inventory for s in stores}
    alloc = {s.id: 0 for s in stores}
    thresholds = {s.mu_demand: calculate_threshold(t, s.mu_demand) for s in stores}
    for _ in range(F_online_orders):
        best_store_id, best_gain = None, -float('inf')
        for s in stores:
            inv = current_inv[s.id]
            if inv > thresholds[s.mu_demand]:
                ec_with = calculate_expected_contribution_next_day(inv - 1, s.mu_demand)
                ec_without = calculate_expected_contribution_next_day(inv, s.mu_demand)
                gain = (P_PRICE - C_SHIPPING) + C_PENALTY + (ec_with - ec_without)
                if gain > best_gain:
                    best_gain = gain
                    best_store_id = s.id
        if best_gain > 0:
            alloc[best_store_id] += 1
            current_inv[best_store_id] -= 1
        else:
            break
    return alloc

@lru_cache(maxsize=None)
def dummy_value_function_vj(inventory, mu_demand, t, q):
    days_left = R_PERIOD_DAYS - t + 1
    val = calculate_expected_contribution_next_day(inventory, mu_demand)
    return val * days_left

def osip_fulfillment_policy(stores, F_online_orders, t, value_function):
    dp_cache = {}
    def solve_dp(store_idx, orders_left):
        if store_idx == len(stores) or orders_left == 0:
            return (0.0, [])
        state = (store_idx, orders_left)
        if state in dp_cache:
            return dp_cache[state]
        s = stores[store_idx]
        best_val, best_alloc = -float('inf'), []
        for f_j in range(min(orders_left, s.inventory) + 1):
            profit = (P_PRICE - C_SHIPPING) * f_j
            replenishment = s.outstanding_order if t == L_LEAD_TIME_DAYS else 0
            future_inv = s.inventory - f_j + replenishment
            future_val = value_function(future_inv, s.mu_demand, (t % R_PERIOD_DAYS) + 1, s.outstanding_order)
            rest_val, rest_alloc = solve_dp(store_idx + 1, orders_left - f_j)
            total_val = profit + future_val + rest_val
            if total_val > best_val:
                best_val, best_alloc = total_val, [f_j] + rest_alloc
        dp_cache[state] = (best_val, best_alloc)
        return dp_cache[state]

    best_profit, final_alloc = -float('inf'), {}
    for F_fulfill in range(F_online_orders + 1):
        penalty = C_PENALTY * (F_online_orders - F_fulfill)
        val, alloc = solve_dp(0, F_fulfill)
        profit = val - penalty
        if profit > best_profit:
            best_profit = profit
            padded = alloc + [0]*(len(stores)-len(alloc))
            final_alloc = {stores[i].id: padded[i] for i in range(len(stores))}
    return final_alloc

# ---
# MAIN SIMULATION
# ---
def run_simulation():
    print("\n--- 🚚 Omni-Channel Simulation Start ---")
    store_types = params['store_types']
    stores = [
        Store(1, "small", store_types['small']['mu_j'], 20),
        Store(2, "medium", store_types['medium']['mu_j'], 3000),
        Store(3, "large", store_types['large']['mu_j'], 40)
    ]
    stores[0].outstanding_order = 15
    stores[1].outstanding_order = 1500
    stores[2].outstanding_order = 35
    np.random.seed(42)

    # For plots
    daily_online, osip_fulfilled, mfp_fulfilled, tmfp_fulfilled, instore_sales = [], [], [], [], []

    for t in range(1, R_PERIOD_DAYS + 1):
        print(f"\n--- Day t={t} (Epoch 1) ---")
        if t == 1:
            print("Ordering decision made (using preset Q).")
        print("In-store demand occurs:")
        day_sales = 0
        for s in stores:
            demand = poisson.rvs(s.mu_demand)
            sold = min(s.inventory, demand)
            s.inventory -= sold
            s.sales_history.append(sold)
            s.inventory_history.append(s.inventory)
            day_sales += sold
            print(f"  {s.id} ({s.type}): mu={s.mu_demand:.2f}, Demand={demand}, Sold={sold}, Inv={s.inventory}")
        instore_sales.append(day_sales)

        F = poisson.rvs(MU_0_ONLINE)
        daily_online.append(F)
        print(f"Online orders received: F = {F} (from mu_0={MU_0_ONLINE:.2f})")

        print(f"\n--- Day t={t} (Epoch 2) ---")
        print("Making online fulfillment decisions...")

        stores_mfp = [Store(s.id, s.type, s.mu_demand, s.inventory) for s in stores]
        stores_tmfp = [Store(s.id, s.type, s.mu_demand, s.inventory) for s in stores]
        stores_osip = [Store(s.id, s.type, s.mu_demand, s.inventory) for s in stores]
        for i in range(len(stores)):
            stores_osip[i].outstanding_order = stores[i].outstanding_order

        alloc_mfp = mfp_fulfillment_policy(stores_mfp, F)
        alloc_tmfp = tmfp_fulfillment_policy(stores_tmfp, F, t)
        alloc_osip = osip_fulfillment_policy(stores_osip, F, t, dummy_value_function_vj)

        mfp_fulfilled.append(sum(alloc_mfp.values()))
        tmfp_fulfilled.append(sum(alloc_tmfp.values()))
        osip_fulfilled.append(sum(alloc_osip.values()))

        print(f"  MFP Policy:  {alloc_mfp}")
        print(f"  TMFP Policy: {alloc_tmfp}")
        print(f"  OSIP Policy: {alloc_osip} (Using dummy v_j)")

        print("Applying OSIP policy for state transition...")
        for s in stores:
            f_j = alloc_osip[s.id]
            s.inventory -= f_j
            s.allocations_osip.append(f_j)
            if t == L_LEAD_TIME_DAYS:
                print(f"  Store {s.id}: Replenishment {s.outstanding_order} arrived!")
                s.inventory += s.outstanding_order
                s.outstanding_order = 0

        print("End of day state (I_j''):")
        for s in stores:
            print(s)

    print("\n--- 🚚 Simulation End ---")

    # --- VISUALIZATION ---
    days = range(1, R_PERIOD_DAYS + 1)

    plt.figure(figsize=(8,5))
    plt.plot(days, daily_online, 'o-', label='Online Orders')
    plt.plot(days, osip_fulfilled, 's-', label='OSIP Fulfilled')
    plt.plot(days, mfp_fulfilled, '^-', label='MFP Fulfilled')
    plt.plot(days, tmfp_fulfilled, 'v-', label='TMFP Fulfilled')
    plt.title("Online Orders vs Fulfillment (All Policies)")
    plt.xlabel("Day"); plt.ylabel("Units")
    plt.legend(); plt.grid(True); plt.show()

    plt.figure(figsize=(7,4))
    plt.plot(days, instore_sales, 'o-', color='green')
    plt.title("In-store Sales per Day")
    plt.xlabel("Day"); plt.ylabel("Units Sold")
    plt.grid(True); plt.show()

    plt.figure(figsize=(8,5))
    for s in stores:
        plt.plot(days, s.inventory_history, label=f"{s.type.title()} Store {s.id}")
    plt.title("Inventory Levels by Store")
    plt.xlabel("Day"); plt.ylabel("Inventory")
    plt.legend(); plt.grid(True); plt.show()

    plt.figure(figsize=(8,5))
    for s in stores:
        plt.plot(days, s.allocations_osip, label=f"{s.type.title()} Store {s.id}")
    plt.title("Online Orders Fulfilled per Store (OSIP)")
    plt.xlabel("Day"); plt.ylabel("Units Allocated")
    plt.legend(); plt.grid(True); plt.show()


if __name__ == "__main__":
    run_simulation()
