"""
Policy Comparison Analysis
==========================
Compares the performance of MFP, TMFP, and OSIP policies across the 7-day period.
Calculates total profit for each policy.
"""

import numpy as np
from scipy.stats import poisson
from functools import lru_cache

# Import from corrected version
import sys
sys.path.insert(0, '.')

# Parameters (from a_corrected.py)
P_PRICE = 7.6
C_HOLDING = 1.0
C_PRODUCT = 5.32
C_SHIPPING = 10.0
C_PENALTY = 10.0
R_PERIOD_DAYS = 7
L_LEAD_TIME_DAYS = 2
MU_0_ONLINE = 39.86094584

# Store demand (corrected distribution)
STORE_DEMAND = {
    "small": 6.71,
    "medium": 26.84,
    "large": 40.25
}

# ---
# HELPER FUNCTIONS
# ---

class Store:
    """Store class for tracking"""
    def __init__(self, id, type, mu_demand, initial_inventory, outstanding_order):
        self.id = id
        self.type = type
        self.mu_demand = mu_demand
        self.inventory = initial_inventory
        self.outstanding_order = outstanding_order
    
    def copy(self):
        return Store(self.id, self.type, self.mu_demand, self.inventory, self.outstanding_order)

@lru_cache(maxsize=None)
def calculate_expected_contribution_next_day(inventory, mu_demand):
    if inventory <= 0:
        return 0.0
    expected_sales = 0.0
    upper_limit = max(50, int(mu_demand + 4 * np.sqrt(mu_demand)))
    for d in range(upper_limit + 1):
        prob = poisson.pmf(d, mu_demand)
        sales = min(inventory, d)
        expected_sales += prob * sales
    return P_PRICE * expected_sales - C_HOLDING * inventory

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

@lru_cache(maxsize=None)
def dummy_value_function_vj(inventory, mu_demand, t, q):
    days_left_in_period = R_PERIOD_DAYS - t + 1
    val = calculate_expected_contribution_next_day(inventory, mu_demand)
    return val * days_left_in_period

def mfp_fulfillment_policy(stores, F_online_orders):
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

def osip_fulfillment_policy(stores, F_online_orders, t, value_function):
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
            future_value_vj = value_function(future_inv, s.mu_demand, (t % R_PERIOD_DAYS) + 1, s.outstanding_order)
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
# SIMULATION WITH PROFIT TRACKING
# ---

def simulate_policy(policy_name, policy_func):
    """Run full 7-day simulation for one policy and calculate total profit"""
    
    # Initialize stores (same as a_corrected.py)
    stores = [
        Store(1, "small", STORE_DEMAND['small'], 30, int(STORE_DEMAND['small'] * 7)),
        Store(2, "medium", STORE_DEMAND['medium'], 120, int(STORE_DEMAND['medium'] * 7)),
        Store(3, "medium", STORE_DEMAND['medium'], 120, int(STORE_DEMAND['medium'] * 7)),
        Store(4, "large", STORE_DEMAND['large'], 180, int(STORE_DEMAND['large'] * 7))
    ]
    
    np.random.seed(42)  # Same seed for fair comparison
    
    total_profit = 0
    total_online_fulfilled = 0
    total_online_orders = 0
    total_instore_sales = 0
    total_instore_demand = 0
    lost_sales = 0
    
    metrics = {
        'daily_profits': [],
        'online_fulfillment_rate': [],
        'instore_fulfillment_rate': [],
        'total_holding_cost': 0
    }
    
    for t in range(1, R_PERIOD_DAYS + 1):
        daily_profit = 0
        
        # In-store demand
        for s in stores:
            demand = poisson.rvs(s.mu_demand)
            sold = min(s.inventory, demand)
            lost = demand - sold
            
            # Revenue from in-store sales
            instore_revenue = sold * P_PRICE
            daily_profit += instore_revenue
            
            # Holding cost
            holding_cost = s.inventory * C_HOLDING
            daily_profit -= holding_cost
            metrics['total_holding_cost'] += holding_cost
            
            s.inventory -= sold
            total_instore_sales += sold
            total_instore_demand += demand
            lost_sales += lost
        
        # Online orders
        F_online_orders = poisson.rvs(MU_0_ONLINE)
        total_online_orders += F_online_orders
        
        # Get allocation from policy
        if policy_name == "MFP":
            allocations = policy_func(stores, F_online_orders)
        elif policy_name == "TMFP":
            allocations = policy_func(stores, F_online_orders, t)
        else:  # OSIP
            allocations = policy_func(stores, F_online_orders, t, dummy_value_function_vj)
        
        # Apply allocations and calculate online profit
        total_fulfilled = sum(allocations.values())
        unfulfilled = F_online_orders - total_fulfilled
        
        # Online revenue (price - shipping cost)
        online_revenue = total_fulfilled * (P_PRICE - C_SHIPPING)
        daily_profit += online_revenue
        
        # Penalty for unfulfilled orders
        penalty = unfulfilled * C_PENALTY
        daily_profit -= penalty
        
        total_online_fulfilled += total_fulfilled
        
        # Update inventory based on fulfillment
        for s in stores:
            s.inventory -= allocations[s.id]
        
        # Replenishment
        if t == L_LEAD_TIME_DAYS:
            for s in stores:
                s.inventory += s.outstanding_order
                s.outstanding_order = 0
        
        total_profit += daily_profit
        metrics['daily_profits'].append(daily_profit)
        metrics['online_fulfillment_rate'].append(total_fulfilled / F_online_orders if F_online_orders > 0 else 1.0)
        metrics['instore_fulfillment_rate'].append(sold / demand if demand > 0 else 1.0)
    
    metrics['total_profit'] = total_profit
    metrics['online_fulfillment_rate_avg'] = total_online_fulfilled / total_online_orders if total_online_orders > 0 else 0
    metrics['instore_fulfillment_rate_avg'] = total_instore_sales / total_instore_demand if total_instore_demand > 0 else 0
    metrics['total_online_fulfilled'] = total_online_fulfilled
    metrics['total_online_orders'] = total_online_orders
    metrics['total_instore_sales'] = total_instore_sales
    metrics['lost_sales'] = lost_sales
    
    return metrics

# ---
# MAIN COMPARISON
# ---

def main():
    print("\n" + "="*80)
    print("POLICY PERFORMANCE COMPARISON")
    print("="*80)
    print("Comparing MFP, TMFP, and OSIP over 7-day simulation period")
    print("Product: Milk | Network: 4 stores (1 small, 2 medium, 1 large)")
    print("="*80 + "\n")
    
    # Run simulations
    results = {}
    results['MFP'] = simulate_policy('MFP', mfp_fulfillment_policy)
    results['TMFP'] = simulate_policy('TMFP', tmfp_fulfillment_policy)
    results['OSIP'] = simulate_policy('OSIP', osip_fulfillment_policy)
    
    # Display results
    print(f"{'Metric':<40} {'MFP':>12} {'TMFP':>12} {'OSIP':>12}")
    print("-"*80)
    
    # Total profit (main metric)
    print(f"{'TOTAL PROFIT ($)':<40} {results['MFP']['total_profit']:>12.2f} {results['TMFP']['total_profit']:>12.2f} {results['OSIP']['total_profit']:>12.2f}")
    print("-"*80)
    
    # Online fulfillment
    print(f"{'Online orders received':<40} {results['MFP']['total_online_orders']:>12} {results['TMFP']['total_online_orders']:>12} {results['OSIP']['total_online_orders']:>12}")
    print(f"{'Online orders fulfilled':<40} {results['MFP']['total_online_fulfilled']:>12} {results['TMFP']['total_online_fulfilled']:>12} {results['OSIP']['total_online_fulfilled']:>12}")
    print(f"{'Online fulfillment rate (%)':<40} {results['MFP']['online_fulfillment_rate_avg']*100:>12.1f} {results['TMFP']['online_fulfillment_rate_avg']*100:>12.1f} {results['OSIP']['online_fulfillment_rate_avg']*100:>12.1f}")
    print("-"*80)
    
    # In-store performance
    print(f"{'In-store sales (units)':<40} {results['MFP']['total_instore_sales']:>12} {results['TMFP']['total_instore_sales']:>12} {results['OSIP']['total_instore_sales']:>12}")
    print(f"{'In-store lost sales (units)':<40} {results['MFP']['lost_sales']:>12} {results['TMFP']['lost_sales']:>12} {results['OSIP']['lost_sales']:>12}")
    print(f"{'In-store fulfillment rate (%)':<40} {results['MFP']['instore_fulfillment_rate_avg']*100:>12.1f} {results['TMFP']['instore_fulfillment_rate_avg']*100:>12.1f} {results['OSIP']['instore_fulfillment_rate_avg']*100:>12.1f}")
    print("-"*80)
    
    # Costs
    print(f"{'Total holding cost ($)':<40} {results['MFP']['total_holding_cost']:>12.2f} {results['TMFP']['total_holding_cost']:>12.2f} {results['OSIP']['total_holding_cost']:>12.2f}")
    
    print("\n" + "="*80)
    
    # Determine winner
    best_policy = max(results.keys(), key=lambda k: results[k]['total_profit'])
    best_profit = results[best_policy]['total_profit']
    
    print(f"🏆 WINNER: {best_policy} with total profit of ${best_profit:.2f}")
    print("="*80)
    
    # Show profit differences
    print("\nProfit Comparison to Best Policy:")
    for policy in ['MFP', 'TMFP', 'OSIP']:
        diff = results[policy]['total_profit'] - best_profit
        pct = (diff / best_profit * 100) if best_profit != 0 else 0
        status = "👑 BEST" if policy == best_policy else f"{diff:+.2f} ({pct:+.1f}%)"
        print(f"  {policy}: {status}")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
