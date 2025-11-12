"""
OSIP Decision Maker
===================
This script applies the One-Step Policy Improvement (OSIP) algorithm
to make fulfillment decisions based on user-provided inputs.

Usage:
    python osip_decision_maker.py
"""

import numpy as np
from scipy.stats import poisson
from functools import lru_cache
import json

# ---
# CONFIGURATION
# ---
class Config:
    """Configuration parameters for the OSIP algorithm"""
    def __init__(self):
        # Economic parameters (can be customized)
        self.p_price = 7.6
        self.c_holding = 1.0
        self.c_product = 5.32
        self.c_shipping = 10.0
        self.c_penalty = 10.0
        
        # Time parameters
        self.R_period_days = 7
        self.L_lead_time_days = 2

# Global config instance
config = Config()

# ---
# STORE CLASS
# ---
class Store:
    """Represents a single retail store"""
    def __init__(self, store_id, store_type, mu_demand, inventory, outstanding_order=0):
        self.id = store_id
        self.type = store_type
        self.mu_demand = mu_demand
        self.inventory = inventory
        self.outstanding_order = outstanding_order
    
    def __repr__(self):
        return f"Store {self.id} ({self.type}): μ={self.mu_demand}, Inv={self.inventory}, Q={self.outstanding_order}"
    
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'mu_demand': self.mu_demand,
            'inventory': self.inventory,
            'outstanding_order': self.outstanding_order
        }

# ---
# HELPER FUNCTIONS
# ---

@lru_cache(maxsize=None)
def calculate_expected_contribution_next_day(inventory, mu_demand):
    """
    Calculates expected contribution (profit) from in-store sales for next day.
    EC_j = p * (Expected_Sales) - c_h * I_j
    """
    if inventory <= 0:
        return 0.0
    
    expected_sales = 0.0
    upper_limit = max(50, int(mu_demand + 4 * np.sqrt(mu_demand)))
    
    for d in range(upper_limit + 1):
        prob = poisson.pmf(d, mu_demand)
        sales = min(inventory, d)
        expected_sales += prob * sales
    
    revenue = config.p_price * expected_sales
    holding_cost = config.c_holding * inventory
    
    return revenue - holding_cost

@lru_cache(maxsize=None)
def dummy_value_function_vj(inventory, mu_demand, t, q):
    """
    Simplified value function for OSIP.
    In production, this would be replaced with pre-computed value iteration results.
    """
    days_left_in_period = config.R_period_days - t + 1
    val = calculate_expected_contribution_next_day(inventory, mu_demand)
    return val * max(1, days_left_in_period)

# ---
# OSIP FULFILLMENT POLICY
# ---

def osip_fulfillment_policy(stores, F_online_orders, current_day, value_function=None):
    """
    One-Step Policy Improvement (OSIP) Algorithm
    
    Args:
        stores: List of Store objects
        F_online_orders: Number of online orders to fulfill
        current_day: Current day in the review period (1-7)
        value_function: Optional custom value function (uses dummy if None)
    
    Returns:
        dict: Allocation decisions {store_id: num_orders_allocated}
    """
    if value_function is None:
        value_function = dummy_value_function_vj
    
    dp_cache = {}
    
    def solve_dp(store_idx, orders_left):
        """
        Dynamic programming solver for optimal allocation.
        Returns (max_value, allocation_list)
        """
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
            # Immediate profit from fulfilling f_j orders
            immediate_profit_fj = (config.p_price - config.c_shipping) * f_j
            
            # Check for replenishment arrival
            replenishment = s.outstanding_order if current_day == config.L_lead_time_days else 0
            future_inv = s.inventory - f_j + replenishment
            
            # Future value of this inventory state
            future_value_vj = value_function(
                future_inv, 
                s.mu_demand,
                (current_day % config.R_period_days) + 1,
                s.outstanding_order
            )
            
            # Get value from remaining stores
            value_from_rest, alloc_from_rest = solve_dp(store_idx + 1, orders_left - f_j)
            
            current_total_value = immediate_profit_fj + future_value_vj + value_from_rest
            
            if current_total_value > best_value:
                best_value = current_total_value
                best_allocation = [f_j] + alloc_from_rest
        
        dp_cache[state] = (best_value, best_allocation)
        return best_value, best_allocation
    
    # Find optimal number of orders to fulfill
    best_total_profit = -float('inf')
    final_allocations = {}
    
    for F_to_fulfill in range(F_online_orders + 1):
        penalty_cost = config.c_penalty * (F_online_orders - F_to_fulfill)
        dp_cache = {}
        total_value_from_dp, alloc_list = solve_dp(0, F_to_fulfill)
        total_profit = total_value_from_dp - penalty_cost
        
        if total_profit > best_total_profit:
            best_total_profit = total_profit
            padded_alloc = alloc_list + [0] * (len(stores) - len(alloc_list))
            final_allocations = {stores[i].id: padded_alloc[i] for i in range(len(stores))}
    
    return final_allocations, best_total_profit

# ---
# INPUT/OUTPUT FUNCTIONS
# ---

def get_user_input():
    """Collect input from the user"""
    print("\n" + "="*60)
    print("OSIP Decision Maker - Fulfillment Optimization")
    print("="*60)
    
    # Get current day
    while True:
        try:
            current_day = int(input("\nEnter current day in review period (1-7): "))
            if 1 <= current_day <= 7:
                break
            print("Please enter a value between 1 and 7")
        except ValueError:
            print("Please enter a valid number")
    
    # Get number of online orders
    while True:
        try:
            F_online_orders = int(input("Enter number of online orders to fulfill: "))
            if F_online_orders >= 0:
                break
            print("Please enter a non-negative number")
        except ValueError:
            print("Please enter a valid number")
    
    # Get number of stores
    while True:
        try:
            num_stores = int(input("Enter number of stores: "))
            if num_stores > 0:
                break
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Get store details
    stores = []
    print(f"\nEnter details for {num_stores} store(s):")
    for i in range(num_stores):
        print(f"\n--- Store {i+1} ---")
        store_type = input(f"  Store type (small/medium/large): ").strip()
        
        while True:
            try:
                mu_demand = float(input(f"  Average daily demand (μ_j): "))
                if mu_demand >= 0:
                    break
                print("  Please enter a non-negative number")
            except ValueError:
                print("  Please enter a valid number")
        
        while True:
            try:
                inventory = int(input(f"  Current inventory: "))
                if inventory >= 0:
                    break
                print("  Please enter a non-negative number")
            except ValueError:
                print("  Please enter a valid number")
        
        while True:
            try:
                outstanding = int(input(f"  Outstanding order (arriving day 3): "))
                if outstanding >= 0:
                    break
                print("  Please enter a non-negative number")
            except ValueError:
                print("  Please enter a valid number")
        
        store = Store(i+1, store_type, mu_demand, inventory, outstanding)
        stores.append(store)
    
    return stores, F_online_orders, current_day

def display_results(stores, allocations, profit, F_online_orders):
    """Display the OSIP decision results"""
    print("\n" + "="*60)
    print("OSIP DECISION RESULTS")
    print("="*60)
    
    print("\nStore States:")
    print("-" * 60)
    for store in stores:
        print(store)
    
    print(f"\nOnline Orders to Fulfill: {F_online_orders}")
    print("\nOptimal Allocation:")
    print("-" * 60)
    
    total_allocated = 0
    for store in stores:
        alloc = allocations[store.id]
        total_allocated += alloc
        percentage = (alloc / F_online_orders * 100) if F_online_orders > 0 else 0
        print(f"  Store {store.id} ({store.type}): {alloc} orders ({percentage:.1f}%)")
    
    unfulfilled = F_online_orders - total_allocated
    print(f"\n  Total Fulfilled: {total_allocated}")
    print(f"  Unfulfilled: {unfulfilled}")
    
    if unfulfilled > 0:
        print(f"  Penalty Cost: ${unfulfilled * config.c_penalty:.2f}")
    
    print(f"\nExpected Total Profit: ${profit:.2f}")
    
    print("\n" + "="*60)

def save_results_to_file(stores, allocations, profit, F_online_orders, current_day):
    """Save results to JSON file"""
    results = {
        'current_day': current_day,
        'online_orders': F_online_orders,
        'stores': [s.to_dict() for s in stores],
        'allocations': allocations,
        'total_allocated': sum(allocations.values()),
        'unfulfilled': F_online_orders - sum(allocations.values()),
        'expected_profit': profit,
        'config': {
            'price': config.p_price,
            'shipping_cost': config.c_shipping,
            'penalty_cost': config.c_penalty,
            'holding_cost': config.c_holding
        }
    }
    
    filename = f"osip_decision_day{current_day}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {filename}")

# ---
# EXAMPLE PRESET SCENARIOS
# ---

def run_example_scenario(scenario_name="milk"):
    """Run a preset example scenario"""
    print("\n" + "="*60)
    print(f"Running Example Scenario: {scenario_name.upper()}")
    print("="*60)
    
    if scenario_name == "milk":
        # Real-world Milk scenario
        stores = [
            Store(1, "small", 2.0, 16, 15),
            Store(2, "medium", 738.0, 2293, 1500),
            Store(3, "large", 6.0, 34, 35)
        ]
        F_online_orders = 33
        current_day = 1
    
    elif scenario_name == "generic":
        # Generic test case
        stores = [
            Store(1, "small", 2.0, 20, 15),
            Store(2, "medium", 4.0, 30, 25),
            Store(3, "large", 6.0, 40, 35)
        ]
        config.p_price = 100.0
        config.c_product = 70.0
        F_online_orders = 10
        current_day = 1
    
    else:
        print(f"Unknown scenario: {scenario_name}")
        return
    
    print(f"\nDay: {current_day}")
    print(f"Online Orders: {F_online_orders}")
    print("\nStores:")
    for store in stores:
        print(f"  {store}")
    
    print("\nRunning OSIP algorithm...")
    allocations, profit = osip_fulfillment_policy(stores, F_online_orders, current_day)
    
    display_results(stores, allocations, profit, F_online_orders)
    
    save = input("\nSave results to file? (y/n): ").strip().lower()
    if save == 'y':
        save_results_to_file(stores, allocations, profit, F_online_orders, current_day)

# ---
# MAIN
# ---

def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("OSIP DECISION MAKER")
    print("One-Step Policy Improvement for Fulfillment Optimization")
    print("="*60)
    
    print("\nOptions:")
    print("1. Enter custom input")
    print("2. Run example scenario (Milk - real data)")
    print("3. Run example scenario (Generic test case)")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        stores, F_online_orders, current_day = get_user_input()
        print("\nRunning OSIP algorithm...")
        allocations, profit = osip_fulfillment_policy(stores, F_online_orders, current_day)
        display_results(stores, allocations, profit, F_online_orders)
        
        save = input("\nSave results to file? (y/n): ").strip().lower()
        if save == 'y':
            save_results_to_file(stores, allocations, profit, F_online_orders, current_day)
    
    elif choice == "2":
        run_example_scenario("milk")
    
    elif choice == "3":
        run_example_scenario("generic")
    
    elif choice == "4":
        print("\nExiting...")
        return
    
    else:
        print("\nInvalid option. Exiting...")
        return
    
    print("\n" + "="*60)
    print("Thank you for using OSIP Decision Maker!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
