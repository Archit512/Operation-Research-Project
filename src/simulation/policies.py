try:
    from .utils import (
        calculate_expected_contribution_next_day,
        calculate_threshold,
        dummy_value_function_vj
    )
except ImportError:
    from utils import (
        calculate_expected_contribution_next_day,
        calculate_threshold,
        dummy_value_function_vj
    )


def mfp_fulfillment_policy(stores, F_online_orders, p_price, c_shipping, c_penalty, c_holding, max_demand=30):
    
    current_inventories = {s.id: s.inventory for s in stores}
    allocations = {s.id: 0 for s in stores}
    
    for _ in range(F_online_orders):
        best_store_id = None
        best_marginal_gain = -float('inf')
        
        for s in stores:
            inv = current_inventories[s.id]
            
            if inv > 0:
                # Calculate marginal opportunity cost
                ec_with_fulfill = calculate_expected_contribution_next_day(
                    inv - 1, s.mu_demand, p_price, c_holding, max_demand
                )
                ec_without_fulfill = calculate_expected_contribution_next_day(
                    inv, s.mu_demand, p_price, c_holding, max_demand
                )
                
                marginal_oc = ec_with_fulfill - ec_without_fulfill
                gain = (p_price - c_shipping) + c_penalty + marginal_oc
                
                if gain > best_marginal_gain:
                    best_marginal_gain = gain
                    best_store_id = s.id
        
        if best_marginal_gain > 0:
            allocations[best_store_id] += 1
            current_inventories[best_store_id] -= 1
        else:
            break
            
    return allocations


def tmfp_fulfillment_policy(stores, F_online_orders, t, L, R, p_price, c_shipping, c_penalty, c_holding, max_demand=30):
    
    current_inventories = {s.id: s.inventory for s in stores}
    allocations = {s.id: 0 for s in stores}
    
    # Calculate thresholds for each store type
    thresholds = {}
    for s in stores:
        if s.mu_demand not in thresholds:
            thresholds[s.mu_demand] = calculate_threshold(t, s.mu_demand, L, R, p_price, c_holding)
    
    for _ in range(F_online_orders):
        best_store_id = None
        best_marginal_gain = -float('inf')
        
        for s in stores:
            inv = current_inventories[s.id]
            threshold = thresholds[s.mu_demand]
            
            # TMFP constraint: only fulfill if inventory > threshold
            if inv > threshold:
                ec_with_fulfill = calculate_expected_contribution_next_day(
                    inv - 1, s.mu_demand, p_price, c_holding, max_demand
                )
                ec_without_fulfill = calculate_expected_contribution_next_day(
                    inv, s.mu_demand, p_price, c_holding, max_demand
                )
                
                marginal_oc = ec_with_fulfill - ec_without_fulfill
                gain = (p_price - c_shipping) + c_penalty + marginal_oc
                
                if gain > best_marginal_gain:
                    best_marginal_gain = gain
                    best_store_id = s.id
        
        if best_marginal_gain > 0:
            allocations[best_store_id] += 1
            current_inventories[best_store_id] -= 1
        else:
            break
            
    return allocations


def osip_fulfillment_policy(stores, F_online_orders, t, L, R, p_price, c_shipping, c_penalty, 
                            c_holding, max_demand=30, value_function=None):
    """
    One-Step Policy Improvement (OSIP)
    
    Uses dynamic programming to find optimal allocation by considering
    both immediate profit and future value of inventory states.
    
    Solves: argmax_f { C(s', f) + ~v(s'') }
    
    Args:
        stores: List of Store objects
        F_online_orders (int): Number of online orders
        t (int): Current time period
        L (int): Lead time
        R (int): Review period
        p_price (float): Product price
        c_shipping (float): Shipping cost
        c_penalty (float): Penalty cost
        c_holding (float): Holding cost
        max_demand (int): Upper limit for demand
        value_function: Optional custom value function (uses dummy if None)
        
    Returns:
        dict: Allocation decisions {store_id: num_orders_allocated}
    """
    if value_function is None:
        # Use dummy value function with bound parameters
        def value_fn(inv, mu, tp, q):
            return dummy_value_function_vj(inv, mu, tp, q, R, p_price, c_holding, max_demand)
        value_function = value_fn
    
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
            immediate_profit_fj = (p_price - c_shipping) * f_j
            
            # Check for replenishment arrival
            replenishment = s.outstanding_order if t == L else 0
            future_inv = s.inventory - f_j + replenishment
            
            # Future value of this inventory state
            future_value_vj = value_function(
                future_inv, s.mu_demand, (t % R) + 1, s.outstanding_order
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
        penalty_cost = c_penalty * (F_online_orders - F_to_fulfill)
        dp_cache = {}
        total_value_from_dp, alloc_list = solve_dp(0, F_to_fulfill)
        total_profit = total_value_from_dp - penalty_cost
        
        if total_profit > best_total_profit:
            best_total_profit = total_profit
            padded_alloc = alloc_list + [0] * (len(stores) - len(alloc_list))
            final_allocations = {stores[i].id: padded_alloc[i] for i in range(len(stores))}
    
    return final_allocations
