#!/usr/bin/env python

import numpy as np
from scipy.stats import poisson
import math
from functools import lru_cache
import matplotlib.pyplot as plt

# --- 1. PARAMETERS ---
params = {
  "product_simulated": "Milk",
  "mu_0_online_demand": 39.86094584,
  "total_instore_demand": 738.0,
  "p_price": 7.6,
  "c_p_product_cost": 5.32,
  "c_h_holding_cost": 1.0,
  "c_s_shipping_cost": 10.0,
  "c_l_penalty_cost": 10.0,
  "R_period_days": 7,
  "L_lead_time_days": 2,
  "num_stores_small": 10,
  "num_stores_medium": 10,
  "num_stores_large": 10,
  "demand_ratio": {"small": 1.0, "medium": 4.0, "large": 6.0}
}

def calculate_per_store_demand():
    total = params['total_instore_demand']
    sc, mc, lc = params['num_stores_small'], params['num_stores_medium'], params['num_stores_large']
    sr, mr, lr = params['demand_ratio'].values()
    total_units = sc*sr + mc*mr + lc*lr
    per_unit = total / total_units
    return {"small": per_unit*sr, "medium": per_unit*mr, "large": per_unit*lr}

per_store_demand = calculate_per_store_demand()
params['store_types'] = {
    "small": {"mu_j": per_store_demand['small']},
    "medium": {"mu_j": per_store_demand['medium']},
    "large": {"mu_j": per_store_demand['large']}
}

# --- 2. CONSTANTS ---
P_PRICE = params['p_price']; C_HOLDING = params['c_h_holding_cost']
C_PRODUCT = params['c_p_product_cost']; C_SHIPPING = params['c_s_shipping_cost']
C_PENALTY = params['c_l_penalty_cost']; R_PERIOD_DAYS = params['R_period_days']
L_LEAD_TIME_DAYS = params['L_lead_time_days']; MU_0_ONLINE = params['mu_0_online_demand']

# --- 3. CLASS ---
class Store:
    def __init__(self, id, type, mu_demand, inv):
        self.id, self.type, self.mu_demand = id, type, mu_demand
        self.inventory, self.outstanding_order = inv, 0
        self.inventory_history, self.sales_history = [], []
    def __repr__(self):
        return f"Store {self.id} ({self.type}): mu_j={self.mu_demand:.2f}, Inv={self.inventory}, Q={self.outstanding_order}"

# --- 4. HELPERS ---
@lru_cache(None)
def calculate_expected_contribution_next_day(inv, mu):
    if inv <= 0: return 0
    exp_sales = sum(poisson.pmf(d, mu)*min(inv,d) for d in range(int(mu+4*np.sqrt(mu))+1))
    return P_PRICE*exp_sales - C_HOLDING*inv

@lru_cache(None)
def get_days_until_replenishment(t,L,R):
    return L-t+1 if t<=L else (R-t+1)+L

@lru_cache(None)
def calculate_threshold(t,mu):
    days = get_days_until_replenishment(t,L_LEAD_TIME_DAYS,R_PERIOD_DAYS)
    if days<=0: return 0
    mu_future = mu*days; cost_future = C_HOLDING*days
    if (cost_future+P_PRICE)==0: return 0
    target = P_PRICE/(cost_future+P_PRICE)
    return int(poisson.ppf(target,mu_future))

# --- 5. POLICIES ---
def mfp_fulfillment_policy(stores,F):
    invs={s.id:s.inventory for s in stores}; alloc={s.id:0 for s in stores}
    for _ in range(F):
        best=-float('inf'); best_id=None
        for s in stores:
            inv=invs[s.id]
            if inv>0:
                gain=(P_PRICE-C_SHIPPING)+C_PENALTY+(calculate_expected_contribution_next_day(inv-1,s.mu_demand)-calculate_expected_contribution_next_day(inv,s.mu_demand))
                if gain>best: best, best_id=gain, s.id
        if best>0: alloc[best_id]+=1; invs[best_id]-=1
        else: break
    return alloc

def tmfp_fulfillment_policy(stores,F,t):
    invs={s.id:s.inventory for s in stores}; alloc={s.id:0 for s in stores}
    th={s.mu_demand:calculate_threshold(t,s.mu_demand) for s in stores}
    for _ in range(F):
        best=-float('inf'); best_id=None
        for s in stores:
            inv=invs[s.id]
            if inv>th[s.mu_demand]:
                gain=(P_PRICE-C_SHIPPING)+C_PENALTY+(calculate_expected_contribution_next_day(inv-1,s.mu_demand)-calculate_expected_contribution_next_day(inv,s.mu_demand))
                if gain>best: best, best_id=gain, s.id
        if best>0: alloc[best_id]+=1; invs[best_id]-=1
        else: break
    return alloc

@lru_cache(None)
def dummy_value_function_vj(inv,mu,t,q):
    days=R_PERIOD_DAYS-t+1
    return calculate_expected_contribution_next_day(inv,mu)*days

def osip_fulfillment_policy(stores,F,t,vf):
    dp_cache={}
    def solve_dp(i,left):
        if i==len(stores) or left==0: return 0.0,[]
        if (i,left) in dp_cache: return dp_cache[(i,left)]
        s=stores[i]; best=-float('inf'); best_alloc=[]
        for f in range(min(left,s.inventory)+1):
            val=(P_PRICE-C_SHIPPING)*f+vf(s.inventory-f+(s.outstanding_order if t==L_LEAD_TIME_DAYS else 0),s.mu_demand,(t%R_PERIOD_DAYS)+1,s.outstanding_order)
            next_val, next_alloc=solve_dp(i+1,left-f)
            if val+next_val>best: best,val+next_val; best_alloc=[f]+next_alloc
        dp_cache[(i,left)]=(best,best_alloc); return best,best_alloc
    best_total=-float('inf'); final={}
    for Ff in range(F+1):
        penalty=C_PENALTY*(F-Ff)
        val,alloc=solve_dp(0,Ff)
        if val-penalty>best_total:
            best_total=val-penalty
            padded=alloc+[0]*(len(stores)-len(alloc))
            final={stores[i].id:padded[i] for i in range(len(stores))}
    return final

# --- 6. SIMULATION ---
def run_simulation():
    print("--- 🚚 Omni-Channel Simulation Start (CORRECTED+VISUAL) ---")
    st=params['store_types']
    stores=[Store(1,"small",st['small']['mu_j'],30),
            Store(2,"medium",st['medium']['mu_j'],120),
            Store(3,"medium",st['medium']['mu_j'],120),
            Store(4,"large",st['large']['mu_j'],180)]
    for s in stores:
        s.outstanding_order=int(s.mu_demand*7)
    np.random.seed(42)
    daily_online,osip_f,mfp_f,tmfp_f=[],[],[],[]

    for t in range(1,R_PERIOD_DAYS+1):
        print(f"\n--- Day {t} ---")
        day_sales=0
        for s in stores:
            d=poisson.rvs(s.mu_demand); sold=min(s.inventory,d)
            s.inventory-=sold; s.sales_history.append(sold); s.inventory_history.append(s.inventory)
            day_sales+=sold; print(f"  {s}")
        F=poisson.rvs(MU_0_ONLINE); daily_online.append(F)
        alloc_m=mfp_fulfillment_policy(stores,F)
        alloc_t=tmfp_fulfillment_policy(stores,F,t)
        alloc_o=osip_fulfillment_policy(stores,F,t,dummy_value_function_vj)
        osip_f.append(sum(alloc_o.values())); mfp_f.append(sum(alloc_m.values())); tmfp_f.append(sum(alloc_t.values()))
        print("  OSIP allocations:",alloc_o)
        for s in stores:
            s.inventory-=alloc_o[s.id]
            if t==L_LEAD_TIME_DAYS:
                s.inventory+=s.outstanding_order; s.outstanding_order=0

    # --- VISUALS ---
    days=range(1,R_PERIOD_DAYS+1)
    plt.figure(figsize=(8,5))
    plt.plot(days,daily_online,'o-',label='Online Orders')
    plt.plot(days,osip_f,'s-',label='OSIP Fulfilled')
    plt.plot(days,mfp_f,'^-',label='MFP Fulfilled')
    plt.plot(days,tmfp_f,'v-',label='TMFP Fulfilled')
    plt.title("Online Orders vs Fulfillment (Policies)")
    plt.xlabel("Day"); plt.ylabel("Units"); plt.legend(); plt.grid(True); plt.show()

    plt.figure(figsize=(8,5))
    for s in stores:
        plt.plot(days,s.inventory_history,label=f"{s.type.title()} {s.id}")
    plt.title("Inventory Level per Store (7 days)")
    plt.xlabel("Day"); plt.ylabel("Inventory"); plt.legend(); plt.grid(True); plt.show()

if __name__=="__main__":
    run_simulation()
