"""
Policy Comparison Analysis (with Visualization)
================================================
Compares MFP, TMFP, and OSIP policies over 7 days.
Generates profit, fulfillment, and service-level plots.
"""

import numpy as np
from scipy.stats import poisson
from functools import lru_cache
import matplotlib.pyplot as plt

# --- PARAMETERS ---
P_PRICE = 7.6
C_HOLDING = 1.0
C_PRODUCT = 5.32
C_SHIPPING = 10.0
C_PENALTY = 10.0
R_PERIOD_DAYS = 7
L_LEAD_TIME_DAYS = 2
MU_0_ONLINE = 39.86094584

STORE_DEMAND = {"small": 6.71, "medium": 26.84, "large": 40.25}

# --- STORE CLASS ---
class Store:
    def __init__(self, id, type, mu_demand, inv, order):
        self.id, self.type, self.mu_demand = id, type, mu_demand
        self.inventory, self.outstanding_order = inv, order
    def copy(self):
        return Store(self.id, self.type, self.mu_demand, self.inventory, self.outstanding_order)

# --- HELPERS ---
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
    days=get_days_until_replenishment(t,L_LEAD_TIME_DAYS,R_PERIOD_DAYS)
    if days<=0: return 0
    mu_future=mu*days; cost_future=C_HOLDING*days
    if (cost_future+P_PRICE)==0: return 0
    target=P_PRICE/(cost_future+P_PRICE)
    return int(poisson.ppf(target,mu_future))

@lru_cache(None)
def dummy_value_function_vj(inv,mu,t,q):
    return calculate_expected_contribution_next_day(inv,mu)*(R_PERIOD_DAYS-t+1)

# --- POLICIES ---
def mfp_fulfillment_policy(stores,F):
    invs={s.id:s.inventory for s in stores}; alloc={s.id:0 for s in stores}
    for _ in range(F):
        best=-float('inf'); best_id=None
        for s in stores:
            inv=invs[s.id]
            if inv>0:
                gain=(P_PRICE-C_SHIPPING)+C_PENALTY+(calculate_expected_contribution_next_day(inv-1,s.mu_demand)-calculate_expected_contribution_next_day(inv,s.mu_demand))
                if gain>best: best,best_id=gain,s.id
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
                if gain>best: best,best_id=gain,s.id
        if best>0: alloc[best_id]+=1; invs[best_id]-=1
        else: break
    return alloc

def osip_fulfillment_policy(stores,F,t,vf):
    dp_cache={}
    def solve_dp(i,left):
        if i==len(stores) or left==0: return 0.0,[]
        if (i,left) in dp_cache: return dp_cache[(i,left)]
        s=stores[i]; best=-float('inf'); best_alloc=[]
        for f in range(min(left,s.inventory)+1):
            val=(P_PRICE-C_SHIPPING)*f + vf(s.inventory-f+(s.outstanding_order if t==L_LEAD_TIME_DAYS else 0), s.mu_demand, (t%R_PERIOD_DAYS)+1, s.outstanding_order)
            next_val,next_alloc=solve_dp(i+1,left-f)
            if val+next_val>best: best,val+next_val; best_alloc=[f]+next_alloc
        dp_cache[(i,left)]=(best,best_alloc); return dp_cache[(i,left)]
    best_total=-float('inf'); final={}
    for Ff in range(F+1):
        penalty=C_PENALTY*(F-Ff)
        val,alloc=solve_dp(0,Ff)
        if val-penalty>best_total:
            best_total=val-penalty
            padded=alloc+[0]*(len(stores)-len(alloc))
            final={stores[i].id:padded[i] for i in range(len(stores))}
    return final

# --- SIMULATION ---
def simulate_policy(name, func):
    stores=[Store(1,"small",STORE_DEMAND['small'],30,int(STORE_DEMAND['small']*7)),
            Store(2,"medium",STORE_DEMAND['medium'],120,int(STORE_DEMAND['medium']*7)),
            Store(3,"medium",STORE_DEMAND['medium'],120,int(STORE_DEMAND['medium']*7)),
            Store(4,"large",STORE_DEMAND['large'],180,int(STORE_DEMAND['large']*7))]
    np.random.seed(42)
    metrics={'daily_profits':[], 'total_profit':0, 'total_online_fulfilled':0, 'total_online_orders':0}
    for t in range(1,R_PERIOD_DAYS+1):
        daily_profit=0
        for s in stores:
            d=poisson.rvs(s.mu_demand); sold=min(s.inventory,d)
            daily_profit+=sold*P_PRICE - s.inventory*C_HOLDING; s.inventory-=sold
        F=poisson.rvs(MU_0_ONLINE); metrics['total_online_orders']+=F
        if name=="MFP": alloc=func(stores,F)
        elif name=="TMFP": alloc=func(stores,F,t)
        else: alloc=func(stores,F,t,dummy_value_function_vj)
        fulfilled=sum(alloc.values())
        unfulfilled=F-fulfilled
        daily_profit += fulfilled*(P_PRICE-C_SHIPPING) - unfulfilled*C_PENALTY
        metrics['total_online_fulfilled']+=fulfilled
        for s in stores: s.inventory-=alloc[s.id]
        if t==L_LEAD_TIME_DAYS:
            for s in stores:
                s.inventory+=s.outstanding_order; s.outstanding_order=0
        metrics['daily_profits'].append(daily_profit)
        metrics['total_profit']+=daily_profit
    metrics['fulfillment_rate'] = metrics['total_online_fulfilled']/metrics['total_online_orders']
    return metrics

# --- MAIN ---
def main():
    print("\n=== POLICY PERFORMANCE COMPARISON (With Graphs) ===\n")
    results={'MFP':simulate_policy('MFP',mfp_fulfillment_policy),
             'TMFP':simulate_policy('TMFP',tmfp_fulfillment_policy),
             'OSIP':simulate_policy('OSIP',osip_fulfillment_policy)}
    print(f"{'Policy':<10} {'Total Profit ($)':>20} {'Online Fulfillment (%)':>30}")
    print("-"*65)
    for p in results:
        print(f"{p:<10} {results[p]['total_profit']:>20.2f} {results[p]['fulfillment_rate']*100:>30.2f}")
    print("-"*65)
    winner=max(results,key=lambda k:results[k]['total_profit'])
    print(f"🏆 Best Policy: {winner} (${results[winner]['total_profit']:.2f})")

    # --- VISUALIZATION ---
    policies=list(results.keys())

    # 1️⃣ Total Profit Comparison
    profits=[results[p]['total_profit'] for p in policies]
    plt.figure(figsize=(6,4))
    plt.bar(policies,profits)
    plt.title("Total Profit Comparison")
    plt.ylabel("Profit ($)")
    plt.grid(True,axis='y',alpha=0.4)
    plt.show()

    # 2️⃣ Daily Profit Curves
    plt.figure(figsize=(8,5))
    for p in policies:
        plt.plot(range(1,R_PERIOD_DAYS+1),results[p]['daily_profits'],label=p)
    plt.title("Daily Profit Over Time")
    plt.xlabel("Day"); plt.ylabel("Profit ($)")
    plt.legend(); plt.grid(True); plt.show()

    # 3️⃣ Fulfillment Rate Comparison
    rates=[results[p]['fulfillment_rate']*100 for p in policies]
    plt.figure(figsize=(6,4))
    plt.bar(policies,rates,color=['orange','skyblue','limegreen'])
    plt.title("Average Online Fulfillment Rate")
    plt.ylabel("Fulfillment Rate (%)")
    plt.grid(True,axis='y',alpha=0.4)
    plt.show()

if __name__=="__main__":
    main()
