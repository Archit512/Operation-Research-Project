#!/usr/bin/env python3
"""
milp_optimization.py

Deterministic MILP optimizing expected-profit over a 7-day horizon.
Assumes expected demands (mu) as deterministic daily demand.
Solves with Pyomo + GLPK (or any SCIP/GLPK/CBC solver available).
"""

from pyomo.environ import (
    ConcreteModel, Var, NonNegativeIntegers, NonNegativeReals, RangeSet,
    Param, Constraint, Objective, SolverFactory, maximize, value
)
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

# -------------------------
# 1) PARAMETERS (from your corrected setup)
# -------------------------
P_PRICE = 7.6
C_HOLDING = 1.0
C_PRODUCT = 5.32
C_SHIPPING = 10.0
C_PENALTY = 10.0
R_PERIOD_DAYS = 7
L_LEAD_TIME_DAYS = 2
MU_0_ONLINE = 39.86094584

# store-level expected (mu) values (same as a_corrected distribution sample)
# We'll define 4 stores: 1 small, 2 mediums, 1 large (same as your scripts)
STORE_IDS = [1, 2, 3, 4]
STORE_TYPES = {1: "small", 2: "medium", 3: "medium", 4: "large"}
# expected per-store mu (rounded to sensible ints)
STORE_MU = {
    1: 6.71,   # small
    2: 26.84,  # medium
    3: 26.84,  # medium
    4: 40.25   # large
}

# initial inventories same as a_corrected sample
I0 = {1: 30, 2: 120, 3: 120, 4: 180}

# outstanding orders that were placed before period start (arrive at t=L)
OUTSTANDING = {j: int(math.ceil(STORE_MU[j] * 7)) for j in STORE_IDS}

# deterministic (expected) daily in-store demand and online orders
D = { (j, t): int(round(STORE_MU[j])) for j in STORE_IDS for t in range(1, R_PERIOD_DAYS + 1) }
F = { t: int(round(MU_0_ONLINE)) for t in range(1, R_PERIOD_DAYS + 1) }

# -------------------------
# 2) BUILD Pyomo MODEL
# -------------------------
model = ConcreteModel(name="OmniChannel_MILP")

# index sets
model.J = RangeSet(1, len(STORE_IDS))           # stores indices 1..4
model.T = RangeSet(0, R_PERIOD_DAYS)           # 0..R (we use t=0 as initial inventory)
# map pyomo store index to our store ids
py_to_store = {i: STORE_IDS[i-1] for i in model.J}

# Parameters (mapped)
def mu_init(model, i):
    return STORE_MU[py_to_store[i]]
model.mu = Param(model.J, initialize=mu_init, within=NonNegativeReals)

def d_init(model, i, t):
    if t == 0:
        return 0
    return D[(py_to_store[i], t)]
model.d = Param(model.J, model.T, initialize=d_init, within=NonNegativeReals)

def F_init(model, t):
    if t == 0:
        return 0
    return F[t]
model.F = Param(model.T, initialize=F_init, within=NonNegativeReals)

# initial inventory param I0_j
def I0_init(model, i):
    return I0[py_to_store[i]]
model.I0 = Param(model.J, initialize=I0_init, within=NonNegativeReals)

# outstanding orders param
def out_init(model, i):
    return OUTSTANDING[py_to_store[i]]
model.OUT = Param(model.J, initialize=out_init, within=NonNegativeReals)

# Decision variables:
# sell[j,t] : number of in-store customers served at store j on day t (integer)
# f[j,t]    : number of online orders fulfilled from store j on day t (integer)
# Q[j,t]    : replenishment quantity ordered at store j on day t (integer)
# I[j,t]    : inventory at end of day t for store j (integer, >=0)
# U[t]      : unfulfilled online orders on day t (integer, >=0)

model.sell = Var(model.J, model.T, domain=NonNegativeIntegers, bounds=(0, 100000))
model.f = Var(model.J, model.T, domain=NonNegativeIntegers, bounds=(0, 100000))
model.Q = Var(model.J, model.T, domain=NonNegativeIntegers, bounds=(0, 100000))
model.I = Var(model.J, model.T, domain=NonNegativeIntegers, bounds=(0, 100000))
model.U = Var(model.T, domain=NonNegativeIntegers, bounds=(0, 100000))

# -------------------------
# 3) CONSTRAINTS
# -------------------------

# a) initial inventory: I[j,0] == I0_j
def init_inventory_rule(m, j, t):
    if t == 0:
        return m.I[j, 0] == m.I0[j]
    return Constraint.Skip
model.init_inventory = Constraint(model.J, model.T, rule=init_inventory_rule)

# b) sales and inventory feasibility: sell[j,t] <= demand[j,t]
def demand_limit_rule(m, j, t):
    if t == 0:
        return m.sell[j, t] == 0
    return m.sell[j, t] <= m.d[j, t]
model.demand_limit = Constraint(model.J, model.T, rule=demand_limit_rule)

# c) inventory balance for t >=1:
# I[j,t] = I[j,t-1] - sell[j,t] - f[j,t] + arrivals
def inventory_balance_rule(m, j, t):
    if t == 0:
        return Constraint.Skip
    arrivals = 0
    # outstanding orders that were placed before time 0 arrive at t == L
    if t == L_LEAD_TIME_DAYS:
        arrivals += m.OUT[j]
    # arrivals from Q ordered previously: Q[j, t-L] arrives at t if t-L >= 1
    if (t - L_LEAD_TIME_DAYS) >= 1:
        arrivals += m.Q[j, t - L_LEAD_TIME_DAYS]
    return m.I[j, t] == m.I[j, t-1] - m.sell[j, t] - m.f[j, t] + arrivals
model.inventory_balance = Constraint(model.J, model.T, rule=inventory_balance_rule)

# d) inventory non-negativity is enforced by variable domain (>=0)
# But also ensure that sell + f cannot exceed inventory available at start of day:
# start_of_day_inventory = I[j,t-1] + arrivals_that_arrive_at_start? (arrivals we set happen at replenishment epoch)
# We consider both sell and f consume inventory available at start (I[j,t-1] + arrivals that occur at start of day).
# For consistency with previous models, arrivals from outstanding orders occur at the end of epoch 2 on day L; here we treat inventory available for sales before arrivals.
# To remain conservative: enforce sell+f <= I[j,t-1] + arrivals_that_arrive_now
def consumption_limit_rule(m, j, t):
    if t == 0:
        return Constraint.Skip
    # arrivals that are available at start of day t (we assume orders arriving on day t are available before sales)
    arrivals_now = 0
    if t == L_LEAD_TIME_DAYS:
        arrivals_now += m.OUT[j]
    if (t - L_LEAD_TIME_DAYS) >= 1:
        arrivals_now += m.Q[j, t - L_LEAD_TIME_DAYS]
    return m.sell[j, t] + m.f[j, t] <= m.I[j, t-1] + arrivals_now
model.consumption_limit = Constraint(model.J, model.T, rule=consumption_limit_rule)

# e) online demand balance per day:
# sum_j f[j,t] + U[t] == F[t]
def online_balance_rule(m, t):
    if t == 0:
        return m.U[t] == 0
    return sum(m.f[j, t] for j in m.J) + m.U[t] == m.F[t]
model.online_balance = Constraint(model.T, rule=online_balance_rule)

# -------------------------
# 4) OBJECTIVE: maximize expected profit
# -------------------------
def objective_rule(m):
    revenue_instore = sum(P_PRICE * m.sell[j, t] for j in m.J for t in m.T if t != 0)
    revenue_online = sum((P_PRICE - C_SHIPPING) * m.f[j, t] for j in m.J for t in m.T if t != 0)
    holding_costs = sum(C_HOLDING * m.I[j, t] for j in m.J for t in m.T if t != 0)
    penalty_costs = sum(C_PENALTY * m.U[t] for t in m.T if t != 0)
    # (product cost is constant equal to number sold * C_PRODUCT - can be included if desired)
    return revenue_instore + revenue_online - holding_costs - penalty_costs
model.obj = Objective(rule=objective_rule, sense=maximize)

# -------------------------
# 5) SOLVE
# -------------------------
def solve_model(model, solver_name='glpk'):
    print("Solving with solver:", solver_name)
    solver = SolverFactory(solver_name)
    if not solver.available():
        raise RuntimeError(f"Solver {solver_name} not available in this environment.")
    results = solver.solve(model, tee=False)
    print("Solver status:", results.solver.status, "| Termination condition:", results.solver.termination_condition)
    return results

if __name__ == "__main__":
    try:
        # prefer glpk; if not available, try 'cbc' if installed
        solver_to_try = 'glpk'
        try:
            solve_model(model, solver_to_try)
        except Exception as e:
            print("glpk not available or failed:", e)
            # fallback to any available solver
            for s in ['cbc', 'gurobi', 'cplex']:
                try:
                    solve_model(model, s)
                    solver_to_try = s
                    break
                except Exception:
                    continue
        # Extract solution
        # Compute metrics
        total_instore_sold = 0
        total_online_fulfilled = 0
        total_unfulfilled = 0
        total_holding_cost = 0
        total_profit = 0.0
        daily_profit = [0.0] * (R_PERIOD_DAYS + 1)

        for t in range(1, R_PERIOD_DAYS + 1):
            day_rev = 0.0
            for j in model.J:
                sold = int(round(value(model.sell[j, t])))
                f_j = int(round(value(model.f[j, t])))
                I_end = int(round(value(model.I[j, t])))
                day_rev += sold * P_PRICE + f_j * (P_PRICE - C_SHIPPING)
                total_instore_sold += sold
                total_online_fulfilled += f_j
                total_holding_cost += I_end * C_HOLDING
            unfulfilled = int(round(value(model.U[t])))
            total_unfulfilled += unfulfilled
            day_rev -= unfulfilled * C_PENALTY
            daily_profit[t] = day_rev
            total_profit += day_rev

        # Print summary
        print("\n=== MILP OPTIMIZATION RESULTS ===")
        print(f"Total profit (expected): ${total_profit:,.2f}")
        print(f"Total in-store sold: {total_instore_sold}")
        print(f"Total online fulfilled: {total_online_fulfilled}")
        print(f"Total online unfulfilled (penalties): {total_unfulfilled}")
        print(f"Total holding cost (approx): ${total_holding_cost:,.2f}")

        # show per-day result and per-store allocations (truncated)
        print("\nDay | F (online) | fulfilled (sum f_j) | unfulfilled | profit")
        for t in range(1, R_PERIOD_DAYS + 1):
            fulfilled = sum(int(round(value(model.f[j, t]))) for j in model.J)
            print(f"{t:3d} | {F[t]:10d} | {fulfilled:18d} | {int(round(value(model.U[t]))):11d} | {daily_profit[t]:8.2f}")

        # per-store allocation table
        print("\nPer-store end inventories (I[j,t] for t=1..T):")
        for j in model.J:
            store_id = py_to_store[j]
            row = [int(round(value(model.I[j, t]))) for t in model.T if t != 0]
            print(f"Store {store_id} ({STORE_TYPES[store_id]}):", row)

        # -------------------------
        # Simple plots: daily profit and fulfilled vs F
        # -------------------------
        days = list(range(1, R_PERIOD_DAYS + 1))
        profits = [daily_profit[t] for t in days]
        fulfilleds = [sum(int(round(value(model.f[j, t]))) for j in model.J) for t in days]
        Fs = [F[t] for t in days]

        plt.figure(figsize=(8,4))
        plt.plot(days, profits, marker='o')
        plt.title("MILP: Daily Profit")
        plt.xlabel("Day"); plt.ylabel("Profit ($)"); plt.grid(True); plt.show()

        plt.figure(figsize=(8,4))
        plt.plot(days, Fs, marker='o', label='Online Orders F_t')
        plt.plot(days, fulfilleds, marker='s', label='Online Fulfilled (sum f_j)')
        plt.title("MILP: Online Orders vs Fulfilled")
        plt.xlabel("Day"); plt.ylabel("Units"); plt.legend(); plt.grid(True); plt.show()

    except Exception as exc:
        print("Error during MILP solve or postprocessing:", exc)
        sys.exit(1)
