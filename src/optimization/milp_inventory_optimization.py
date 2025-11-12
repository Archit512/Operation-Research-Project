from pyomo.environ import ConcreteModel, Var, NonNegativeIntegers, NonNegativeReals, RangeSet, Param, Constraint, Objective, SolverFactory, maximize, value
import pandas as pd, numpy as np, matplotlib.pyplot as plt, math, os, sys

# params
P_PRICE, C_HOLDING, C_SHIPPING, C_PENALTY = 7.6, 1.0, 10.0, 10.0
CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "In-store Customer and Online Order Arrivals.csv")
if not os.path.exists(CSV): raise SystemExit(f"CSV not found: {CSV}")

df = pd.read_csv(CSV)
df.columns = df.columns.str.strip().str.lower()
required = ['day','avg. in-store customer arrivals','avg. online order arrivals']
if any(c not in df.columns for c in required): raise SystemExit("CSV missing required columns: " + ", ".join(required))

df = df.sort_values('day').reset_index(drop=True)
R = min(7, len(df))
in_store = df['avg. in-store customer arrivals'].astype(float).iloc[:R].tolist()
online = df['avg. online order arrivals'].astype(float).iloc[:R].tolist()

# simulate 4 stores from aggregated CSV
STORE_IDS = [1,2,3,4]; STORE_TYPES={1:"small",2:"medium",3:"medium",4:"large"}
WEIGHTS={1:0.10,2:0.30,3:0.30,4:0.30}
D = {(j,t): int(round(in_store[t-1]*WEIGHTS[j])) for t in range(1,R+1) for j in STORE_IDS}
F = {t:int(round(online[t-1])) for t in range(1,R+1)}
I0 = {1:30,2:120,3:120,4:180}
OUT = {j:int(math.ceil(sum(in_store)*WEIGHTS[j])) for j in STORE_IDS}

# build model
model = ConcreteModel()
model.J = RangeSet(1,len(STORE_IDS))
model.T = RangeSet(0,R)
py_to_store = {i:STORE_IDS[i-1] for i in model.J}
model.d = Param(model.J, model.T, initialize=lambda m,i,t: 0 if t==0 else D.get((py_to_store[i],t),0), within=NonNegativeReals)
model.F = Param(model.T, initialize=lambda m,t: 0 if t==0 else F.get(t,0), within=NonNegativeReals)
model.I0 = Param(model.J, initialize=lambda m,i: I0[py_to_store[i]], within=NonNegativeReals)
model.OUT = Param(model.J, initialize=lambda m,i: OUT[py_to_store[i]], within=NonNegativeReals)

model.sell = Var(model.J, model.T, domain=NonNegativeIntegers)
model.f = Var(model.J, model.T, domain=NonNegativeIntegers)
model.Q = Var(model.J, model.T, domain=NonNegativeIntegers)
model.I = Var(model.J, model.T, domain=NonNegativeIntegers)
model.U = Var(model.T, domain=NonNegativeIntegers)

model.init_inventory = Constraint(model.J, model.T, rule=lambda m,j,t: (m.I[j,0]==m.I0[j]) if t==0 else Constraint.Skip)
model.demand_limit = Constraint(model.J, model.T, rule=lambda m,j,t: (m.sell[j,t]==0) if t==0 else (m.sell[j,t] <= m.d[j,t]))
def inv_balance(m,j,t):
    if t==0: return Constraint.Skip
    arrivals=0
    if t==2: arrivals+=m.OUT[j]
    if (t-2)>=1: arrivals+=m.Q[j,t-2]
    return m.I[j,t]==m.I[j,t-1]-m.sell[j,t]-m.f[j,t]+arrivals
model.inventory_balance = Constraint(model.J, model.T, rule=inv_balance)
def cons_limit(m,j,t):
    if t==0: return Constraint.Skip
    arrivals=0
    if t==2: arrivals+=m.OUT[j]
    if (t-2)>=1: arrivals+=m.Q[j,t-2]
    return m.sell[j,t]+m.f[j,t] <= m.I[j,t-1]+arrivals
model.consumption_limit = Constraint(model.J, model.T, rule=cons_limit)
model.online_balance = Constraint(model.T, rule=lambda m,t: (m.U[t]==0) if t==0 else (sum(m.f[j,t] for j in m.J)+m.U[t]==m.F[t]))

model.obj = Objective(rule=lambda m: sum(P_PRICE*m.sell[j,t] for j in m.J for t in m.T if t!=0)
                      + sum((P_PRICE-C_SHIPPING)*m.f[j,t] for j in m.J for t in m.T if t!=0)
                      - sum(C_HOLDING*m.I[j,t] for j in m.J for t in m.T if t!=0)
                      - sum(C_PENALTY*m.U[t] for t in m.T if t!=0),
                      sense=maximize)

# solve
def try_solvers(m):
    for s in ['glpk','cbc']:
        try:
            solver = SolverFactory(s)
            if solver.available():
                solver.solve(m, tee=False)
                return s
        except Exception:
            continue
    raise RuntimeError("No solver available (glpk/cbc)")

solver_used = try_solvers(model)

# extract results
total_profit=total_instore=total_online=total_unfilled=total_hold=0
daily_profit=[0.0]*(R+1)
for t in range(1,R+1):
    rev=0
    for j in model.J:
        sold=int(round(value(model.sell[j,t])))
        f_j=int(round(value(model.f[j,t])))
        I_end=int(round(value(model.I[j,t])))
        rev += sold*P_PRICE + f_j*(P_PRICE-C_SHIPPING)
        total_instore+=sold; total_online+=f_j; total_hold+=I_end*C_HOLDING
    unfilled=int(round(value(model.U[t])))
    total_unfilled+=unfilled
    rev -= unfilled*C_PENALTY
    daily_profit[t]=rev; total_profit+=rev

print(f"Solver: {solver_used}")
print(f"Profit: ${total_profit:,.2f}  In-store sold: {total_instore}  Online fulfilled: {total_online}  Unfulfilled: {total_unfilled}")

for t in range(1,R+1):
    fulfilled=sum(int(round(value(model.f[j,t]))) for j in model.J)
    print(f"Day {t}: Online {F.get(t,0)} Fulfilled {fulfilled} Unfilled {int(round(value(model.U[t])))} Profit {daily_profit[t]:.2f}")

# plots
days=list(range(1,R+1))
plt.figure(figsize=(7,3)); plt.plot(days,[daily_profit[d] for d in days],marker='o'); plt.title("Daily Profit"); plt.grid(True); plt.show()
plt.figure(figsize=(7,3)); plt.plot(days,[F.get(d,0) for d in days],marker='o',label='Online'); plt.plot(days,[sum(int(round(value(model.f[j,d]))) for j in model.J) for d in days],marker='s',label='Fulfilled'); plt.legend(); plt.grid(True); plt.show()
