# Operation Research Project

## Structure

- `src/simulation/` — Simulation and policy scripts
- `src/optimization/` — MILP optimization
- `src/visualization/` — Visualization scripts
- `Dataset/` — Source CSV data (previously duplicated under `data/raw/`)
- `results/` — Output files
- `requirements.txt` — Python dependencies

## Usage

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   For MILP, install GLPK and add to PATH.

2. **Run simulations**
   ```bash
   python src/simulation/sim_theoretical.py
   python src/simulation/sim_real_data.py
   python src/simulation/sim_corrected.py
   python src/simulation/policy_comparison.py
   ```

3. **Run optimization**
   ```bash
   python src/optimization/milp_inventory_optimization.py
   ```

4. **Visualize results**
   ```bash
   python src/visualization/visual_corrected_sim.py
   python src/visualization/visual_original_sim.py
   python src/visualization/visual_policy_comparison.py
   ```

5. **Interactive decision tool**
   ```bash
   python src/interactive/osip_decision_tool.py
   ```

## File Descriptions

### Simulations (`src/simulation/`)
- `sim_theoretical.py` — Original theoretical simulation (base test case)
- `sim_real_data.py` — Simulation using real product data parameters
- `sim_corrected.py` — Corrected demand distribution across all stores
- `policy_comparison.py` — Compares MFP, TMFP, and OSIP policies

### Optimization (`src/optimization/`)
- `milp_inventory_optimization.py` — Deterministic MILP optimization using Pyomo

### Visualization (`src/visualization/`)
- `visual_original_sim.py` — Visualizations for original simulation
- `visual_corrected_sim.py` — Visualizations for corrected simulation
- `visual_policy_comparison.py` — Policy comparison visualizations

### Interactive Tools (`src/interactive/`)
- `osip_decision_tool.py` — Interactive OSIP decision maker

## Issues & Improvements

- Original simulation assigned all demand to one store. Fixed in `sim_corrected.py`.
- Policy comparison shows which policy (MFP/TMFP/OSIP) performs best under different scenarios.
- MILP optimization provides theoretical upper bound benchmark for policy performance.
- Visualizations help communicate results and compare approaches.

## Data

Raw data now lives only in `Dataset/` (duplicate `data/raw/` placeholders removed). Outputs and results go in `results/`.

## Requirements

- Python 3.x
- numpy, scipy, matplotlib, pyomo
- GLPK (for MILP)

## References and Sources

### Dataset
- **Omnichannel Retail Datasets** - Bhowmick, J. & Pazour, J. (2024)
  - Repository: [JoyjitBhowmick/Omnichannel-Retail-Datasets](https://github.com/JoyjitBhowmick/Omnichannel-Retail-Datasets)
  - Citation: Joyjit Bhowmick and Jennifer Pazour. "A Connected In-Store and Online Customer Data Set for Omnichannel Retail Logistics Research." In *IIE Annual Conference Proceedings*, pages 1–6. Institute of Industrial and Systems Engineers (IISE), 2024.

### Key Data Files
- `In-store Customer and Online Order Arrivals.csv` - Customer arrival patterns by hour and day
- `Product Information.csv` - Product dimensions, weight, and pricing data
- `Demand and Shopping Behavior.csv` - Customer purchase patterns for in-store and online channels
- `Product Family Mapping.csv` - Mapping between online (Instacart) and in-store product categories
- `Aisle ID to Products.csv` - Product family definitions and groupings

### Methodology References
- **Instacart Dataset** - Online grocery shopping patterns and product categories
- **Google Popular Times** - In-store customer arrival estimation
- **Fulfillment Policies**:
  - MFP (Myopic Fulfillment Policy) - Greedy allocation strategy
  - TMFP (Threshold-based Myopic Fulfillment Policy) - Inventory-constrained allocation
  - OSIP (One-Step Policy Improvement) - Dynamic programming approach

### Tools and Libraries
- **Pyomo** - Python-based optimization modeling language
- **GLPK** - GNU Linear Programming Kit (open-source solver)
- **NumPy/SciPy** - Numerical computing and statistical distributions
- **Matplotlib** - Data visualization

---

For details, see comments in each script and the full documentation above.