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
   python src/simulation/a.py
   python src/simulation/b.py
   python src/simulation/a_corrected.py
   python src/simulation/policy_comparison.py
   ```

3. **Run optimization**
   ```bash
   python src/optimization/milp_optimization.py
   ```

4. **Visualize results**
   ```bash
   python src/visualization/visual_corrected_sim.py
   python src/visualization/visual_original_sim.py
   python src/visualization/visual_policy_comparison.py
   ```

## Issues & Improvements

- Original simulation (`a.py`) assigned all demand to one store. Corrected in `a_corrected.py`.
- Policy comparison (`policy_comparison.py`) shows which policy is best under different scenarios.
- MILP optimization (`milp_optimization.py`) provides a benchmark for policy performance.
- Visualizations help communicate results and compare approaches.
- (Interactive tool removed on request to simplify repository.)

## Data

Raw data now lives only in `Dataset/` (duplicate `data/raw/` placeholders removed). Outputs and results go in `results/`.

## Requirements

- Python 3.x
- numpy, scipy, matplotlib, pyomo
- GLPK (for MILP)

---

For details, see comments in each script and the full documentation above.