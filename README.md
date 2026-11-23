# Mitigating Overseas Supply-Chain Vulnerability in Domestic Manufacturing via Multi-Period Mixed-Integer Linear Programming (MILP)

**Dual-Sourced Procurement, Intermodal Container Lot Sizing, and Inventory Resilience on a 26-Week U.S. Clean-Tech Benchmark**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Solver: SciPy HiGHS](https://img.shields.io/badge/Solver-HiGHS%20MILP-teal.svg)](https://highs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Domain: Clean-Tech Supply Chain](https://img.shields.io/badge/Domain-U.S.%20Manufacturing%20OR-orange.svg)](#empirical-calibration--governing-standards)

---

## Overview

This repository contains the reproducible synthetic supply-chain benchmark, non-anticipative rolling-horizon Mixed-Integer Linear Programming (MILP) optimizer, heuristic benchmarking suite, and 300-DPI figures accompanying the white paper *"Mitigating Overseas Supply-Chain Vulnerability in Domestic Manufacturing via Multi-Period Mixed-Integer Linear Programming (MILP)."*

U.S. domestic clean-tech and advanced manufacturing facilities face a structural procurement dilemma: offshore Tier-1 suppliers offer lower factory-gate unit prices, yet expose domestic assembly lines to multi-week transpacific ocean lead times, port congestion shocks, escalated **USTR Section 301 tariffs (25%–50%)**, and strict U.S. highway intermodal container weight limits. Conversely, domestic U.S. and USMCA nearshore suppliers provide 1-to-2-week ground replenishment and qualify for **IRC Section 45X Advanced Manufacturing Production Credits**, but operate under tighter weekly capacity caps and higher nominal sticker prices.

Evaluated across a **26-week (6-month) rolling operational horizon** for a U.S.-based **Grid-Scale Battery Energy Storage System (BESS)** assembly facility subjected to an unannounced 4-week transpacific port bottleneck (Weeks 11–14) and a +35% Q2/Q3 grid-interconnection demand surge (Weeks 14–19), the proposed **Non-Anticipative Rolling-Horizon MILP (Policy C)** outperforms traditional Enterprise Resource Planning (ERP) policies on an identical end-of-week information clock:

- **79.79% Cost Reduction vs. Static Single-Source $(s, S)$ Ocean (Policy A):** Cuts 26-week total supply-chain cost from **$105.22M** (54.66% line fill rate) to **$21.27M** (**96.07%** line fill rate).
- **30.35% Cost Reduction & 95.18% Air-Freight Cut vs. Reactive Air $(s, S)$ (Policy B1):** Saves **$9.27M** over Policy B1 ($30.54M), slashes emergency air cargo from **$4.57M** to **$220.66K**, and eliminates physical warehouse overflow (**79.93%** vs. **119.01%** peak utilization).
- **71.76% Pure Optimization Gain vs. Dual-Sourced Heuristic $(s, S)$ (Policy B2):** On the exact same multi-tier supplier network, replaces myopic rule-of-thumb replenishment ($75.32M) with multi-period container lot optimization ($21.27M), capturing **81.43%** U.S. domestic sourcing share and coming within **2.75%** of the clairvoyant **Perfect-Foresight Reference MILP ($20.70M)**.

---

## Repository Structure

```text
milp-supply-chain-resilience/
├── data/
│   ├── components.csv                 # 6 critical 100 kWh BESS sub-assemblies & cost/weight specs
│   ├── suppliers_lanes.csv            # 72 supplier-chassis-equipment lanes (OCEMA payload physics)
│   ├── weekly_demand_capacity.csv     # 26-week demand schedule & regional supplier capacity caps
│   └── in_transit_pipeline.csv        # Pre-horizon open purchase orders arriving Weeks 1–4
├── figures/
│   ├── fig1_cost_breakdown_benchmark.png      # 5-policy landed cost decomposition & line fill rate
│   ├── fig2_inventory_disruption_dynamics.png # LFP module buffer trajectory & stockout defense
│   ├── fig3_sourcing_mix_and_montecarlo.png   # Dynamic regional sourcing & N=50 Monte Carlo savings
│   └── fig4_policy_sensitivity_frontier.png   # IRC §45X credit & Section 301 tariff reshoring frontier
├── results/
│   ├── benchmark_comparison.csv       # Full 26-week financial & operational KPI table (Policies A–D)
│   └── monte_carlo_stress_test.csv    # 50-trial randomized unannounced port-strike results
├── src/
│   ├── generate_supply_chain_data.py  # OCEMA & NREL/USTR/IRS-calibrated dataset generator (seed=42)
│   └── optimize_inventory_milp.py     # HiGHS rolling-horizon MILP, heuristics, & Monte Carlo engine
├── requirements.txt
└── README.md
```

---

## Optimization & Simulation Architecture

```text
OCEMA & Policy-Calibrated Inputs (data/, seed=42)
        │
        ▼
[Identical End-of-Week Decision Clock (t = 1 ... 26)]
  ├── Step 1: Receive scheduled arrivals & apply realized week-t demand D_{i,t}
  ├── Step 2: Update ending inventory I_{i,t}, backlog B_{i,t}, & observe week-t port capacity
  └── Step 3: Place end-of-week-t replenishment orders using strictly non-anticipative state
        │
        ├──► [Baseline Heuristics: Policies A, B1, B2]
        │      • Stationary (s, S) min-max triggers across single-source, reactive air, & dual-sourced rules
        │
        ├──► [Primary Operational Model: Policy C — Rolling-Horizon MILP]
        │      • Solves multi-period container/trailer MILP over [t+1 .. T] via HiGHS (~0.78s/epoch)
        │      • Uses strictly forecast demand & explicit 2-week strike recovery belief rule
        │      • Executes ONLY week-t immediate integer container/trailer orders
        │
        └──► [Clairvoyant Benchmark: Policy D — Perfect-Foresight Reference MILP]
               • Evaluates theoretical cost benchmark under full horizon visibility
```

---

## Mathematical Formulation (MILP Architecture)

### Sets, Indices & Decision Variables
* $i \in \mathcal{I} = \{1, \dots, 6\}$: Critical BESS sub-assemblies (`CMP_LFP_MOD`, `CMP_PCS_INV`, `CMP_BMS_CTL`, `CMP_THM_PLT`, `CMP_HV_HARN`, `CMP_ENC_CAB`).
* $l \in \mathcal{L}_i$: Qualified supplier-equipment lanes across 4 suppliers (`SUP_APAC_1`, `SUP_APAC_2`, `SUP_MEX_1`, `SUP_USA_1`), 2 freight modes (`std`, `exp`), and chassis-specific equipment (`20ft_Dry_Slider`, `20ft_Dry_TriAxle`, `40ft_HC_Gooseneck`, `53ft_FTL_DryVan`, `26ft_Box_Truck`, `LTL_Pallet_Lot`, `Air_PMC_Pallet`).
* $N_{l,t} \in \mathbb{Z}_{\ge 0}$: Integer containers, trailers, or pallet lots dispatched on lane $l$ in week $t$.
* $Y_{l,t} \in \{0, 1\}$: Binary order setup indicator ($1$ if lane $l$ is utilized in week $t$, $0$ otherwise).
* $I_{i,t}, B_{i,t} \ge 0$: Ending on-hand inventory and cumulative unfulfilled backlog (units) of component $i$ in week $t$.

### Objective Function & Operational Constraints

Minimize total 26-week net landed supply-chain cost across unit procurement ($c_l$), Section 301 import tariffs ($\tau_l$), IRC §45X domestic credits ($\delta_l$), intermodal freight ($f_l$), fixed dispatch/customs fees ($K_l$), weekly holding cost ($h_i$), and assembly line-stoppage penalties ($p_i$):

$$\min Z = \sum_{t \in \mathcal{T}} \sum_{i \in \mathcal{I}} \left[ \sum_{l \in \mathcal{L}_i} \left( \left(c_l + \tau_l - \delta_l + f_l\right) q_l N_{l,t} + K_l Y_{l,t} \right) + h_i I_{i,t} + p_i B_{i,t} \right]$$

Subject to exact OCEMA container lot capacities $q_l = \min\left(\lfloor V_l / v_i \rfloor, \lfloor W_l / (1.07 \, w_i^{\text{net}}) \rfloor, q_i^{\max}\right)$ and the following operational constraints:

1. **Multi-Period Inventory Flow Balance (with Lane Lead Times $L_l$ and Pre-Horizon Pipeline $\bar{Q}_{i,t}$):**
   $$I_{i,t} - B_{i,t} = I_{i,t-1} - B_{i,t-1} + \bar{Q}_{i,t} + \sum_{l \in \mathcal{L}_i : t - L_l \ge 1} q_l N_{l, t - L_l} - D_{i,t} \quad \forall i \in \mathcal{I}, \; t \in \mathcal{T}$$
2. **Minimum Order Quantity (MOQ) & Tightened Big-M Lot Linking:**
   $$m_l Y_{l,t} \le N_{l,t} \le U_{l,t} Y_{l,t}, \qquad U_{l,t} = \min\left(\left\lfloor \frac{\text{Cap}_{s(l),i,t}}{q_l} \right\rfloor, \left\lfloor \frac{4 D_i^{\text{forecast, peak}}}{q_l} \right\rfloor\right) \quad \forall l \in \mathcal{L}, \; t \in \mathcal{T}$$
3. **Weekly Supplier Production & Port Capacity Limits:**
   $$\sum_{l \in \mathcal{L}_{i,s}} q_l N_{l,t} \le \text{Cap}_{i,s,t} \quad \forall i \in \mathcal{I}, \; s \in \mathcal{S}, \; t \in \mathcal{T}$$
4. **U.S. Assembly Plant Physical Warehouse Cubic Limit & Terminal Safety Stock:**
   $$\sum_{i \in \mathcal{I}} v_i I_{i,t} \le 480.0\text{ m}^3 \quad \forall t \in \mathcal{T}, \qquad I_{i,T} \ge \text{SS}_i \quad \forall i \in \mathcal{I}$$

---

## Summary of Empirical Results & Visual Findings

### Deterministic 26-Week Five-Policy Benchmark (`benchmark_comparison.csv`)

| Policy Architecture | Total 26-Wk Cost ($) | Net Proc. & §301 Tariffs (less §45X) | Std. Freight & Setup ($) | Emergency Air Freight ($) | Holding Cost ($) | Stockout Penalty ($) | Line Fill Rate (%) | U.S. Domestic Share (%) | Peak Whse. Util. (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Static Single-Source $(s, S)$ Ocean** | $105,221,942 | $17,270,000 | $599,711 | $0 | $33,661 | $87,318,570 | 54.66% | 0.00% | 35.04% |
| **B1: Reactive Air-Expediting $(s, S)$** | $30,541,137 | $25,321,050 | $500,754 | $4,573,447 | $145,886 | $0 | **100.00%** | 0.00% | 119.01% *(Overflow)* |
| **B2: Dual-Sourced Heuristic $(s, S)$** | $75,317,829 | $17,414,112 | $562,617 | $0 | $119,680 | $57,221,420 | 69.51% | 21.38% | 95.31% |
| **C: Rolling-Horizon MILP (Online)** | **$21,270,961** | **$17,337,011** | **$228,161** | **$220,658** | **$170,831** | **$3,314,300** | **96.07%** | **81.43%** | **79.93%** *(Feasible)* |
| **D: Perfect-Foresight Reference MILP** | $20,700,813 | $16,712,685 | $232,250 | $188,790 | $114,358 | $3,452,730 | 95.09% | 81.84% | 56.31% *(Feasible)* |

### Non-Anticipative Monte Carlo Stress Test (`N = 50` Trials, `monte_carlo_stress_test.csv`)
Across 50 randomized trials varying unannounced port-strike start weeks ($t_{\text{start}} \in [6, 16]$), strike durations ($2\text{–}5\text{ weeks}$), and weekly demand volatility ($\sigma = 8\%$):
- **Mean Savings vs. Static $(s, S)$ (Policy A):** **79.21%** (Interquartile Range: **78.14% – 80.03%**)
- **Mean Savings vs. Reactive Air $(s, S)$ (Policy B1):** **28.61%**
- **Mean Savings vs. Dual-Sourced Heuristic (Policy B2):** **71.83%** (Mean Policy C Line Fill Rate: **95.92%**)

### Key Diagnostic Visuals (`figures/`)

- **`fig1_cost_breakdown_benchmark.png`:** Decomposes total 26-week landed costs and production line fill rates across Policies A–D, isolating the value of dual-sourcing network access (A $\rightarrow$ B2) from multi-period MILP optimization (B2 $\rightarrow$ C).
- **`fig2_inventory_disruption_dynamics.png`:** Tracks on-hand inventory buffers and assembly line backorders for 25 kWh LFP battery modules across the Week 11–14 port bottleneck and Week 14–19 grid demand surge.
- **`fig3_sourcing_mix_and_montecarlo.png`:** Illustrates Policy C's dynamic weekly reallocation across U.S. Domestic (81.43%), USMCA Nearshore (5.99%), and Offshore APAC (12.58%) alongside the $N=50$ Monte Carlo cost-reduction boxplots.
- **`fig4_policy_sensitivity_frontier.png`:** Maps optimal U.S. domestic reshoring share and plant landed cost across IRC §45X credit levels (0%–125% of statutory) under baseline and +40% escalated Section 301 tariff regimes.

---

## Quickstart & Reproducibility

### 1. Clone the Repository & Install Dependencies

```bash
git clone https://github.com/jenny6449/milp-supply-chain-resilience.git
cd milp-supply-chain-resilience
pip install -r requirements.txt
```

### 2. Regenerate the OCEMA-Calibrated Supply-Chain Benchmark *(Optional)*

```bash
python src/generate_supply_chain_data.py
```

### 3. Run Non-Anticipativity Verification, 5-Policy Benchmark, & Monte Carlo Simulation

```bash
python src/optimize_inventory_milp.py
```

---

## Empirical Calibration & Governing Standards

- **NREL (`NREL/TP-6A40-93281`) & BloombergNEF:** 2025–2026 utility-scale BESS component cost benchmarks ($210/kWh delivered hardware, $372/kW PCS) and North American manufacturing cost differentials.
- **USTR Section 301 Modifications (`89 FR 76581`, Jan 1, 2026):** 25.0% import tariff on non-EV lithium-ion battery modules/parts and steel enclosures; 50.0% tariff on power semiconductors (PCS & BMS); 0.0% duty under USMCA rules of origin.
- **IRS IRC Section 45X (`26 U.S.C. § 45X` / Form 7207):** $35/kWh domestic battery cell-to-module production credit and $0.02/W AC commercial inverter production credit.
- **OCEMA / Maersk U.S. Highway Weight Guidelines & U.S. DOT (80,000 lb GVW):** Intermodal container gross cargo limits (17,780 kg 20-ft Standard Slider; 19,960 kg 20-ft Tri-Axle; 19,820 kg 40-ft High-Cube Gooseneck; 19,500 kg 53-ft Dry Van) inclusive of +7% packaging and dunnage tare.
- **Freightos Baltic Index (`FBX01`), Freightos Air Index (`FAX`), & ASCM (APICS):** Transpacific ocean container rates, emergency air-cargo rates ($5.80–$6.10/kg), holding cost (~0.45%/wk), and assembly line-stoppage SLA benchmarks.

---

## Author & Citation

**Author:** Juyeon Cho (Jenny), M.S. in Business Analytics (Duke University) — *Manufacturing Data & Analytics Engineer*

If you reference this benchmark dataset or rolling-horizon MILP optimization framework in academic or industrial work, please cite:

```bibtex
@misc{milp_supply_chain_resilience_2026,
  author       = {Cho, Juyeon},
  title        = {Mitigating Overseas Supply-Chain Vulnerability in Domestic Manufacturing via Multi-Period Mixed-Integer Linear Programming (MILP)},
  year         = {2026},
  publisher    = {GitHub / Zenodo},
  url          = {https://github.com/chochoju/milp-supply-chain-resilience}
}
```