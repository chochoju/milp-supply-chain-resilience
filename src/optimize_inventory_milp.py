"""
Multi-Period Mixed-Integer Linear Programming (MILP) for U.S. Domestic
Manufacturing Supply-Chain & Inventory Resilience

Methodological Architecture:
1. Identical End-of-Week Decision Timing across all online policies (A, B1, B2, C):
   (1) Receive scheduled deliveries & apply week t realized demand;
   (2) Update ending inventory I_{i,t}, backlog B_{i,t}, and observe week t port capacity;
   (3) Place week t orders using updated state, open pipeline, and forecasts for t+1..T.
2. Five-Policy Benchmark Comparison:
   - Policy A : Static Single-Source (s, S) Ocean (Offshore APAC only)
   - Policy B1: Reactive Air-Expediting (s, S) (Offshore APAC Ocean + Emergency Air)
   - Policy B2: Dual-Sourced Heuristic (s, S) (Access to U.S. Domestic, USMCA & APAC)
   - Policy C : Non-Anticipative Rolling-Horizon MILP (Primary Operational Benchmark;
                uses forecast demand & explicit 2-week strike recovery belief)
   - Policy D : Perfect-Foresight Reference MILP (Clairvoyant benchmark)
3. Automated Non-Anticipativity Verification Test & balanced Monte Carlo Stress Test.
4. Exports 4 300-DPI figures in `figures/` and CSVs in `results/`.

Seed: 42
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.optimize import milp, LinearConstraint, Bounds


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_comp = pd.read_csv(os.path.join(data_dir, "components.csv"))
    df_lanes = pd.read_csv(os.path.join(data_dir, "suppliers_lanes.csv"))
    df_sched = pd.read_csv(os.path.join(data_dir, "weekly_demand_capacity.csv"))
    df_pipe = pd.read_csv(os.path.join(data_dir, "in_transit_pipeline.csv"))
    return df_comp, df_lanes, df_sched, df_pipe


def compute_component_fill_rate(backlog: np.ndarray, demand: np.ndarray) -> float:
    """Share of current demand served in its due week under FIFO fulfillment."""
    total_demand = float(np.sum(demand))
    if total_demand == 0:
        return 100.0
    current_demand_unfilled = np.minimum(backlog, demand)
    return 100.0 * (1.0 - float(np.sum(current_demand_unfilled)) / total_demand)


def solve_milp_subproblem(
    df_comp: pd.DataFrame,
    df_lanes: pd.DataFrame,
    init_inv_arr: np.ndarray,
    init_bo_arr: np.ndarray,
    pipe_future: np.ndarray,
    demand_future: np.ndarray,
    cap_future: dict,
    horizon_len: int,
    warehouse_max_m3: float = 480.0,
    sec45x_multiplier: float = 1.0,
    tariff_multiplier: float = 1.0,
    inventory_week_offset: int = 0,
    mip_rel_gap: float = 2e-3,
    time_limit: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Core HiGHS MILP solver over a given remaining horizon of length `horizon_len`.
    The inventory week offset maps inventory periods to order weeks. It is one
    in rolling mode, where inventory is first projected for the following week.
    """
    t_start = time.perf_counter()

    comp_ids = df_comp["component_id"].tolist()
    n_comp = len(comp_ids)
    comp_idx_map = {c: idx for idx, c in enumerate(comp_ids)}

    lanes = df_lanes.to_dict("records")
    n_lanes = len(lanes)
    H = horizon_len

    n_N = n_lanes * H
    n_Y = n_lanes * H
    n_I = n_comp * H
    n_B = n_comp * H
    total_vars = n_N + n_Y + n_I + n_B

    def idx_N(l: int, tau: int) -> int:
        return l * H + tau

    def idx_Y(l: int, tau: int) -> int:
        return n_N + l * H + tau

    def idx_I(i: int, tau: int) -> int:
        return n_N + n_Y + i * H + tau

    def idx_B(i: int, tau: int) -> int:
        return n_N + n_Y + n_I + i * H + tau

    c_obj = np.zeros(total_vars)
    lb = np.zeros(total_vars)
    ub = np.full(total_vars, np.inf)
    integrality = np.zeros(total_vars, dtype=int)

    for l_idx, lane in enumerate(lanes):
        lot_units = lane["lot_size_units"]
        comp_idx = comp_idx_map[lane["component_id"]]
        # Derived bound uses strictly `demand_future` (forecast demand in rolling mode)
        peak_4wk_demand = float(np.max(demand_future[comp_idx]) * 4.0)

        eff_tariff = lane["sec301_tariff_cost"] * tariff_multiplier
        eff_credit = lane["domestic_credit"] * sec45x_multiplier
        eff_unit_landed = (
            lane["unit_cost"] + eff_tariff + lane["unit_freight_cost"] - eff_credit
        )
        lot_cost = eff_unit_landed * lot_units
        fixed_fee = lane["fixed_order_fee"] + 1e-3 * (l_idx + 1)

        for tau in range(H):
            v_n = idx_N(l_idx, tau)
            v_y = idx_Y(l_idx, tau)
            c_obj[v_n] = lot_cost
            c_obj[v_y] = fixed_fee
            integrality[v_n] = 1
            integrality[v_y] = 1
            ub[v_y] = 1.0

            sup_cap = float(cap_future[(lane["component_id"], lane["supplier_id"], tau)])
            if lane["freight_mode"] == "exp":
                sup_cap *= 1.5
            effective_cap = min(sup_cap, peak_4wk_demand)

            arrives_in_horizon = tau + int(lane["lead_time_wks"]) - inventory_week_offset < H
            max_lots = max(0, math.floor(effective_cap / lot_units)) if arrives_in_horizon else 0
            ub[v_n] = float(max_lots)
            if max_lots < lane["min_lots"]:
                ub[v_y] = 0.0

    for i_idx, comp in df_comp.iterrows():
        for tau in range(H):
            c_obj[idx_I(i_idx, tau)] = comp["holding_cost_wk"]
            c_obj[idx_B(i_idx, tau)] = comp["stockout_penalty_wk"]

    rows = []
    rhs_l = []
    rhs_u = []

    # Constraint 1: Inventory Flow Balance
    lanes_by_comp = {c: [] for c in comp_ids}
    for l_idx, lane in enumerate(lanes):
        lanes_by_comp[lane["component_id"]].append((l_idx, lane))

    for i_idx, comp in df_comp.iterrows():
        cid = comp["component_id"]
        for tau in range(H):
            row = {}
            row[idx_I(i_idx, tau)] = 1.0
            row[idx_B(i_idx, tau)] = -1.0
            if tau > 0:
                row[idx_I(i_idx, tau - 1)] = -1.0
                row[idx_B(i_idx, tau - 1)] = 1.0

            for l_idx, lane in lanes_by_comp[cid]:
                lead = int(lane["lead_time_wks"])
                order_tau = tau + inventory_week_offset - lead
                if order_tau >= 0:
                    row[idx_N(l_idx, order_tau)] = row.get(idx_N(l_idx, order_tau), 0.0) - float(lane["lot_size_units"])

            net_rhs = float(pipe_future[i_idx, tau] - demand_future[i_idx, tau])
            if tau == 0:
                net_rhs += float(init_inv_arr[i_idx] - init_bo_arr[i_idx])

            rows.append(row)
            rhs_l.append(net_rhs)
            rhs_u.append(net_rhs)

    # Constraint 2: Lot-Sizing Linking
    for l_idx, lane in enumerate(lanes):
        min_l = float(lane["min_lots"])
        for tau in range(H):
            v_n = idx_N(l_idx, tau)
            v_y = idx_Y(l_idx, tau)
            max_l = ub[v_n]

            rows.append({v_n: 1.0, v_y: -min_l})
            rhs_l.append(0.0)
            rhs_u.append(np.inf)

            rows.append({v_n: 1.0, v_y: -max_l})
            rhs_l.append(-np.inf)
            rhs_u.append(0.0)

    # Constraint 3: Supplier Weekly Capacity
    lanes_by_comp_sup = {}
    for l_idx, lane in enumerate(lanes):
        key = (lane["component_id"], lane["supplier_id"])
        lanes_by_comp_sup.setdefault(key, []).append((l_idx, lane))

    for (cid, sid), lane_list in lanes_by_comp_sup.items():
        for tau in range(H):
            row = {}
            for l_idx, lane in lane_list:
                row[idx_N(l_idx, tau)] = float(lane["lot_size_units"])
            cap_val = float(cap_future[(cid, sid, tau)])
            has_exp = any(l["freight_mode"] == "exp" for _, l in lane_list)
            effective_cap = cap_val * 1.5 if has_exp else cap_val
            rows.append(row)
            rhs_l.append(-np.inf)
            rhs_u.append(effective_cap)
            if has_exp:
                standard = {idx_N(l_idx, tau): float(lane["lot_size_units"])
                            for l_idx, lane in lane_list if lane["freight_mode"] == "std"}
                rows.append(standard)
                rhs_l.append(-np.inf)
                rhs_u.append(cap_val)

    # Constraint 4: Warehouse Physical Cubic Capacity
    for tau in range(H):
        row = {}
        for i_idx, comp in df_comp.iterrows():
            row[idx_I(i_idx, tau)] = float(comp["unit_volume_m3"])
        rows.append(row)
        rhs_l.append(-np.inf)
        rhs_u.append(warehouse_max_m3)

    A_lil = lil_matrix((len(rows), total_vars), dtype=float)
    for r_idx, r_dict in enumerate(rows):
        for col_idx, val in r_dict.items():
            A_lil[r_idx, col_idx] = val
    A_csr = A_lil.tocsr()

    res = milp(
        c=c_obj,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(A_csr, rhs_l, rhs_u),
        options={"mip_rel_gap": mip_rel_gap, "time_limit": time_limit},
    )
    solve_time = time.perf_counter() - t_start

    if res.x is None:
        raise RuntimeError(f"HiGHS MILP failed to find a feasible solution: {res.message}")

    x = res.x
    N_sol = np.round(x[:n_N]).reshape((n_lanes, H))
    Y_sol = np.round(x[n_N : n_N + n_Y]).reshape((n_lanes, H))
    I_sol = np.maximum(0.0, x[n_N + n_Y : n_N + n_Y + n_I]).reshape((n_comp, H))
    B_sol = np.maximum(0.0, x[n_N + n_Y + n_I :]).reshape((n_comp, H))
    return N_sol, Y_sol, I_sol, B_sol, solve_time, res.status


def evaluate_cost_and_kpis(
    df_comp: pd.DataFrame,
    df_lanes: pd.DataFrame,
    I_hist: np.ndarray,
    B_hist: np.ndarray,
    demand_arr: np.ndarray,
    orders_records: list[dict],
    warehouse_max_m3: float = 480.0,
    sec45x_multiplier: float = 1.0,
    tariff_multiplier: float = 1.0,
    solve_time_sec: float = 0.0,
) -> dict:
    """
    Unified accounting engine ensuring identical cost & KPI evaluation across all policies.
    """
    lanes_lookup = df_lanes.set_index("lane_id").to_dict("index")

    cost_procurement = 0.0
    cost_tariffs = 0.0
    cost_credits = 0.0
    cost_std_freight = 0.0
    cost_exp_freight = 0.0
    cost_setup = 0.0

    for ord_rec in orders_records:
        lane = lanes_lookup[ord_rec["lane_id"]]
        qty = ord_rec["order_qty"]
        cost_procurement += qty * lane["unit_cost"]
        cost_tariffs += qty * (lane["sec301_tariff_cost"] * tariff_multiplier)
        cost_credits += qty * (lane["domestic_credit"] * sec45x_multiplier)
        if lane["freight_mode"] == "std":
            cost_std_freight += qty * lane["unit_freight_cost"]
        else:
            cost_exp_freight += qty * lane["unit_freight_cost"]
        cost_setup += lane["fixed_order_fee"]

    cost_holding = float(np.sum(I_hist * df_comp["holding_cost_wk"].values[:, None]))
    cost_stockout = float(np.sum(B_hist * df_comp["stockout_penalty_wk"].values[:, None]))
    total_net_cost = (
        cost_procurement
        + cost_tariffs
        - cost_credits
        + cost_std_freight
        + cost_exp_freight
        + cost_setup
        + cost_holding
        + cost_stockout
    )

    df_orders = pd.DataFrame(orders_records)
    total_ordered_units = df_orders["order_qty"].sum() if not df_orders.empty else 1.0
    usa_units = (
        df_orders[df_orders["supplier_region"] == "Domestic_USA"]["order_qty"].sum()
        if not df_orders.empty
        else 0.0
    )
    usmca_units = (
        df_orders[df_orders["supplier_region"] == "Nearshore_USMCA"]["order_qty"].sum()
        if not df_orders.empty
        else 0.0
    )
    fill_rate_pct = compute_component_fill_rate(B_hist, demand_arr)
    wh_vol_by_wk = np.sum(I_hist * df_comp["unit_volume_m3"].values[:, None], axis=0)
    peak_wh_util_pct = float(100.0 * np.max(wh_vol_by_wk) / warehouse_max_m3)

    return {
        "solve_time_sec": solve_time_sec,
        "total_net_cost": total_net_cost,
        "cost_before_backlog_penalty": total_net_cost - cost_stockout,
        "ending_backlog_units": float(B_hist[:, -1].sum()),
        "cost_procurement": cost_procurement,
        "cost_tariffs": cost_tariffs,
        "cost_credits": cost_credits,
        "cost_std_freight": cost_std_freight,
        "cost_exp_freight": cost_exp_freight,
        "cost_setup": cost_setup,
        "cost_holding": cost_holding,
        "cost_stockout": cost_stockout,
        "fill_rate_pct": fill_rate_pct,
        "usa_share_pct": 100.0 * usa_units / total_ordered_units,
        "usmca_share_pct": 100.0 * usmca_units / total_ordered_units,
        "peak_wh_util_pct": peak_wh_util_pct,
        "I_sol": I_hist,
        "B_sol": B_hist,
        "demand_arr": demand_arr,
        "df_orders": df_orders,
        "wh_vol_by_wk": wh_vol_by_wk,
    }


def simulate_rolling_horizon_milp(
    df_comp: pd.DataFrame,
    df_lanes: pd.DataFrame,
    df_sched: pd.DataFrame,
    df_pipe: pd.DataFrame,
    warehouse_max_m3: float = 480.0,
    sec45x_multiplier: float = 1.0,
    tariff_multiplier: float = 1.0,
    mip_rel_gap: float = 3e-3,
) -> dict:
    """
    Policy C: Non-Anticipative Rolling-Horizon MILP with End-of-Week Decision Clock.
    At each week t = 0 .. T-1 (ordering through T-2):
      1. Receive scheduled arrivals & apply realized demand D_{i,t}.
      2. Update ending inventory I_{i,t}, backlog B_{i,t}, and observe week t port capacity.
      3. Solve MILP over remaining future horizon [t+1 .. T-1] using:
         - Current state (I_{i,t}, B_{i,t}) and open pipeline arrivals
         - Forecast demand for future weeks (never realized future demand)
         - Recovery forecast for port strikes:
           * If no strike observed in week t: forecast nominal capacity for all future weeks.
           * If capacity is reduced at t, forecast the same reduction at t+1
             and nominal capacity from t+2 onward.
      4. Execute ONLY week t's immediate container/trailer order decisions.
    """
    t_total_start = time.perf_counter()

    comp_ids = df_comp["component_id"].tolist()
    n_comp = len(comp_ids)
    comp_idx_map = {c: idx for idx, c in enumerate(comp_ids)}

    weeks = sorted(df_sched["week"].unique().tolist())
    T = len(weeks)
    t_idx_map = {w: idx for idx, w in enumerate(weeks)}

    lanes = df_lanes.to_dict("records")

    # Pre-index realized demand, forecast demand, and actual weekly capacity
    actual_demand_arr = np.zeros((n_comp, T))
    forecast_demand_arr = np.zeros((n_comp, T))
    for _, row in df_sched[["week", "component_id", "actual_demand", "forecast_demand"]].drop_duplicates().iterrows():
        ci = comp_idx_map[row["component_id"]]
        ti = t_idx_map[row["week"]]
        actual_demand_arr[ci, ti] = row["actual_demand"]
        forecast_demand_arr[ci, ti] = row["forecast_demand"]

    actual_cap_map = {}
    for _, row in df_sched.iterrows():
        actual_cap_map[(row["component_id"], row["supplier_id"], t_idx_map[row["week"]])] = float(row["weekly_supply_cap"])

    # Nominal (un-shocked) supplier weekly capacities
    nominal_cap_by_comp_sup = {}
    unique_sups = df_lanes[["supplier_id", "supplier_region"]].drop_duplicates().to_dict("records")
    for _, comp in df_comp.iterrows():
        cid = comp["component_id"]
        for sup in unique_sups:
            reg = sup["supplier_region"]
            if reg == "Offshore_APAC":
                nom = float(round(45 * comp["units_per_bess"] * 1.50))
            elif reg == "Nearshore_USMCA":
                nom = float(round(45 * comp["units_per_bess"] * 0.85))
            else:
                nom = float(round(45 * comp["units_per_bess"] * 0.75))
            nominal_cap_by_comp_sup[(cid, sup["supplier_id"])] = nom

    # Initialize arrivals schedule with pre-horizon pipeline
    arrivals = np.zeros((n_comp, T + 12))
    for _, row in df_pipe.iterrows():
        w_idx = int(row["arrival_week"]) - 1
        if 0 <= w_idx < T + 12:
            arrivals[comp_idx_map[row["component_id"]], w_idx] += row["scheduled_qty"]

    I_hist = np.zeros((n_comp, T))
    B_hist = np.zeros((n_comp, T))
    inv = df_comp["initial_inventory"].values.astype(float).copy()
    bo = np.zeros(n_comp, dtype=float)
    orders_records = []

    limited_solves = 0
    for t in range(T):
        # Step 1: Receive scheduled deliveries & apply week t realized demand
        inv += arrivals[:, t]
        for i_idx in range(n_comp):
            total_req = actual_demand_arr[i_idx, t] + bo[i_idx]
            if inv[i_idx] >= total_req:
                inv[i_idx] -= total_req
                bo[i_idx] = 0.0
            else:
                bo[i_idx] = total_req - inv[i_idx]
                inv[i_idx] = 0.0

        # Step 2: Record end-of-week t inventory/backlog & observe week t port capacity
        I_hist[:, t] = inv
        B_hist[:, t] = bo

        if t == T - 1:
            break

        # Order index zero is week t. Inventory index zero is week t+1.
        H_rem = T - t - 1
        demand_rem = forecast_demand_arr[:, t + 1 : T]
        pipe_rem = arrivals[:, t + 1 : T]

        # Build non-anticipative capacity forecast using explicit 2-week recovery rule:
        # tau=0 is current ordering epoch (end of week t, governed by observed week t capacity).
        # If week t experienced an offshore port strike (observed_cap < nominal), forecast
        # reduced capacity at tau=1 (week t+1) and full nominal recovery at tau >= 2.
        cap_rem = {}
        for _, comp in df_comp.iterrows():
            cid = comp["component_id"]
            for sup in unique_sups:
                sid = sup["supplier_id"]
                nom_cap = nominal_cap_by_comp_sup[(cid, sid)]
                obs_cap_t = actual_cap_map[(cid, sid, t)]
                strike_active_t = obs_cap_t < (0.8 * nom_cap)

                for tau in range(H_rem):
                    if tau == 0:
                        cap_rem[(cid, sid, tau)] = obs_cap_t
                    elif tau == 1 and strike_active_t:
                        cap_rem[(cid, sid, tau)] = obs_cap_t  # 2-week recovery belief
                    else:
                        cap_rem[(cid, sid, tau)] = nom_cap

        N_sub, _, _, _, _, status = solve_milp_subproblem(
            df_comp=df_comp,
            df_lanes=df_lanes,
            init_inv_arr=inv,
            init_bo_arr=bo,
            pipe_future=pipe_rem,
            demand_future=demand_rem,
            cap_future=cap_rem,
            horizon_len=H_rem,
            warehouse_max_m3=warehouse_max_m3,
            sec45x_multiplier=sec45x_multiplier,
            tariff_multiplier=tariff_multiplier,
            inventory_week_offset=1,
            mip_rel_gap=mip_rel_gap,
            time_limit=4.0,
        )
        limited_solves += int(status != 0)

        # Step 4: Execute ONLY tau = 0 (end-of-week t) order decisions
        for l_idx, lane in enumerate(lanes):
            lots_t = int(N_sub[l_idx, 0])
            if lots_t > 0:
                qty = lots_t * int(lane["lot_size_units"])
                arr_idx = t + int(lane["lead_time_wks"])
                if arr_idx < T + 12:
                    arrivals[comp_idx_map[lane["component_id"]], arr_idx] += qty

                orders_records.append(
                    {
                        "week": t + 1,
                        "lane_id": lane["lane_id"],
                        "component_id": lane["component_id"],
                        "supplier_id": lane["supplier_id"],
                        "supplier_region": lane["supplier_region"],
                        "freight_mode": lane["freight_mode"],
                        "equipment_type": lane["equipment_type"],
                        "lots": lots_t,
                        "order_qty": qty,
                    }
                )

    total_time = time.perf_counter() - t_total_start
    outcome = evaluate_cost_and_kpis(
        df_comp=df_comp,
        df_lanes=df_lanes,
        I_hist=I_hist,
        B_hist=B_hist,
        demand_arr=actual_demand_arr,
        orders_records=orders_records,
        warehouse_max_m3=warehouse_max_m3,
        sec45x_multiplier=sec45x_multiplier,
        tariff_multiplier=tariff_multiplier,
        solve_time_sec=total_time,
    )
    outcome["limits_reached"] = limited_solves
    return outcome


def solve_perfect_foresight_milp(
    df_comp: pd.DataFrame,
    df_lanes: pd.DataFrame,
    df_sched: pd.DataFrame,
    df_pipe: pd.DataFrame,
    warehouse_max_m3: float = 480.0,
    sec45x_multiplier: float = 1.0,
    tariff_multiplier: float = 1.0,
    mip_rel_gap: float = 2e-3,
) -> dict:
    """
    Policy D: Perfect-Foresight Reference MILP (Clairvoyant Benchmark).
    """
    comp_ids = df_comp["component_id"].tolist()
    n_comp = len(comp_ids)
    comp_idx_map = {c: idx for idx, c in enumerate(comp_ids)}
    weeks = sorted(df_sched["week"].unique().tolist())
    T = len(weeks)
    t_idx_map = {w: idx for idx, w in enumerate(weeks)}

    demand_arr = np.zeros((n_comp, T))
    for _, row in df_sched[["week", "component_id", "actual_demand"]].drop_duplicates().iterrows():
        demand_arr[comp_idx_map[row["component_id"]], t_idx_map[row["week"]]] = row["actual_demand"]

    cap_map = {}
    for _, row in df_sched.iterrows():
        cap_map[(row["component_id"], row["supplier_id"], t_idx_map[row["week"]])] = float(row["weekly_supply_cap"])

    pipe_arr = np.zeros((n_comp, T))
    for _, row in df_pipe.iterrows():
        w = int(row["arrival_week"])
        if w in t_idx_map:
            pipe_arr[comp_idx_map[row["component_id"]], t_idx_map[w]] += row["scheduled_qty"]

    N_sol, _, I_sol, B_sol, solve_time, _ = solve_milp_subproblem(
        df_comp=df_comp,
        df_lanes=df_lanes,
        init_inv_arr=df_comp["initial_inventory"].values.astype(float),
        init_bo_arr=np.zeros(n_comp),
        pipe_future=pipe_arr,
        demand_future=demand_arr,
        cap_future=cap_map,
        horizon_len=T,
        warehouse_max_m3=warehouse_max_m3,
        sec45x_multiplier=sec45x_multiplier,
        tariff_multiplier=tariff_multiplier,
        mip_rel_gap=mip_rel_gap,
        time_limit=15.0,
    )

    lanes = df_lanes.to_dict("records")
    orders_records = []
    for l_idx, lane in enumerate(lanes):
        lot_u = int(lane["lot_size_units"])
        for t in range(T):
            lots = int(N_sol[l_idx, t])
            if lots > 0:
                orders_records.append(
                    {
                        "week": t + 1,
                        "lane_id": lane["lane_id"],
                        "component_id": lane["component_id"],
                        "supplier_id": lane["supplier_id"],
                        "supplier_region": lane["supplier_region"],
                        "freight_mode": lane["freight_mode"],
                        "equipment_type": lane["equipment_type"],
                        "lots": lots,
                        "order_qty": lots * lot_u,
                    }
                )

    return evaluate_cost_and_kpis(
        df_comp=df_comp,
        df_lanes=df_lanes,
        I_hist=I_sol,
        B_hist=B_sol,
        demand_arr=demand_arr,
        orders_records=orders_records,
        warehouse_max_m3=warehouse_max_m3,
        sec45x_multiplier=sec45x_multiplier,
        tariff_multiplier=tariff_multiplier,
        solve_time_sec=solve_time,
    )


def simulate_heuristic_policy(
    df_comp: pd.DataFrame,
    df_lanes: pd.DataFrame,
    df_sched: pd.DataFrame,
    df_pipe: pd.DataFrame,
    mode: str = "static_ocean",
    warehouse_max_m3: float = 480.0,
) -> dict:
    """
    Simulates traditional ERP (s, S) Min-Max Reorder Policies on the exact same
    end-of-week decision clock as Policy C:
      - mode="static_ocean" : Policy A (Single-Source Offshore Ocean s, S)
      - mode="reactive_air" : Policy B1 (Single-Source Offshore Ocean + Reactive Air s, S)
      - mode="dual_sourced" : Policy B2 (Dual-Sourced Heuristic with access to U.S. Domestic,
                              USMCA Nearshore, and Offshore APAC lanes)
    """
    comp_ids = df_comp["component_id"].tolist()
    n_comp = len(comp_ids)
    weeks = sorted(df_sched["week"].unique().tolist())
    T = len(weeks)

    I_hist = np.zeros((n_comp, T))
    B_hist = np.zeros((n_comp, T))
    demand_arr = np.zeros((n_comp, T))
    orders_records = []

    for i_idx, comp in df_comp.iterrows():
        cid = comp["component_id"]
        sub_apac = df_sched[(df_sched["component_id"] == cid) & (df_sched["supplier_id"] == "SUP_APAC_1")].sort_values("week")
        sub_usa = df_sched[(df_sched["component_id"] == cid) & (df_sched["supplier_id"] == "SUP_USA_1")].sort_values("week")
        sub_mex = df_sched[(df_sched["component_id"] == cid) & (df_sched["supplier_id"] == "SUP_MEX_1")].sort_values("week")

        actual_d = sub_apac["actual_demand"].values
        forecast_d = sub_apac["forecast_demand"].values
        apac_cap = sub_apac["weekly_supply_cap"].values
        usa_cap = sub_usa["weekly_supply_cap"].values
        mex_cap = sub_mex["weekly_supply_cap"].values
        demand_arr[i_idx, :] = actual_d

        std_apac = df_lanes[
            (df_lanes["component_id"] == cid)
            & (df_lanes["supplier_id"] == "SUP_APAC_1")
            & (df_lanes["equipment_type"] == "20ft_Dry_TriAxle")
        ].iloc[0]
        air_lanes = df_lanes[
            (df_lanes["component_id"] == cid)
            & (df_lanes["supplier_id"] == "SUP_APAC_1")
            & (df_lanes["equipment_type"] == "Air_PMC_Pallet")
        ]
        air_lane = None if air_lanes.empty else air_lanes.iloc[0]
        usa_box = df_lanes[
            (df_lanes["component_id"] == cid)
            & (df_lanes["supplier_id"] == "SUP_USA_1")
            & (df_lanes["equipment_type"] == "26ft_Box_Truck")
        ].iloc[0]
        mex_ftl = df_lanes[
            (df_lanes["component_id"] == cid)
            & (df_lanes["supplier_id"] == "SUP_MEX_1")
            & (df_lanes["equipment_type"] == "53ft_FTL_DryVan")
        ].iloc[0]

        mu_d = float(forecast_d[0])
        L_apac = int(std_apac["lead_time_wks"])
        s_reorder = mu_d * L_apac + 1.35 * mu_d
        S_order_up_to = s_reorder + 2.2 * mu_d

        arrivals = np.zeros(T + 12)
        for _, prow in df_pipe[df_pipe["component_id"] == cid].iterrows():
            w_idx = int(prow["arrival_week"]) - 1
            if 0 <= w_idx < T + 12:
                arrivals[w_idx] += prow["scheduled_qty"]

        inv = float(comp["initial_inventory"])
        backorder = 0.0

        for t in range(T):
            # Step 1: Receive arrivals & apply realized demand
            inv += arrivals[t]
            total_req = actual_d[t] + backorder
            if inv >= total_req:
                inv -= total_req
                backorder = 0.0
            else:
                backorder = total_req - inv
                inv = 0.0

            # Step 2: Record state & observe week t capacity
            I_hist[i_idx, t] = inv
            B_hist[i_idx, t] = backorder

            if t == T - 1:
                break
            next_f_d = float(forecast_d[t + 1])
            air_qty = 0
            open_pipeline = float(np.sum(arrivals[t + 1 : t + 1 + L_apac]))
            inv_position = inv - backorder + open_pipeline

            # Step 3: Place end-of-week t orders according to heuristic policy
            if mode == "dual_sourced":
                # Dual-sourced heuristic prioritizes U.S. Domestic when §45X makes it
                # unit-cost competitive or when inventory cover is tight, then USMCA, then APAC
                usa_advantage = usa_box["net_landed_unit_cost"] <= std_apac["net_landed_unit_cost"]
                short_term_cover = (inv - backorder + arrivals[t + 1]) < 1.6 * next_f_d

                if usa_advantage or short_term_cover:
                    target_usa = max(0.0, 2.2 * next_f_d - (inv - backorder + arrivals[t + 1]))
                    lot_u = int(usa_box["lot_size_units"])
                    n_lots = min(math.ceil(target_usa / lot_u), math.floor(usa_cap[t] / lot_u))
                    if n_lots >= 1 and t + int(usa_box["lead_time_wks"]) < T:
                        qty = n_lots * lot_u
                        arrivals[t + int(usa_box["lead_time_wks"])] += qty
                        inv_position += qty
                        orders_records.append(
                            {
                                "week": t + 1,
                                "lane_id": usa_box["lane_id"],
                                "component_id": cid,
                                "supplier_id": "SUP_USA_1",
                                "supplier_region": "Domestic_USA",
                                "freight_mode": "std",
                                "equipment_type": "26ft_Box_Truck",
                                "lots": n_lots,
                                "order_qty": qty,
                            }
                        )

                # If offshore port is disrupted or inv_position still below s, tap USMCA nearshore
                if inv_position <= s_reorder and apac_cap[t] < 0.8 * (45 * comp["units_per_bess"] * 1.5):
                    gap_mex = max(0.0, s_reorder - inv_position)
                    lot_m = int(mex_ftl["lot_size_units"])
                    n_lots_m = min(math.ceil(gap_mex / lot_m), math.floor(mex_cap[t] / lot_m))
                    if n_lots_m >= 1 and t + int(mex_ftl["lead_time_wks"]) < T:
                        qty_m = n_lots_m * lot_m
                        arrivals[t + int(mex_ftl["lead_time_wks"])] += qty_m
                        inv_position += qty_m
                        orders_records.append(
                            {
                                "week": t + 1,
                                "lane_id": mex_ftl["lane_id"],
                                "component_id": cid,
                                "supplier_id": "SUP_MEX_1",
                                "supplier_region": "Nearshore_USMCA",
                                "freight_mode": "std",
                                "equipment_type": "53ft_FTL_DryVan",
                                "lots": n_lots_m,
                                "order_qty": qty_m,
                            }
                        )

            elif mode == "reactive_air" and air_lane is not None:
                air_lead = int(air_lane["lead_time_wks"])
                if t + air_lead < T and inv - backorder < 1.1 * next_f_d:
                    gap_air = max(0.0, 1.6 * next_f_d - (inv - backorder))
                    air_lot = int(air_lane["lot_size_units"])
                    n_air_lots = min(math.ceil(gap_air / air_lot), math.floor(1.5 * apac_cap[t] / air_lot))
                    air_qty = n_air_lots * air_lot
                    if air_qty:
                        arrivals[t + air_lead] += air_qty
                        inv_position += air_qty
                        orders_records.append(
                            {
                                "week": t + 1,
                                "lane_id": air_lane["lane_id"],
                                "component_id": cid,
                                "supplier_id": "SUP_APAC_1",
                                "supplier_region": "Offshore_APAC",
                                "freight_mode": "exp",
                                "equipment_type": "Air_PMC_Pallet",
                                "lots": n_air_lots,
                                "order_qty": air_qty,
                            }
                        )

            # Standard Ocean (s, S) Replenishment
            if inv_position <= s_reorder and t + L_apac < T:
                target_qty = max(0.0, S_order_up_to - inv_position)
                ocean_lot = int(std_apac["lot_size_units"])
                n_ocean_lots = max(1, math.ceil(target_qty / ocean_lot))
                remaining_cap = min(apac_cap[t], 1.5 * apac_cap[t] - air_qty)
                max_cap_lots = math.floor(remaining_cap / ocean_lot)
                n_ocean_lots = min(n_ocean_lots, max_cap_lots)

                if n_ocean_lots >= 1:
                    ocean_qty = n_ocean_lots * ocean_lot
                    arrivals[t + L_apac] += ocean_qty
                    orders_records.append(
                        {
                            "week": t + 1,
                            "lane_id": std_apac["lane_id"],
                            "component_id": cid,
                            "supplier_id": "SUP_APAC_1",
                            "supplier_region": "Offshore_APAC",
                            "freight_mode": "std",
                            "equipment_type": "20ft_Dry_TriAxle",
                            "lots": n_ocean_lots,
                            "order_qty": ocean_qty,
                        }
                    )

    return evaluate_cost_and_kpis(
        df_comp=df_comp,
        df_lanes=df_lanes,
        I_hist=I_hist,
        B_hist=B_hist,
        demand_arr=demand_arr,
        orders_records=orders_records,
        warehouse_max_m3=warehouse_max_m3,
    )


def verify_non_anticipativity(
    df_comp: pd.DataFrame,
    df_lanes: pd.DataFrame,
    df_sched: pd.DataFrame,
    df_pipe: pd.DataFrame,
    split_week: int = 10,
) -> None:
    """
    Verification test: Two scenarios with identical observations through `split_week`
    and divergent port-strike / demand realizations after `split_week` must produce
    100% identical Rolling-Horizon MILP orders through `split_week`.
    """
    df_alt = df_sched.copy()
    mask_after = df_alt["week"] > split_week
    df_alt.loc[mask_after, "actual_demand"] = np.round(df_alt.loc[mask_after, "actual_demand"] * 1.25).astype(int)
    mask_apac_after = mask_after & (df_alt["supplier_region"] == "Offshore_APAC")
    df_alt.loc[mask_apac_after, "weekly_supply_cap"] = 0

    res_1 = simulate_rolling_horizon_milp(df_comp, df_lanes, df_sched, df_pipe, mip_rel_gap=3e-3)
    res_2 = simulate_rolling_horizon_milp(df_comp, df_lanes, df_alt, df_pipe, mip_rel_gap=3e-3)

    ord_1 = res_1["df_orders"][res_1["df_orders"]["week"] <= split_week].reset_index(drop=True)
    ord_2 = res_2["df_orders"][res_2["df_orders"]["week"] <= split_week].reset_index(drop=True)
    assert ord_1.equals(ord_2), "Non-anticipativity check failed: orders differed before split_week!"
    print(f"[VERIFIED] Non-Anticipativity Test Passed: Identical Rolling-Horizon MILP orders through Week {split_week}.")


def evaluate_stress_scenario(scenario, df_comp, df_lanes, df_sched, df_pipe) -> dict:
    trial, shock_start, shock_dur, demand_mult = scenario
    shock_end = shock_start + shock_dur - 1
    df_sched_mc = df_sched.copy()
    for _, comp in df_comp.iterrows():
        cid = comp["component_id"]
        base_apac_cap = int(round(45 * comp["units_per_bess"] * 1.50))
        mask_apac_comp = (df_sched_mc["component_id"] == cid) & (df_sched_mc["supplier_region"] == "Offshore_APAC")
        df_sched_mc.loc[mask_apac_comp, "weekly_supply_cap"] = base_apac_cap

    units_per_bess = df_sched_mc["component_id"].map(
        df_comp.set_index("component_id")["units_per_bess"]
    )
    for t in range(1, 27):
        mask_t = df_sched_mc["week"] == t
        forecast_builds = df_sched_mc.loc[mask_t, "forecast_demand"] / units_per_bess[mask_t]
        actual_builds = np.maximum(1, np.round(forecast_builds * demand_mult[t - 1]))
        df_sched_mc.loc[mask_t, "actual_demand"] = (actual_builds * units_per_bess[mask_t]).astype(int)

        mask_apac = mask_t & (df_sched_mc["supplier_region"] == "Offshore_APAC")
        if shock_start <= t <= shock_end:
            df_sched_mc.loc[mask_apac, "weekly_supply_cap"] = np.round(
                df_sched_mc.loc[mask_apac, "weekly_supply_cap"] * 0.15
            ).astype(int)

    res_static = simulate_heuristic_policy(df_comp, df_lanes, df_sched_mc, df_pipe, mode="static_ocean")
    res_air = simulate_heuristic_policy(df_comp, df_lanes, df_sched_mc, df_pipe, mode="reactive_air")
    res_dual_h = simulate_heuristic_policy(df_comp, df_lanes, df_sched_mc, df_pipe, mode="dual_sourced")
    res_rh_milp = simulate_rolling_horizon_milp(df_comp, df_lanes, df_sched_mc, df_pipe, mip_rel_gap=5e-3)

    return {
        "trial": trial,
        "shock_start_wk": shock_start,
        "shock_duration_wks": shock_dur,
        "static_ss_cost": res_static["total_net_cost"],
        "reactive_air_cost": res_air["total_net_cost"],
        "dual_heuristic_cost": res_dual_h["total_net_cost"],
        "rolling_milp_cost": res_rh_milp["total_net_cost"],
        "savings_vs_static_pct": 100.0 * (res_static["total_net_cost"] - res_rh_milp["total_net_cost"]) / res_static["total_net_cost"],
        "savings_vs_air_pct": 100.0 * (res_air["total_net_cost"] - res_rh_milp["total_net_cost"]) / res_air["total_net_cost"],
        "savings_vs_dual_heur_pct": 100.0 * (res_dual_h["total_net_cost"] - res_rh_milp["total_net_cost"]) / res_dual_h["total_net_cost"],
        "rolling_milp_fill_rate": res_rh_milp["fill_rate_pct"],
        "rolling_milp_usa_share_pct": res_rh_milp["usa_share_pct"],
        "rolling_milp_peak_warehouse_pct": res_rh_milp["peak_wh_util_pct"],
        "rolling_solver_limits": res_rh_milp["limits_reached"],
    }


def run_monte_carlo_stress_test(
    df_comp: pd.DataFrame,
    df_lanes: pd.DataFrame,
    df_sched: pd.DataFrame,
    df_pipe: pd.DataFrame,
    paths_per_strike: int = 3,
    seed: int = 42,
    workers: int = 1,
) -> pd.DataFrame:
    """Cross all 44 strike timings and durations with independent demand paths."""
    rng = np.random.default_rng(seed)
    strike_cases = [(start, duration) for start in range(6, 17) for duration in range(2, 6)]
    scenarios = [
        (trial, start, duration, rng.normal(1.0, 0.08, size=26))
        for trial, (start, duration) in enumerate(strike_cases * paths_per_strike, start=1)
    ]
    evaluate = partial(evaluate_stress_scenario, df_comp=df_comp, df_lanes=df_lanes,
                       df_sched=df_sched, df_pipe=df_pipe)
    if workers == 1:
        return pd.DataFrame(map(evaluate, scenarios))
    records = []
    # Start clean interpreters so workers do not inherit the native HiGHS scheduler.
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        for record in pool.map(evaluate, scenarios):
            records.append(record)
            if len(records) % 12 == 0:
                print(f"  Completed {len(records)} of {len(scenarios)} scenarios", flush=True)
    return pd.DataFrame(records)



def generate_figures(
    res_static: dict,
    res_air: dict,
    res_dual_h: dict,
    res_rh_milp: dict,
    res_pf_milp: dict,
    df_mc: pd.DataFrame,
    df_comp: pd.DataFrame,
    df_lanes: pd.DataFrame,
    df_sched: pd.DataFrame,
    df_pipe: pd.DataFrame,
    figures_dir: str = "figures",
) -> None:
    os.makedirs(figures_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # FIGURE 1: 5-Policy Landed Cost Decomposition & Line Fill Rate
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.8), gridspec_kw={"width_ratios": [1.65, 1]})

    policies = [
        "A: Static\n(s, S) Ocean",
        "B1: Reactive\nAir (s, S)",
        "B2: Dual-Src\nHeuristic",
        "C: Rolling\nMILP (Online)",
        "D: Perfect-\nForesight Ref.",
    ]
    results_list = [res_static, res_air, res_dual_h, res_rh_milp, res_pf_milp]

    net_proc = [
        (r["cost_procurement"] + r["cost_tariffs"] - r["cost_credits"]) / 1e6
        for r in results_list
    ]
    std_frt = [(r["cost_std_freight"] + r["cost_setup"]) / 1e6 for r in results_list]
    exp_frt = [r["cost_exp_freight"] / 1e6 for r in results_list]
    holding = [r["cost_holding"] / 1e6 for r in results_list]
    stockout = [r["cost_stockout"] / 1e6 for r in results_list]

    x = np.arange(len(policies))
    w = 0.54

    ax1.bar(x, net_proc, w, label="Net procurement and modeled import costs", color="#2B5B84")
    ax1.bar(x, std_frt, w, bottom=net_proc, label="Standard Ocean/Ground Freight & Setup", color="#5C9EAD")
    b2 = np.array(net_proc) + np.array(std_frt)
    ax1.bar(x, exp_frt, w, bottom=b2, label="Emergency Air Expedite Freight", color="#E07A5F")
    b3 = b2 + np.array(exp_frt)
    ax1.bar(x, holding, w, bottom=b3, label="Inventory Holding Cost", color="#81B29A")
    b4 = b3 + np.array(holding)
    ax1.bar(x, stockout, w, bottom=b4, label="Component backlog penalties", color="#D62828")

    totals = b4 + np.array(stockout)
    for idx, tot in enumerate(totals):
        ax1.text(idx, tot + 0.8, f"${tot:.2f}M", ha="center", va="bottom", fontweight="bold", fontsize=9.2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, fontsize=9.0, fontweight="bold")
    ax1.set_ylabel("Total 26-Week Supply-Chain Cost ($ Millions)", fontsize=10.5, fontweight="bold")
    ax1.set_title("A. Modeled Cost and Backlog Penalties", fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_ylim(0, max(totals) * 1.18)
    ax1.grid(axis="y")
    ax1.legend(loc="upper right", fontsize=8.0, frameon=True)

    fill_rates = [r["fill_rate_pct"] for r in results_list]
    colors_fr = ["#D62828", "#E07A5F", "#457B9D", "#2A9D8F", "#1D3557"]
    ax2.bar(x, fill_rates, width=0.50, color=colors_fr)
    for idx, fr in enumerate(fill_rates):
        ax2.text(idx, fr + 1.2, f"{fr:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9.2)

    ax2.set_xticks(x)
    ax2.set_xticklabels(["A:\nStatic", "B1:\nAir", "B2:\nDual-H", "C:\nRH-MILP", "D:\nPF-Ref"], fontsize=9.0, fontweight="bold")
    ax2.set_ylabel("On-Time Component Fill Rate (%)", fontsize=10.5, fontweight="bold")
    ax2.set_title("B. Component Service Level", fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_ylim(0, 112)
    ax2.grid(axis="y")

    fig.suptitle(
        "U.S. Clean-Tech Supply-Chain Benchmark: Non-Anticipative Rolling-Horizon MILP vs. Heuristics",
        fontsize=12.6,
        fontweight="bold",
        y=0.97,
    )
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.06, right=0.98, wspace=0.24)
    plt.savefig(os.path.join(figures_dir, "fig1_cost_breakdown_benchmark.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 2: Inventory Trajectory & Stockout Defense (LFP Battery Modules)
    # -------------------------------------------------------------------------
    weeks = np.arange(1, 27)
    lfp_idx = 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)

    ax1.axvspan(11, 14.9, color="#F4A261", alpha=0.22, label="Offshore Port Bottleneck (Wks 11–14)")
    ax1.axvspan(14, 19.9, color="#457B9D", alpha=0.14, label="Q2/Q3 Grid Demand Surge (Wks 14–19)")

    ax1.plot(weeks, res_static["I_sol"][lfp_idx], "o--", color="#D62828", lw=1.8, markersize=4, label="A: Static (s, S) Ocean")
    ax1.plot(weeks, res_dual_h["I_sol"][lfp_idx], "^-.", color="#457B9D", lw=1.8, markersize=4, label="B2: Dual-Sourced Heuristic")
    ax1.plot(weeks, res_rh_milp["I_sol"][lfp_idx], "D-", color="#2A9D8F", lw=2.4, markersize=5, label="C: Rolling-Horizon MILP (Online)")
    ax1.plot(weeks, res_rh_milp["demand_arr"][lfp_idx], ":", color="#333333", lw=1.8, label="Weekly module demand")

    ax1.set_ylabel("LFP Battery Modules (Units)", fontsize=10.5, fontweight="bold")
    ax1.set_title("A. On-Hand Inventory Buffer Trajectory (25 kWh LFP Battery Modules)", fontsize=11.5, fontweight="bold")
    ax1.grid(True)
    ax1.legend(loc="upper left", ncol=2, fontsize=8.5, frameon=True)

    ax2.axvspan(11, 14.9, color="#F4A261", alpha=0.22)
    ax2.axvspan(14, 19.9, color="#457B9D", alpha=0.14)

    ax2.fill_between(weeks, res_static["B_sol"][lfp_idx], color="#D62828", alpha=0.30, label="A: Static (s, S) Shortage")
    ax2.plot(weeks, res_static["B_sol"][lfp_idx], color="#D62828", lw=2)
    ax2.fill_between(weeks, res_dual_h["B_sol"][lfp_idx], color="#457B9D", alpha=0.30, label="B2: Dual-Sourced Heuristic Shortage")
    ax2.plot(weeks, res_dual_h["B_sol"][lfp_idx], color="#457B9D", lw=2)
    ax2.plot(weeks, res_rh_milp["B_sol"][lfp_idx], color="#2A9D8F", lw=2.8, label="C: Rolling-Horizon MILP Shortage")

    ax2.set_xlabel("Planning Horizon (Week)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Outstanding Backorders (Units)", fontsize=10.5, fontweight="bold")
    ax2.set_title("B. LFP Module Backlog During Port and Demand Disruptions", fontsize=11.5, fontweight="bold")
    ax2.set_xticks(weeks)
    ax2.grid(True)
    ax2.legend(loc="upper left", fontsize=9, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig2_inventory_disruption_dynamics.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 3: Dynamic Regional Sourcing Mix & Non-Anticipative Monte Carlo
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.3, 1]})

    df_ord = res_rh_milp["df_orders"]
    piv = (
        df_ord.pivot_table(index="week", columns="supplier_region", values="order_qty", aggfunc="sum")
        .reindex(weeks, fill_value=0)
    )
    for col in ["Offshore_APAC", "Nearshore_USMCA", "Domestic_USA"]:
        if col not in piv.columns:
            piv[col] = 0.0

    row_sums = piv.sum(axis=1).replace(0, 1)
    pct_apac = 100.0 * piv["Offshore_APAC"] / row_sums
    pct_mex = 100.0 * piv["Nearshore_USMCA"] / row_sums
    pct_usa = 100.0 * piv["Domestic_USA"] / row_sums

    ax1.bar(weeks, pct_usa, label="U.S. domestic", color="#1D3557")
    ax1.bar(weeks, pct_mex, bottom=pct_usa, label="Mexico nearshore", color="#457B9D")
    ax1.bar(weeks, pct_apac, bottom=pct_usa + pct_mex, label="Offshore APAC", color="#A8DADC")
    ax1.axvspan(11, 14.9, color="#E63946", alpha=0.18, label="Port Disruption Window")

    ax1.set_xlabel("Order Placement Week", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Share of Weekly Ordered Units (%)", fontsize=10.5, fontweight="bold")
    ax1.set_title("A. Rolling-Horizon MILP Regional Sourcing Reallocation", fontsize=11.5, fontweight="bold")
    ax1.set_ylim(0, 108)
    ax1.grid(axis="y")
    ax1.legend(loc="lower right", fontsize=8.0, frameon=True)

    bp_data = [
        df_mc["savings_vs_static_pct"].values,
        df_mc["savings_vs_air_pct"].values,
        df_mc["savings_vs_dual_heur_pct"].values,
    ]
    bp = ax2.boxplot(
        bp_data,
        tick_labels=["vs. Static\n(s, S) Ocean", "vs. Reactive\nAir (s, S)", "vs. Dual-Src\nHeuristic"],
        patch_artist=True,
        widths=0.45,
        medianprops={"color": "#111111", "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], ["#2A9D8F", "#457B9D", "#E9C46A"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)

    ax2.set_ylabel("26-Week Modeled Total Cost Reduction (%)", fontsize=10.5, fontweight="bold")
    ax2.set_title(f"B. Non-Anticipative Monte Carlo Savings (N={len(df_mc)})", fontsize=11.5, fontweight="bold")
    ax2.grid(axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig3_sourcing_mix_and_montecarlo.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 4: Policy Sensitivity Frontier (IRC §45X Credit & Section 301 Tariffs)
    # -------------------------------------------------------------------------
    credit_scales = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25]
    usa_shares_base = []
    usa_shares_high_tariff = []
    total_costs_base = []

    for cs in credit_scales:
        r_base = solve_perfect_foresight_milp(
            df_comp, df_lanes, df_sched, df_pipe, sec45x_multiplier=cs, tariff_multiplier=1.0, mip_rel_gap=3e-3
        )
        r_high = solve_perfect_foresight_milp(
            df_comp, df_lanes, df_sched, df_pipe, sec45x_multiplier=cs, tariff_multiplier=1.4, mip_rel_gap=3e-3
        )
        usa_shares_base.append(r_base["usa_share_pct"])
        usa_shares_high_tariff.append(r_high["usa_share_pct"])
        total_costs_base.append(r_base["total_net_cost"] / 1e6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.0))

    pct_labels = [int(c * 100) for c in credit_scales]
    ax1.plot(pct_labels, usa_shares_base, "o-", color="#1D3557", lw=2.4, label="Baseline tariff assumptions (25%–50%)")
    ax1.plot(pct_labels, usa_shares_high_tariff, "s--", color="#E63946", lw=2.2, label="Elevated Tariff Regime (+40% Escalation)")
    ax1.axvline(100, color="#2A9D8F", linestyle=":", lw=1.8, label="Baseline module discount (100%)")

    ax1.set_xlabel("Domestic module discount (% of baseline)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("U.S. Domestic Sourcing Share (%)", fontsize=10, fontweight="bold")
    ax1.set_title("A. Perfect-Foresight Domestic Sourcing Share", fontsize=11.5, fontweight="bold")
    ax1.grid(True)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)

    ax2.plot(pct_labels, total_costs_base, "D-", color="#2B5B84", lw=2.4)
    for x_v, y_v in zip(pct_labels, total_costs_base):
        ax2.annotate(f"${y_v:.2f}M", (x_v, y_v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold")

    ax2.set_xlabel("Domestic module discount (% of baseline)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("26-Week Modeled Total Cost ($ Millions)", fontsize=10, fontweight="bold")
    ax2.set_title("B. Perfect-Foresight Cost Sensitivity", fontsize=11.5, fontweight="bold")
    ax2.set_ylim(min(total_costs_base) * 0.94, max(total_costs_base) * 1.05)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_policy_sensitivity_frontier.png"), dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-per-strike", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.paths_per_strike < 1 or args.workers < 1:
        parser.error("paths per strike and workers must be positive")
    os.makedirs("results", exist_ok=True)
    df_comp, df_lanes, df_sched, df_pipe = load_data("data")

    print("0. Verifying Strict Non-Anticipativity of Rolling-Horizon MILP...")
    verify_non_anticipativity(df_comp, df_lanes, df_sched, df_pipe, split_week=10)

    print("1. Simulating Policy A: Traditional Static Single-Source (s, S) Ocean...")
    res_static = simulate_heuristic_policy(df_comp, df_lanes, df_sched, df_pipe, mode="static_ocean")

    print("2. Simulating Policy B1: Reactive Air-Expediting (s, S)...")
    res_air = simulate_heuristic_policy(df_comp, df_lanes, df_sched, df_pipe, mode="reactive_air")

    print("3. Simulating Policy B2: Dual-Sourced Heuristic (s, S) (U.S. + USMCA + APAC)...")
    res_dual_h = simulate_heuristic_policy(df_comp, df_lanes, df_sched, df_pipe, mode="dual_sourced")

    print("4. Simulating Policy C: Non-Anticipative Rolling-Horizon MILP (Primary Operational Benchmark)...")
    res_rh_milp = simulate_rolling_horizon_milp(df_comp, df_lanes, df_sched, df_pipe, mip_rel_gap=3e-3)

    print("5. Solving Policy D: Perfect-Foresight Reference MILP...")
    res_pf_milp = solve_perfect_foresight_milp(df_comp, df_lanes, df_sched, df_pipe, mip_rel_gap=5e-5)

    summary_rows = []
    for name, r in [
        ("A: Static Single-Source (s, S) Ocean", res_static),
        ("B1: Reactive Air-Expediting (s, S)", res_air),
        ("B2: Dual-Sourced Heuristic (s, S)", res_dual_h),
        ("C: Rolling-Horizon MILP (Non-Anticipative)", res_rh_milp),
        ("D: Perfect-Foresight Reference MILP", res_pf_milp),
    ]:
        summary_rows.append(
            {
                "Policy": name,
                "Total_Net_Cost_USD": round(r["total_net_cost"], 2),
                "Net_Procurement_Tariff_USD": round(r["cost_procurement"] + r["cost_tariffs"] - r["cost_credits"], 2),
                "Std_Freight_Setup_USD": round(r["cost_std_freight"] + r["cost_setup"], 2),
                "Emergency_Air_Freight_USD": round(r["cost_exp_freight"], 2),
                "Holding_Cost_USD": round(r["cost_holding"], 2),
                "Stockout_Penalty_USD": round(r["cost_stockout"], 2),
                "Component_Fill_Rate_Pct": round(r["fill_rate_pct"], 2),
                "Cost_Before_Backlog_Penalty_USD": round(r["cost_before_backlog_penalty"], 2),
                "Ending_Backlog_Units": round(r["ending_backlog_units"], 2),
                "USA_Domestic_Share_Pct": round(r["usa_share_pct"], 2),
                "USMCA_Nearshore_Share_Pct": round(r["usmca_share_pct"], 2),
                "Peak_Warehouse_Util_Pct": round(r["peak_wh_util_pct"], 2),
            }
        )
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/benchmark_comparison.csv", index=False)

    print("\n--- Executive 26-Week 5-Policy Benchmark Comparison ---")
    print(df_summary.to_string(index=False))
    print(f"\nRolling-Horizon MILP Total Runtime (25 order decisions): {res_rh_milp['solve_time_sec']:.2f}s | Perfect-Foresight Ref: {res_pf_milp['solve_time_sec']:.2f}s")

    sensitivity_rows = []
    for scale in (0.10, 0.25, 0.50, 1.00):
        components = df_comp.assign(stockout_penalty_wk=df_comp["stockout_penalty_wk"] * scale)
        static = simulate_heuristic_policy(components, df_lanes, df_sched, df_pipe, mode="static_ocean")
        rolling = res_rh_milp if scale == 1.0 else simulate_rolling_horizon_milp(components, df_lanes, df_sched, df_pipe)
        sensitivity_rows.append({
            "penalty_multiplier": scale,
            "static_cost": static["total_net_cost"],
            "rolling_cost": rolling["total_net_cost"],
            "rolling_fill_pct": rolling["fill_rate_pct"],
            "solver_limits": rolling["limits_reached"],
        })
    pd.DataFrame(sensitivity_rows).to_csv("results/backlog_penalty_sensitivity.csv", index=False)
    print(f"\n6. Running {44 * args.paths_per_strike} strike and demand scenarios...")
    df_mc = run_monte_carlo_stress_test(df_comp, df_lanes, df_sched, df_pipe,
                                       paths_per_strike=args.paths_per_strike, seed=42, workers=args.workers)
    df_mc.to_csv("results/monte_carlo_stress_test.csv", index=False)
    print(
        f"Monte Carlo Mean Savings vs. Static (s, S): "
        f"{df_mc['savings_vs_static_pct'].mean():.2f}% (IQR: "
        f"{df_mc['savings_vs_static_pct'].quantile(0.25):.2f}%–{df_mc['savings_vs_static_pct'].quantile(0.75):.2f}%)"
    )
    print(
        f"Monte Carlo Mean Savings vs. Reactive Air (s, S): "
        f"{df_mc['savings_vs_air_pct'].mean():.2f}% | vs. Dual-Sourced Heuristic: "
        f"{df_mc['savings_vs_dual_heur_pct'].mean():.2f}% (Mean RH-MILP Fill Rate: {df_mc['rolling_milp_fill_rate'].mean():.2f}%)"
    )

    print("\n7. Generating 4 300-DPI Figures in 'figures/'...")
    generate_figures(res_static, res_air, res_dual_h, res_rh_milp, res_pf_milp, df_mc, df_comp, df_lanes, df_sched, df_pipe)
    print("[SUCCESS] All figures and CSV results exported cleanly.")


if __name__ == "__main__":
    main()