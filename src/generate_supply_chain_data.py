"""
Synthetic Benchmark Generator for U.S. Domestic Clean-Tech Manufacturing

Incorporates Real-World OCEMA Container, Chassis & Fleet Payload Physics:
- Offshore Ocean Freight: Restricted strictly to ISO 20-ft Dry and 40-ft High-Cube
  containers with OCEMA / Maersk U.S. highway drayage gross cargo limits:
    * 20ft_Dry_Slider    : 17,780 kg (39,200 lbs) max gross cargo
    * 20ft_Dry_TriAxle   : 19,960 kg (44,000 lbs) max gross cargo (+chassis fee)
    * 40ft_HC_Gooseneck  : 19,820 kg (43,700 lbs) max gross cargo
  All container payload checks evaluate gross packaged unit weight (net product
  weight + 7% pallets, crating, blocking, bracing, and dunnage as a synthetic tare allowance).
- U.S. Domestic & USMCA Ground: Flexible North American fleet options including
  53-ft Dry Van FTL (95 m3 usable, 19,500 kg gross payload), 26-ft Box Truck
  (45 m3 usable, 4,535 kg gross payload), and LTL Palletized Lots.
- Expedited Air Cargo: Air PMC/ULD pallet lots (1-week transit).

Scenario inputs are synthetic. OCEMA recommendations provide drayage payload
reference values. Tariff proxies and a hypothetical U.S. module price
discount are scenario assumptions. The generator excludes air for LFP modules.

Seed: 42
"""

import os
import math
import numpy as np
import pandas as pd


PACKAGING_TARE_MULTIPLIER = 1.07  # +7% pallets, crating, blocking, bracing & dunnage


def compute_equipment_capacity(
    unit_volume_m3: float,
    gross_unit_weight_kg: float,
    usable_volume_m3: float,
    max_gross_payload_kg: float,
) -> tuple[int, str]:
    """
    Computes integer units per container/trailer based on usable volume and
    OCEMA/DOT gross cargo payload limits, returning (lot_size_units, binding_constraint).
    """
    max_by_vol = math.floor(usable_volume_m3 / unit_volume_m3)
    max_by_wt = math.floor(max_gross_payload_kg / gross_unit_weight_kg)
    units = max(1, min(max_by_vol, max_by_wt))
    binding = "Weight_Out" if max_by_wt < max_by_vol else "Cube_Out"
    return units, binding


def generate_supply_chain_benchmark(output_dir: str = "data", seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. COMPONENT MASTER TABLE (6 Critical Sub-Assemblies per 100 kWh BESS Unit)
    # -------------------------------------------------------------------------
    # Line-stoppage penalty (stockout_penalty_wk) set near 1.25x offshore unit value per week
    # as a scenario assumption for the cost of persistent shortages.
    components_data = [
        {
            "component_id": "CMP_LFP_MOD",
            "name": "LFP Battery Module (25 kWh Sub-Pack)",
            "category": "Energy Storage Core",
            "units_per_bess": 4,           # 4 x 25 kWh = 100 kWh enclosure
            "base_apac_unit_cost": 2100.0, # ~$84/kWh BNEF Asian pack benchmark
            "net_unit_weight_kg": 185.0,
            "gross_unit_weight_kg": round(185.0 * PACKAGING_TARE_MULTIPLIER, 2),
            "unit_volume_m3": 0.42,
            "holding_cost_wk": 10.50,      # ~0.45%/wk capital + climate storage
            "stockout_penalty_wk": 2650.0, # Modeled component backlog penalty
            "initial_inventory": 260,
            "target_ending_ss": 200,
            "sec301_tariff_rate": 0.25,    # Synthetic offshore tariff proxy
            "sec45x_credit_unit": 250.0,   # Hypothetical full pass-through of $10/kWh for an eligible cell-based module
        },
        {
            "component_id": "CMP_PCS_INV",
            "name": "Bidirectional PCS Inverter (50 kW)",
            "category": "Power Electronics",
            "units_per_bess": 1,
            "base_apac_unit_cost": 3200.0,
            "net_unit_weight_kg": 95.0,
            "gross_unit_weight_kg": round(95.0 * PACKAGING_TARE_MULTIPLIER, 2),
            "unit_volume_m3": 0.35,
            "holding_cost_wk": 15.00,
            "stockout_penalty_wk": 3950.0,
            "initial_inventory": 70,
            "target_ending_ss": 55,
            "sec301_tariff_rate": 0.50,    # Synthetic offshore tariff proxy
            "sec45x_credit_unit": 0.0,     # PCS eligibility is unspecified in this benchmark
        },
        {
            "component_id": "CMP_BMS_CTL",
            "name": "Master Battery Management System (BMS)",
            "category": "Power Electronics",
            "units_per_bess": 1,
            "base_apac_unit_cost": 680.0,
            "net_unit_weight_kg": 8.5,
            "gross_unit_weight_kg": round(8.5 * PACKAGING_TARE_MULTIPLIER, 2),
            "unit_volume_m3": 0.04,
            "holding_cost_wk": 3.20,
            "stockout_penalty_wk": 1150.0,
            "initial_inventory": 85,
            "target_ending_ss": 65,
            "sec301_tariff_rate": 0.50,    # Synthetic offshore tariff proxy
            "sec45x_credit_unit": 0.0,
        },
        {
            "component_id": "CMP_THM_PLT",
            "name": "Liquid Thermal Cold-Plate Assembly",
            "category": "Thermal Management",
            "units_per_bess": 2,
            "base_apac_unit_cost": 420.0,
            "net_unit_weight_kg": 28.0,
            "gross_unit_weight_kg": round(28.0 * PACKAGING_TARE_MULTIPLIER, 2),
            "unit_volume_m3": 0.14,
            "holding_cost_wk": 2.00,
            "stockout_penalty_wk": 680.0,
            "initial_inventory": 150,
            "target_ending_ss": 110,
            "sec301_tariff_rate": 0.25,    # Synthetic offshore tariff proxy
            "sec45x_credit_unit": 0.0,
        },
        {
            "component_id": "CMP_HV_HARN",
            "name": "High-Voltage Busbar & Wiring Harness",
            "category": "Electrical BOS",
            "units_per_bess": 1,
            "base_apac_unit_cost": 310.0,
            "net_unit_weight_kg": 16.0,
            "gross_unit_weight_kg": round(16.0 * PACKAGING_TARE_MULTIPLIER, 2),
            "unit_volume_m3": 0.06,
            "holding_cost_wk": 1.45,
            "stockout_penalty_wk": 520.0,
            "initial_inventory": 80,
            "target_ending_ss": 60,
            "sec301_tariff_rate": 0.25,
            "sec45x_credit_unit": 0.0,
        },
        {
            "component_id": "CMP_ENC_CAB",
            "name": "NEMA Type 3R Steel Enclosure",
            "category": "Structural BOS",
            "units_per_bess": 1,
            "base_apac_unit_cost": 1150.0,
            "net_unit_weight_kg": 410.0,
            "gross_unit_weight_kg": round(410.0 * PACKAGING_TARE_MULTIPLIER, 2),
            "unit_volume_m3": 1.65,
            "holding_cost_wk": 5.50,
            "stockout_penalty_wk": 1650.0,
            "initial_inventory": 65,
            "target_ending_ss": 50,
            "sec301_tariff_rate": 0.25,    # Synthetic offshore tariff proxy
            "sec45x_credit_unit": 0.0,
        },
    ]
    df_components = pd.DataFrame(components_data)

    # -------------------------------------------------------------------------
    # 2. SUPPLIERS & OCEMA LANE/CHASSIS-SPECIFIC SHIPPING OPTIONS
    # -------------------------------------------------------------------------
    shipping_options = [
        # --- SUP_APAC_1: Tier-1 Low-Cost Offshore (Ningbo/Shenzhen) ---
        {
            "supplier_id": "SUP_APAC_1",
            "supplier_name": "Ningbo Pacific Energy Tech",
            "supplier_region": "Offshore_APAC",
            "freight_mode": "std",
            "equipment_type": "20ft_Dry_Slider",
            "chassis_spec": "Standard Slider Chassis",
            "usable_vol_m3": 28.0,
            "max_gross_payload_kg": 17780.0,  # OCEMA 20ft Dry on Standard Slider
            "price_multiplier": 1.00,
            "tariff_applies": True,
            "module_discount_applies": False,
            "lead_time_wks": 6,
            "equipment_flat_freight": 2950.0,
            "per_kg_freight": 0.0,
            "fixed_order_fee": 950.0,
            "min_lots": 1,
        },
        {
            "supplier_id": "SUP_APAC_1",
            "supplier_name": "Ningbo Pacific Energy Tech",
            "supplier_region": "Offshore_APAC",
            "freight_mode": "std",
            "equipment_type": "20ft_Dry_TriAxle",
            "chassis_spec": "Tri-Axle Slider Chassis",
            "usable_vol_m3": 28.0,
            "max_gross_payload_kg": 19960.0,  # OCEMA 20ft Dry on Tri-Axle Slider
            "price_multiplier": 1.00,
            "tariff_applies": True,
            "module_discount_applies": False,
            "lead_time_wks": 6,
            "equipment_flat_freight": 3170.0,   # Includes +$220 tri-axle chassis split fee
            "per_kg_freight": 0.0,
            "fixed_order_fee": 950.0,
            "min_lots": 1,
        },
        {
            "supplier_id": "SUP_APAC_1",
            "supplier_name": "Ningbo Pacific Energy Tech",
            "supplier_region": "Offshore_APAC",
            "freight_mode": "std",
            "equipment_type": "40ft_HC_Gooseneck",
            "chassis_spec": "Standard Gooseneck (GN) Chassis",
            "usable_vol_m3": 66.0,
            "max_gross_payload_kg": 19820.0,  # OCEMA 40ft High Cube on Standard GN
            "price_multiplier": 0.985,
            "tariff_applies": True,
            "module_discount_applies": False,
            "lead_time_wks": 6,
            "equipment_flat_freight": 4150.0,
            "per_kg_freight": 0.0,
            "fixed_order_fee": 1150.0,
            "min_lots": 1,
        },
        {
            "supplier_id": "SUP_APAC_1",
            "supplier_name": "Ningbo Pacific Energy Tech",
            "supplier_region": "Offshore_APAC",
            "freight_mode": "exp",
            "equipment_type": "Air_PMC_Pallet",
            "chassis_spec": "Air Cargo ULD / Airport Drayage",
            "usable_vol_m3": 4.5,
            "max_gross_payload_kg": 1500.0,
            "price_multiplier": 1.00,
            "tariff_applies": True,
            "module_discount_applies": False,
            "lead_time_wks": 1,
            "equipment_flat_freight": 0.0,
            "per_kg_freight": 5.80,
            "fixed_order_fee": 850.0,
            "min_lots": 1,
        },
        # --- SUP_APAC_2: Secondary Offshore (Busan/Kaohsiung) ---
        {
            "supplier_id": "SUP_APAC_2",
            "supplier_name": "Busan Grid Components Co.",
            "supplier_region": "Offshore_APAC",
            "freight_mode": "std",
            "equipment_type": "20ft_Dry_TriAxle",
            "chassis_spec": "Tri-Axle Slider Chassis",
            "usable_vol_m3": 28.0,
            "max_gross_payload_kg": 19960.0,
            "price_multiplier": 1.06,
            "tariff_applies": True,
            "module_discount_applies": False,
            "lead_time_wks": 5,
            "equipment_flat_freight": 3320.0,
            "per_kg_freight": 0.0,
            "fixed_order_fee": 900.0,
            "min_lots": 1,
        },
        {
            "supplier_id": "SUP_APAC_2",
            "supplier_name": "Busan Grid Components Co.",
            "supplier_region": "Offshore_APAC",
            "freight_mode": "std",
            "equipment_type": "40ft_HC_Gooseneck",
            "chassis_spec": "Standard Gooseneck (GN) Chassis",
            "usable_vol_m3": 66.0,
            "max_gross_payload_kg": 19820.0,
            "price_multiplier": 1.045,
            "tariff_applies": True,
            "module_discount_applies": False,
            "lead_time_wks": 5,
            "equipment_flat_freight": 4350.0,
            "per_kg_freight": 0.0,
            "fixed_order_fee": 1100.0,
            "min_lots": 1,
        },
        {
            "supplier_id": "SUP_APAC_2",
            "supplier_name": "Busan Grid Components Co.",
            "supplier_region": "Offshore_APAC",
            "freight_mode": "exp",
            "equipment_type": "Air_PMC_Pallet",
            "chassis_spec": "Air Cargo ULD / Airport Drayage",
            "usable_vol_m3": 4.5,
            "max_gross_payload_kg": 1500.0,
            "price_multiplier": 1.06,
            "tariff_applies": True,
            "module_discount_applies": False,
            "lead_time_wks": 1,
            "equipment_flat_freight": 0.0,
            "per_kg_freight": 6.10,
            "fixed_order_fee": 820.0,
            "min_lots": 1,
        },
        # --- SUP_MEX_1: Nearshore USMCA (Monterrey, Mexico -> U.S. Plant) ---
        {
            "supplier_id": "SUP_MEX_1",
            "supplier_name": "Monterrey USMCA Industrial",
            "supplier_region": "Nearshore_USMCA",
            "freight_mode": "std",
            "equipment_type": "53ft_FTL_DryVan",
            "chassis_spec": "53-ft Tandem Dry Van Trailer",
            "usable_vol_m3": 95.0,
            "max_gross_payload_kg": 19500.0,
            "price_multiplier": 1.21,
            "tariff_applies": False,
            "module_discount_applies": False,
            "lead_time_wks": 2,
            "equipment_flat_freight": 2400.0,
            "per_kg_freight": 0.0,
            "fixed_order_fee": 480.0,
            "min_lots": 1,
        },
        {
            "supplier_id": "SUP_MEX_1",
            "supplier_name": "Monterrey USMCA Industrial",
            "supplier_region": "Nearshore_USMCA",
            "freight_mode": "exp",
            "equipment_type": "LTL_Team_Expedite",
            "chassis_spec": "Expedited Team-Driver Straight Truck",
            "usable_vol_m3": 24.0,
            "max_gross_payload_kg": 4500.0,
            "price_multiplier": 1.22,
            "tariff_applies": False,
            "module_discount_applies": False,
            "lead_time_wks": 1,
            "equipment_flat_freight": 1650.0,
            "per_kg_freight": 0.0,
            "fixed_order_fee": 520.0,
            "min_lots": 1,
        },
        # --- SUP_USA_1: Domestic U.S. Supplier (Multiple North American Fleet Options) ---
        {
            "supplier_id": "SUP_USA_1",
            "supplier_name": "Heartland Domestic CleanTech Mfg",
            "supplier_region": "Domestic_USA",
            "freight_mode": "std",
            "equipment_type": "53ft_FTL_DryVan",
            "chassis_spec": "53-ft Tandem Dry Van Trailer",
            "usable_vol_m3": 95.0,
            "max_gross_payload_kg": 19500.0,
            "price_multiplier": 1.40,
            "tariff_applies": False,
            "module_discount_applies": True,
            "lead_time_wks": 1,
            "equipment_flat_freight": 1450.0,
            "per_kg_freight": 0.0,
            "fixed_order_fee": 250.0,
            "min_lots": 1,
        },
        {
            "supplier_id": "SUP_USA_1",
            "supplier_name": "Heartland Domestic CleanTech Mfg",
            "supplier_region": "Domestic_USA",
            "freight_mode": "std",
            "equipment_type": "26ft_Box_Truck",
            "chassis_spec": "26-ft Straight Box Truck",
            "usable_vol_m3": 45.0,
            "max_gross_payload_kg": 4535.0,
            "price_multiplier": 1.42,
            "tariff_applies": False,
            "module_discount_applies": True,
            "lead_time_wks": 1,
            "equipment_flat_freight": 780.0,
            "per_kg_freight": 0.0,
            "fixed_order_fee": 200.0,
            "min_lots": 1,
        },
        {
            "supplier_id": "SUP_USA_1",
            "supplier_name": "Heartland Domestic CleanTech Mfg",
            "supplier_region": "Domestic_USA",
            "freight_mode": "exp",
            "equipment_type": "LTL_Pallet_Lot",
            "chassis_spec": "Regional LTL / Hotshot Pallet Service",
            "usable_vol_m3": 14.0,
            "max_gross_payload_kg": 2500.0,
            "price_multiplier": 1.43,
            "tariff_applies": False,
            "module_discount_applies": True,
            "lead_time_wks": 1,
            "equipment_flat_freight": 540.0,
            "per_kg_freight": 0.0,
            "fixed_order_fee": 180.0,
            "min_lots": 1,
        },
    ]

    lanes_rows = []
    for _, comp in df_components.iterrows():
        for opt in shipping_options:
            if comp["component_id"] == "CMP_LFP_MOD" and opt["freight_mode"] == "exp":
                continue
            lot_units, binding_constraint = compute_equipment_capacity(
                unit_volume_m3=comp["unit_volume_m3"],
                gross_unit_weight_kg=comp["gross_unit_weight_kg"],
                usable_volume_m3=opt["usable_vol_m3"],
                max_gross_payload_kg=opt["max_gross_payload_kg"],
            )
            max_practical_lot = int(comp["units_per_bess"] * 45 * 5)
            if lot_units > max_practical_lot:
                lot_units = max_practical_lot
                binding_constraint = "Commercial_Lot_Cap"

            unit_cost = round(comp["base_apac_unit_cost"] * opt["price_multiplier"], 2)
            tariff_cost = (
                round(unit_cost * comp["sec301_tariff_rate"], 2)
                if opt["tariff_applies"]
                else 0.0
            )
            domestic_credit = (
                comp["sec45x_credit_unit"] if opt["module_discount_applies"] else 0.0
            )

            if opt["equipment_flat_freight"] > 0:
                unit_freight = round(opt["equipment_flat_freight"] / lot_units, 2)
            else:
                unit_freight = round(comp["gross_unit_weight_kg"] * opt["per_kg_freight"], 2)

            net_landed_unit_cost = round(
                unit_cost + tariff_cost + unit_freight - domestic_credit, 2
            )
            moq_units = lot_units * opt["min_lots"]

            lanes_rows.append(
                {
                    "lane_id": f"{comp['component_id']}_{opt['supplier_id']}_{opt['equipment_type']}",
                    "component_id": comp["component_id"],
                    "supplier_id": opt["supplier_id"],
                    "supplier_name": opt["supplier_name"],
                    "supplier_region": opt["supplier_region"],
                    "freight_mode": opt["freight_mode"],
                    "equipment_type": opt["equipment_type"],
                    "chassis_spec": opt["chassis_spec"],
                    "max_gross_payload_kg": opt["max_gross_payload_kg"],
                    "lot_size_units": lot_units,
                    "binding_constraint": binding_constraint,
                    "min_lots": opt["min_lots"],
                    "moq_units": moq_units,
                    "unit_cost": unit_cost,
                    "sec301_tariff_cost": tariff_cost,
                    "domestic_credit": domestic_credit,
                    "unit_freight_cost": unit_freight,
                    "net_landed_unit_cost": net_landed_unit_cost,
                    "fixed_order_fee": opt["fixed_order_fee"],
                    "lead_time_wks": opt["lead_time_wks"],
                }
            )
    df_lanes = pd.DataFrame(lanes_rows)

    # -------------------------------------------------------------------------
    # 3. WEEKLY DEMAND & SUPPLIER CAPACITY SCHEDULE (T = 26 Weeks)
    # -------------------------------------------------------------------------
    horizon_weeks = 26
    base_bess_build = np.full(horizon_weeks, 45.0)
    # Q2/Q3 utility grid interconnection surge (Weeks 14-19)
    base_bess_build[13:19] = [56, 61, 64, 62, 58, 53]
    actual_bess_build = np.round(
        base_bess_build + rng.normal(0, 2.2, size=horizon_weeks)
    ).astype(int)

    unique_suppliers = (
        df_lanes[["supplier_id", "supplier_region"]].drop_duplicates().to_dict("records")
    )

    schedule_rows = []
    for t in range(1, horizon_weeks + 1):
        port_shock = 1 if (11 <= t <= 14) else 0
        for _, comp in df_components.iterrows():
            f_demand = int(round(base_bess_build[t - 1] * comp["units_per_bess"]))
            a_demand = int(actual_bess_build[t - 1] * comp["units_per_bess"])

            for sup in unique_suppliers:
                sid = sup["supplier_id"]
                reg = sup["supplier_region"]
                if reg == "Offshore_APAC":
                    base_cap = int(round(45 * comp["units_per_bess"] * 1.50))
                    weekly_cap = int(round(base_cap * 0.15)) if port_shock else base_cap
                elif reg == "Nearshore_USMCA":
                    weekly_cap = int(round(45 * comp["units_per_bess"] * 0.85))
                else:  # Domestic_USA
                    weekly_cap = int(round(45 * comp["units_per_bess"] * 0.75))

                schedule_rows.append(
                    {
                        "week": t,
                        "component_id": comp["component_id"],
                        "supplier_id": sid,
                        "supplier_region": reg,
                        "forecast_demand": f_demand,
                        "actual_demand": a_demand,
                        "weekly_supply_cap": weekly_cap,
                        "disruption_flag": port_shock if reg == "Offshore_APAC" else 0,
                    }
                )
    df_schedule = pd.DataFrame(schedule_rows)

    # -------------------------------------------------------------------------
    # 4. IN-TRANSIT PIPELINE ORDERS (Pre-Horizon Open POs Arriving Weeks 1-5)
    # -------------------------------------------------------------------------
    pipeline_rows = []
    for _, comp in df_components.iterrows():
        cid = comp["component_id"]
        lot_20ft = int(
            df_lanes[
                (df_lanes["component_id"] == cid)
                & (df_lanes["supplier_id"] == "SUP_APAC_1")
                & (df_lanes["equipment_type"] == "20ft_Dry_Slider")
            ]["lot_size_units"].iloc[0]
        )
        lot_usa_box = int(
            df_lanes[
                (df_lanes["component_id"] == cid)
                & (df_lanes["supplier_id"] == "SUP_USA_1")
                & (df_lanes["equipment_type"] == "26ft_Box_Truck")
            ]["lot_size_units"].iloc[0]
        )
        pipeline_rows.append(
            {
                "component_id": cid,
                "supplier_id": "SUP_APAC_1",
                "equipment_type": "20ft_Dry_Slider",
                "arrival_week": 2,
                "scheduled_qty": lot_20ft,
            }
        )
        pipeline_rows.append(
            {
                "component_id": cid,
                "supplier_id": "SUP_APAC_1",
                "equipment_type": "20ft_Dry_Slider",
                "arrival_week": 4,
                "scheduled_qty": lot_20ft,
            }
        )
        pipeline_rows.append(
            {
                "component_id": cid,
                "supplier_id": "SUP_USA_1",
                "equipment_type": "26ft_Box_Truck",
                "arrival_week": 1,
                "scheduled_qty": lot_usa_box,
            }
        )
    df_pipeline = pd.DataFrame(pipeline_rows)

    df_components.to_csv(os.path.join(output_dir, "components.csv"), index=False)
    df_lanes.to_csv(os.path.join(output_dir, "suppliers_lanes.csv"), index=False)
    df_schedule.to_csv(os.path.join(output_dir, "weekly_demand_capacity.csv"), index=False)
    df_pipeline.to_csv(os.path.join(output_dir, "in_transit_pipeline.csv"), index=False)

    print(f"[SUCCESS] Generated Synthetic Supply-Chain Dataset in '{output_dir}/' (seed={seed}):")
    print(f"  - components.csv              : {len(df_components)} components")
    print(f"  - suppliers_lanes.csv         : {len(df_lanes)} lane/chassis equipment options")
    print(f"  - weekly_demand_capacity.csv  : {len(df_schedule)} weekly schedule records (T=26 wks)")
    print(f"  - in_transit_pipeline.csv     : {len(df_pipeline)} open pipeline shipments")


if __name__ == "__main__":
    generate_supply_chain_benchmark()