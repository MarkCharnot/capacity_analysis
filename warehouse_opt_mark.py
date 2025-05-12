import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from scipy.optimize import minimize
import pulp

def load_parameters():
    """
    Load default parameters for warehouse design, inventory handling, and facility constraints.
    """
    return {
        # Inventory handling
        "DOH": 250,
        "operating_days": 240,

        # Pallet dimensions (in inches)
        "pallet_length_inches": 48,
        "pallet_width_inches": 40,
        "rack_height_inches": 79.2,

        # Aisle factor for spacing
        "aisle_factor": 0.5,

        # Building specs
        "ceiling_height_inches": 288,
        "max_utilization": 0.8,

        # Support areas (sq ft)
        "min_office": 1000,
        "min_battery": 500,
        "min_packing": 2000,
        "min_conveyor": 6000,

        # Dock capacity and dimensions
        "outbound_area_per_door": 4000,
        "outbound_pallets_per_door_per_day": 40,
        "max_outbound_doors": 10,
        "inbound_area_per_door": 4000,
        "inbound_pallets_per_door_per_day": 40,
        "max_inbound_doors": 10,

        # Pick modules
        "each_pick_area_fixed": 24000,
        "case_pick_area_fixed": 44000,

        # Multi-facility modeling
        "facility_lease_years": 7,
        "num_facilities": 3,
        "initial_facility_area": 140000,
        "facility_design_area": 350000,

        # Freight distribution by facility
        "freight_pct": [0.23, 0.47, 0.30],
    }

def detect_column(df, candidates):
    """
    Identify the appropriate column from a set of candidate names.
    """
    for col in df.columns:
        low = col.lower().replace(" ", "")
        for cand in candidates:
            if cand in low:
                return col
    return None

def load_and_extend_forecast(path, params):
    """
    Load forecast CSV and extrapolate additional years (up to 2036) using quadratic regression.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8")
        df = df.rename(columns={df.columns[0]: "year", df.columns[1]: "annual_units"})
    except:
        df = pd.read_csv(path, sep=r"\s+", header=None,
                         names=["year", "annual_units"], skiprows=1,
                         engine="python", encoding="utf-8")

    df["year"] = df["year"].astype(int)
    df["annual_units"] = pd.to_numeric(df["annual_units"].astype(str).str.replace(",", ""), errors="raise")
    df = df[df["year"] >= 2025].sort_values("year")

    coeffs = np.polyfit(df["year"], df["annual_units"], deg=2)
    poly = np.poly1d(coeffs)
    future_years = np.arange(2031, 2040)
    future_units = poly(future_years)

    df_ext = pd.DataFrame({"year": list(future_years), "annual_units": future_units})
    df_combined = pd.concat([df, df_ext], ignore_index=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(df["year"], df["annual_units"], color='blue', label='Original Data')
    plt.plot(df_combined["year"], poly(df_combined["year"]), color='red', linestyle='--', label='Fitted Trend')
    plt.scatter(future_years, future_units, color='green', label='Extrapolated')
    plt.xlabel("Year")
    plt.ylabel("Annual Units")
    plt.title("Annual Units Forecast")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df_combined

def load_skus(path):
    """
    Load SKU data and calculate units per pallet.
    """
    df = pd.read_csv(path, sep=",", engine="python", encoding="utf-8")
    up_col = detect_column(df, ["unitspercase", "units_case", "units_per_case"])
    cp_col = detect_column(df, ["casesperpallet", "case_pallet", "cases_per_pallet"])
    sku_col = detect_column(df, ["annual_units", "unitsperyear"])

    if not up_col or not cp_col or not sku_col:
        raise ValueError("skus.csv missing required columns")

    df[up_col] = pd.to_numeric(df[up_col], errors="coerce").fillna(0)
    df[cp_col] = pd.to_numeric(df[cp_col], errors="coerce").fillna(0)
    df["units_per_pallet"] = (df[up_col] * df[cp_col]).astype(float)
    pos = df["units_per_pallet"] > 0
    avg = df.loc[pos, "units_per_pallet"].mean() if pos.any() else 1.0
    df.loc[~pos, "units_per_pallet"] = avg
    df[sku_col] = pd.to_numeric(df[sku_col].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
    df = df[df[sku_col] > 0]
    return df, "units_per_pallet", sku_col

def compute_areas(forecast_df, sku_df, up_col, sku_col, params):
    """
    Compute all required warehouse areas, dock constraints, and facility planning.
    """
    results = []
    gross_base_area_by_year = {}
    area_per_x1_by_year = {}
    static_values_by_year = {}
    net_base_area_by_year = {}
    freight_pct = params.get("freight_pct", [])
    area_per_x1 = params["case_pick_area_fixed"] + params["each_pick_area_fixed"] + params["min_conveyor"]

    for _, row in forecast_df.iterrows():
        year = int(row["year"])
        annual_units = row["annual_units"]
        base_units = sum(sku_df[sku_col])
        demand_factor = annual_units / base_units if base_units else 1

        adjusted_units = sku_df[sku_col] * demand_factor
        annual_pallets = (adjusted_units / sku_df[up_col]).sum()
        daily_pallets = annual_pallets / params["operating_days"]
        storage_pallets = daily_pallets * params["DOH"]

        pallet_ft2 = (params["pallet_length_inches"] / 12) * (params["pallet_width_inches"] / 12)
        levels = math.floor(params["ceiling_height_inches"] / params["rack_height_inches"])
        aisle_factor = params["aisle_factor"]
        storage_area = storage_pallets * pallet_ft2 / levels / (1 - aisle_factor)

        ob_doors = min(math.ceil(daily_pallets / params["outbound_pallets_per_door_per_day"]), params["max_outbound_doors"])
        oa = ob_doors * params["outbound_area_per_door"]
        osl = max(0, math.ceil(daily_pallets / params["outbound_pallets_per_door_per_day"]) - params["max_outbound_doors"]) * params["outbound_area_per_door"]
        ib_doors = min(math.ceil(daily_pallets / params["inbound_pallets_per_door_per_day"]), params["max_inbound_doors"])
        ia = ib_doors * params["inbound_area_per_door"]
        isl = max(0, math.ceil(daily_pallets / params["inbound_pallets_per_door_per_day"]) - params["max_inbound_doors"]) * params["inbound_area_per_door"]
        support_area = params["min_office"] + params["min_battery"] + params["min_packing"]

        net_base_area = storage_area + oa + osl + ia + isl + support_area
        gross_base_area = net_base_area / params["max_utilization"]

        net_base_area_by_year[year] = net_base_area
        gross_base_area_by_year[year] = gross_base_area
        area_per_x1_by_year[year] = area_per_x1

        static_values_by_year[year] = {
            "Storage Pallets": storage_pallets,
            "Storage Area": storage_area,
            "Outbound Dock Area": oa,
            "Outbound Slack Area": osl,
            "Inbound Dock Area": ia,
            "Inbound Slack Area": isl,
            "Support Area": support_area,
            "Net Area": net_base_area
        }
    print(gross_base_area_by_year)
    # Optimization model
    prob = pulp.LpProblem("Optimize_Facilities_and_3PL", pulp.LpMinimize)
    add_facilities_req = {
    2025: pulp.LpVariable("add_facilities_req_y2025", lowBound=0, upBound=0, cat='Integer'),  # Can't add in 2025
    **{
        year: pulp.LpVariable(f"add_facilities_req_y{year}", lowBound=0, upBound=1, cat='Integer')
        for year in range(2026, 2040)
      }
    }
    third_party_req_sqft = {
        year: pulp.LpVariable(f"third_party_req_sqft{year}", lowBound=0, upBound=300000, cat='Continuous')
        for year in range(2025, 2040)
    }
    size_of_fac = {
        year: pulp.LpVariable(f"size_of_fac{year}", lowBound=0, upBound=300000, cat='Integer')
        for year in range(2025, 2040)
    }

    weight_add_facilities = 1000
    weight_third_party = .00000005*1e-5
    weight_facility_size = 10000*1e-6

    prob += (
        pulp.lpSum(weight_add_facilities * add_facilities_req[year] for year in add_facilities_req) +
        pulp.lpSum(weight_third_party * third_party_req_sqft[year] for year in third_party_req_sqft) +
        pulp.lpSum(weight_facility_size * size_of_fac[year] for year in size_of_fac)
    ), "Minimize Facilities and 3PL"

    for year in range(2025, 2040):
        prior_years = [y for y in range(2025, year + 1)]
        total_facility_area_expr = (pulp.lpSum(add_facilities_req[y] for y in prior_years)+1) * area_per_x1_by_year[year]
        total_built_area_expr = pulp.lpSum(size_of_fac[y] for y in prior_years) + params["initial_facility_area"]
        prob += (
            gross_base_area_by_year[year] + total_facility_area_expr - third_party_req_sqft[year]
            <= total_built_area_expr
        ), f"CapacityConstraint_{year}"
        prob += size_of_fac[year] <= 300000 * add_facilities_req[year], f"FacilitySizeLink_{year}"
    prob += pulp.lpSum(add_facilities_req[year] for year in add_facilities_req) <= 8, "TotalFacilityLimit"

    print(prob)
    prob.solve()
    print('Size of Fac Added')
    for year, var in size_of_fac.items():
        print(f"Year {year}: {var.varValue}")
    print('Num of Fac Added')
    for year, var in add_facilities_req.items():
        print(f"Year {year}: {var.varValue}")
    for year in range(2025, 2040):
        facilities = sum(int(add_facilities_req[y].varValue) for y in range(2025, year + 1))
        static = static_values_by_year[year]

        results.append({
            "Year": year,
            **static,
            "Case Pick Area": (facilities + 1) * params["case_pick_area_fixed"],
            "Each Pick Area": (facilities + 1) * params["each_pick_area_fixed"],
            "Conveyor Area": (facilities + 1) * params["min_conveyor"],
            "Gross Area": (net_base_area_by_year[year] + (facilities + 1) * params["case_pick_area_fixed"] + (facilities + 1) * params["each_pick_area_fixed"] + (facilities + 1) * params["min_conveyor"])// params["max_utilization"],
            "Facilities Needed": facilities + 1,
            "3PL Sq Ft Requied": int(third_party_req_sqft[year].varValue)
        })

    return pd.DataFrame(results).sort_values("Year")


def main():
    params = load_parameters()
    forecast_df = load_and_extend_forecast("forecast.csv", params)
    sku_df, up_col, sku_col = load_skus("skus.csv")
    results_df = compute_areas(forecast_df, sku_df, up_col, sku_col, params)

    output_path = 'C:/git/optimizer/outs/results.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        results_df.to_csv(output_path, index=False)
        print(f"Results written to {output_path}")
    except PermissionError:
        print(f"Permission denied writing to {output_path}. Close the file if it's open and retry.")

if __name__ == "__main__":
    main()
