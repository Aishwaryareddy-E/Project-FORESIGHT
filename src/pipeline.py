"""Project FORESIGHT - reproducible raw-to-weekly pipeline."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def load_raw():
    sales = pd.read_csv(RAW / "sales_daily.csv")
    sku = pd.read_csv(RAW / "sku_master.csv")
    calendar = pd.read_csv(RAW / "calendar.csv")
    inventory = pd.read_csv(RAW / "inventory_snapshots.csv")
    return sales, sku, calendar, inventory


def build_pipeline():
    sales, sku, calendar, inventory = load_raw()

    # Standardise raw column names.
    sales = sales.rename(columns={
        "Date":"date", "SKU":"sku", "Units_Sold":"units_sold",
        "Revenue":"revenue", "Price":"price", "Promotion":"promotion"
    })
    sku = sku.rename(columns={
        "SKU":"sku", "Product_Name":"product_name", "Category":"category",
        "Subcategory":"subcategory", "Launch_Date":"launch_date",
        "Cost_Price":"cost_price", "Selling_Price":"selling_price",
        "Gross_Margin_Per_Unit":"gross_margin_per_unit"
    })
    inventory = inventory.rename(columns={
        "Snapshot_Date":"date", "SKU":"sku_id", "Current_Stock":"on_hand_units",
        "On_Order":"on_order_units", "Lead_Time_Days":"lead_time_days",
        "Safety_Stock":"safety_stock", "Reorder_Point":"reorder_point",
        "Inventory_Value":"inventory_value"
    })

    sales["date"] = pd.to_datetime(sales["date"], errors="raise")
    sku["launch_date"] = pd.to_datetime(sku["launch_date"], errors="raise")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    inventory["date"] = pd.to_datetime(inventory["date"], errors="raise")

    # Calendar descriptive blanks are meaningful: no event.
    calendar["holiday"] = calendar["holiday"].fillna("No Holiday")
    calendar["promotion_event"] = calendar["promotion_event"].fillna("No Promotion")

    # Merge the daily fact with SKU master and calendar dimension.
    daily = (
        sales.merge(sku, on="sku", how="left", validate="many_to_one")
             .merge(calendar, on="date", how="left", validate="many_to_one")
    )

    # Validation / data-quality findings.
    dq = {
        "raw_sales_rows": int(len(sales)),
        "raw_sku_rows": int(len(sku)),
        "raw_calendar_rows": int(len(calendar)),
        "raw_inventory_rows": int(len(inventory)),
        "sales_missing_values": int(sales.isna().sum().sum()),
        "sales_duplicate_rows": int(sales.duplicated().sum()),
        "sales_duplicate_sku_date": int(sales.duplicated(["sku", "date"]).sum()),
        "sales_unique_skus": int(sales["sku"].nunique()),
        "sales_unique_dates": int(sales["date"].nunique()),
        "revenue_mismatches": int((sales["revenue"] - sales["units_sold"]*sales["price"]).abs().gt(0.01).sum()),
        "negative_gross_margin_rows": int((daily["gross_margin_per_unit"] < 0).sum()),
        "pre_launch_sales_rows": int((daily["date"] < daily["launch_date"]).sum()),
        "calendar_missing_holiday_labels_before_fill": int(pd.read_csv(RAW/"calendar.csv")["holiday"].isna().sum()),
        "calendar_missing_promotion_labels_before_fill": int(pd.read_csv(RAW/"calendar.csv")["promotion_event"].isna().sum()),
        "inventory_unique_skus": int(inventory["sku_id"].nunique()),
        "inventory_unique_snapshots": int(inventory["date"].nunique()),
    }

    # Keep inventory only for the 50 project SKUs.
    project_skus = set(sku["sku"])
    project_inventory = inventory[inventory["sku_id"].isin(project_skus)].copy()

    # Weekly SKU aggregation.
    iso = daily["date"].dt.isocalendar()
    daily["iso_year"] = iso.year.astype(int)
    daily["iso_week"] = iso.week.astype(int)
    weekly = (
        daily.groupby(["iso_year", "iso_week", "sku"], as_index=False)
             .agg(
                 units_sold=("units_sold", "sum"),
                 revenue=("revenue", "sum"),
                 promotion_days=("promotion", "sum")
             )
    )
    weekly["week_start"] = pd.to_datetime(
        weekly["iso_year"].astype(str) + "-W" + weekly["iso_week"].astype(str) + "-1",
        format="%G-W%V-%u",
    )

    # Time-aware mapping: latest snapshot available on/before week start.
    weekly = weekly.sort_values(["week_start", "sku"]).reset_index(drop=True)
    project_inventory = project_inventory.sort_values(["date", "sku_id"]).reset_index(drop=True)
    merged = pd.merge_asof(
        weekly,
        project_inventory,
        left_on="week_start",
        right_on="date",
        left_by="sku",
        right_by="sku_id",
        direction="backward",
        allow_exact_matches=True,
    )

    sku_info = sku[[
        "sku", "product_name", "category", "subcategory", "launch_date",
        "cost_price", "selling_price", "gross_margin_per_unit"
    ]].drop_duplicates("sku")

    analysis = merged.merge(sku_info, on="sku", how="left", validate="many_to_one")
    analysis = analysis.rename(columns={"date":"inventory_snapshot_date"}).drop(columns=["sku_id"])
    analysis["year"] = analysis["week_start"].dt.year
    analysis["month"] = analysis["week_start"].dt.month
    analysis["quarter"] = analysis["week_start"].dt.quarter
    analysis["week_of_year"] = analysis["week_start"].dt.isocalendar().week.astype(int)
    analysis["week_end"] = analysis["week_start"] + pd.Timedelta(days=6)
    max_sales_date = daily["date"].max()
    analysis["is_partial_week"] = analysis["week_end"] > max_sales_date
    analysis = analysis.sort_values(["sku", "week_start"]).reset_index(drop=True)

    # Validation fields.
    dq.update({
        "project_inventory_rows": int(len(project_inventory)),
        "project_inventory_skus": int(project_inventory["sku_id"].nunique()),
        "project_inventory_snapshots": int(project_inventory["date"].nunique()),
        "analysis_rows": int(len(analysis)),
        "analysis_skus": int(analysis["sku"].nunique()),
        "analysis_unique_sku_week": int(analysis[["sku","week_start"]].drop_duplicates().shape[0]),
        "analysis_missing_values": int(analysis.isna().sum().sum()),
        "analysis_future_inventory_mappings": int((analysis["inventory_snapshot_date"] > analysis["week_start"]).sum()),
        "partial_week_rows": int(analysis["is_partial_week"].sum()),
        "partial_week_start": str(analysis.loc[analysis["is_partial_week"], "week_start"].min().date()) if analysis["is_partial_week"].any() else None,
    })

    # Inventory value is left as supplied; formula is documented as a quality finding.
    inv_check = project_inventory.merge(
        sku[["sku", "cost_price"]].drop_duplicates("sku"),
        left_on="sku_id", right_on="sku", how="left", validate="many_to_one"
    )
    inv_check["calculated_inventory_value"] = inv_check["on_hand_units"] * inv_check["cost_price"]
    dq["inventory_value_mismatches"] = int(
        (inv_check["inventory_value"] - inv_check["calculated_inventory_value"]).abs().gt(0.01).sum()
    )

    analysis.to_csv(PROCESSED / "analysis_ready_weekly.csv", index=False)
    (REPORTS / "data_quality_summary.json").write_text(json.dumps(dq, indent=2, default=str), encoding="utf-8")
    return analysis, dq


if __name__ == "__main__":
    analysis, dq = build_pipeline()
    print(json.dumps(dq, indent=2, default=str))
