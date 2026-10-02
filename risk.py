"""Project FORESIGHT - transparent lead-time risk scoring and decisioning."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def build_risk():
    forecast = pd.read_csv(PROCESSED / "demand_forecast_results.csv", parse_dates=["forecast_week", "reference_week", "latest_inventory_snapshot_date"])
    df = forecast.copy().sort_values(["sku", "forecast_week"]).reset_index(drop=True)

    # Inventory position available at the start of the planning horizon.
    df["starting_inventory_position"] = (
        df.groupby("sku")["latest_known_on_hand_units"].transform("first")
        + df.groupby("sku")["latest_known_on_order_units"].transform("first")
    )

    # Project inventory forward under a transparent planning assumption:
    # starting inventory is consumed by forecast demand across the horizon.
    df["cumulative_prior_forecast"] = (
        df.groupby("sku")["forecast_demand"].cumsum() - df["forecast_demand"]
    )
    df["inventory_before_forecast"] = (
        df["starting_inventory_position"] - df["cumulative_prior_forecast"]
    )
    df["inventory_after_forecast"] = (
        df["inventory_before_forecast"] - df["forecast_demand"]
    )

    # Calculate stockout risk using forecast demand over the replenishment lead time.
    # This follows the FORESIGHT brief: compare lead-time demand with inventory position.

    # Forecast demand expected during the lead time.
    df["lead_time_demand"] = (
        df["forecast_demand"] * df["lead_time_days"] / 7.0
    )

    # Inventory remaining after covering the expected lead-time demand.
    df["inventory_after_lead_time"] = (
        df["inventory_before_forecast"] - df["lead_time_demand"]
    )

    # Buffer above the safety-stock level.
    df["stockout_buffer_units"] = (
        df["inventory_after_lead_time"] - df["safety_stock"]
    )

    # Assign numeric stockout risk.
    df["stockout_risk_score"] = np.select(
        [
            df["inventory_after_lead_time"] < 0,
            df["inventory_after_lead_time"] < df["safety_stock"],
        ],
        [1.0, 0.5],
        default=0.0,
    )

    # Convert score to readable risk level.
    df["stockout_risk"] = np.select(
        [
            df["stockout_risk_score"].eq(1.0),
            df["stockout_risk_score"].eq(0.5),
        ],
        ["High", "Medium"],
        default="Low",
    )

    # Forward coverage uses projected inventory available before that week's demand.
    df["inventory_coverage_weeks"] = (
        df["inventory_before_forecast"].clip(lower=0)
        / df["forecast_demand"].replace(0, np.nan)
    )
    df["inventory_coverage_weeks"] = (
        df["inventory_coverage_weeks"].replace([np.inf, -np.inf], np.nan)
    )

    # Project-defined overstock thresholds; documented as business rules.
    df["overstock_risk_score"] = np.where(
        df["inventory_coverage_weeks"] > 13,
        1.0,
        np.where(df["inventory_coverage_weeks"] > 8, 0.5, 0.0),
    )
    df["overstock_risk"] = np.select(
        [df["overstock_risk_score"].eq(1.0), df["overstock_risk_score"].eq(0.5)],
        ["High", "Medium"],
        default="Low",
    )

    # Create the four decisioning groups required by the FORESIGHT framework.
    # Medium risk is treated as elevated risk so it is not incorrectly labelled Healthy.
    df["decision_quadrant"] = np.select(
        [
            (
                df["stockout_risk"].isin(["Medium", "High"])
                & df["overstock_risk"].isin(["Medium", "High"])
            ),
            (
                df["stockout_risk"].isin(["Medium", "High"])
                & (df["overstock_risk"] == "Low")
            ),
            (
                (df["stockout_risk"] == "Low")
                & df["overstock_risk"].isin(["Medium", "High"])
            ),
        ],
        ["WATCH / VOLATILE", "REORDER NOW", "MARKDOWN / CLEAR"],
        default="HEALTHY",
    )

    # Map each decision to an operational action.
    df["recommended_action"] = np.select(
        [
            df["decision_quadrant"].eq("REORDER NOW"),
            df["decision_quadrant"].eq("MARKDOWN / CLEAR"),
            df["decision_quadrant"].eq("WATCH / VOLATILE"),
        ],
        ["REORDER", "MARKDOWN", "MONITOR"],
        default="NO ACTION",
    )

    # Estimate demand that cannot be covered during the replenishment lead time.
    df["stockout_units"] = np.maximum(
        -df["inventory_after_lead_time"], 0
    )

    # Potential sales value at the selling price.
    df["potential_lost_sales_value"] = (
        df["stockout_units"] * df["selling_price"]
    )

    # Potential gross margin at risk.
    df["stockout_margin_at_risk"] = (
        df["stockout_units"] * df["gross_margin_per_unit"]
    )

    df["overstock_units"] = np.maximum(
        df["inventory_before_forecast"] - (df["forecast_demand"] * 13), 0
    )
    df["potential_overstock_inventory_value"] = (
        df["overstock_units"] * df["cost_price"]
    )

    # Estimate units required to bring projected inventory back to the reorder point.
    df["reorder_units"] = np.maximum(
        df["reorder_point"] - df["inventory_after_lead_time"], 0
    )

    # Estimated replenishment cost.
    df["reorder_value"] = df["reorder_units"] * df["cost_price"]

    # Calculate financial priority for each recommended action.
    df["priority_value"] = np.select(
        [
            df["recommended_action"].eq("REORDER"),
            df["recommended_action"].eq("MARKDOWN"),
        ],
        [
            df["reorder_value"],
            df["potential_overstock_inventory_value"],
        ],
        default=0,
    )

    # Create simple priority bands.
    df["priority_level"] = pd.cut(
        df["priority_value"],
        bins=[-np.inf, 10_000, 100_000, np.inf],
        labels=["Low", "Medium", "High"],
    )

    # Final audit fields.
    df["risk_method"] = "Lead-time demand + progressive inventory position"
    df["inventory_assumption"] = (
        "Latest known on-hand + on-order form starting inventory; exact future inbound dates unavailable."
    )

    df.to_csv(PROCESSED / "risk_scoring_results.csv", index=False)

    summary = {
        "rows": int(len(df)),
        "skus": int(df["sku"].nunique()),
        "forecast_weeks": int(df["forecast_week"].nunique()),
        "stockout_risk": df["stockout_risk"].value_counts().to_dict(),
        "overstock_risk": df["overstock_risk"].value_counts().to_dict(),
        "decision_quadrants": df["decision_quadrant"].value_counts().to_dict(),
        "recommended_actions": df["recommended_action"].value_counts().to_dict(),
        "potential_lost_sales_value": float(df["potential_lost_sales_value"].sum()),
        "potential_overstock_inventory_value": float(df["potential_overstock_inventory_value"].sum()),
        "estimated_reorder_value": float(df["reorder_value"].sum()),
        "negative_coverage_values": int((df["inventory_coverage_weeks"] < 0).sum()),
        "missing_decisions": int(df["decision_quadrant"].isna().sum()),
    }
    import json
    (REPORTS / "risk_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return df


if __name__ == "__main__":
    build_risk()
