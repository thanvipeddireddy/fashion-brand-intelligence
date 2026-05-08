import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


# =========================
# LOAD DATA
# =========================
products = pd.read_csv("fashion_boutique_dataset.csv")
costs = pd.read_csv("true_cost_fast_fashion.csv")
wages = pd.read_csv("country-monthlylivableusd.csv")

products["is_returned"] = products["is_returned"].astype(int)
products["price_gap"] = products["original_price"] - products["current_price"]
products["discount_flag"] = (products["markdown_percentage"] > 0).astype(int)
products["brand_clean"] = products["brand"].str.lower().str.replace(" ", "", regex=False)

costs["brand_clean"] = costs["Brand"].str.lower().str.replace(" ", "", regex=False)
costs["Country"] = costs["Country"].str.strip()
wages["country"] = wages["country"].str.strip()


# =========================
# BRAND STRATEGY TABLE
# =========================
brand_strategy = products.groupby("brand").agg(
    avg_price=("current_price", "mean"),
    avg_original_price=("original_price", "mean"),
    avg_markdown=("markdown_percentage", "mean"),
    avg_rating=("customer_rating", "mean"),
    return_rate=("is_returned", "mean"),
    avg_stock=("stock_quantity", "mean"),
    product_count=("product_id", "count"),
    discount_rate=("discount_flag", "mean"),
).reset_index()

category_mix = pd.crosstab(products["brand"], products["category"], normalize="index")
category_mix.columns = [f"category_{c}_share" for c in category_mix.columns]

season_mix = pd.crosstab(products["brand"], products["season"], normalize="index")
season_mix.columns = [f"season_{c}_share" for c in season_mix.columns]

brand_strategy = (
    brand_strategy
    .merge(category_mix, left_on="brand", right_index=True, how="left")
    .merge(season_mix, left_on="brand", right_index=True, how="left")
)

strategy_features = [
    "avg_price",
    "avg_markdown",
    "avg_rating",
    "return_rate",
    "avg_stock",
    "product_count",
    "discount_rate",
]

X_strategy = StandardScaler().fit_transform(brand_strategy[strategy_features])

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_strategy)

brand_strategy["pc1"] = pcs[:, 0]
brand_strategy["pc2"] = pcs[:, 1]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
brand_strategy["cluster"] = kmeans.fit_predict(X_strategy)

cluster_names = {
    0: "Balanced Operators",
    1: "Discount-Driven Brands",
    2: "High-Scale / High-Risk Brands"
}

brand_strategy["cluster_name"] = brand_strategy["cluster"].map(cluster_names)

pca_loadings = pd.DataFrame(
    pca.components_.T,
    columns=["pc1_loading", "pc2_loading"],
    index=strategy_features
).reset_index().rename(columns={"index": "feature"})


# =========================
# PRODUCT RETURN RISK MODEL
# =========================
model_cols = [
    "brand",
    "category",
    "season",
    "size",
    "color",
    "original_price",
    "current_price",
    "markdown_percentage",
    "price_gap",
    "discount_flag",
    "stock_quantity",
]

df_model = products[model_cols + ["is_returned"]].dropna().copy()

X = pd.get_dummies(
    df_model[model_cols],
    columns=["brand", "category", "season", "size", "color"],
    drop_first=False
)

y = df_model["is_returned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

return_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=4,
    random_state=42
)

return_model.fit(X_train, y_train)

y_pred = return_model.predict(X_test)
y_prob = return_model.predict_proba(X_test)[:, 1]

model_metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "auc": float(roc_auc_score(y_test, y_prob)),
}

products_encoded = pd.get_dummies(
    products[model_cols],
    columns=["brand", "category", "season", "size", "color"],
    drop_first=False
)

products_encoded = products_encoded.reindex(columns=X.columns, fill_value=0)

products["predicted_return_risk"] = return_model.predict_proba(products_encoded)[:, 1]

product_return_risk = products[
    [
        "product_id",
        "brand",
        "category",
        "season",
        "current_price",
        "markdown_percentage",
        "stock_quantity",
        "customer_rating",
        "is_returned",
        "predicted_return_risk",
    ]
].copy()

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": return_model.feature_importances_
}).sort_values("importance", ascending=False)


# =========================
# SUSTAINABILITY OVERLAY
# =========================
cost_brand = costs.groupby("brand_clean").agg(
    production=("Monthly_Production_Tonnes", "mean"),
    release_cycles=("Release_Cycles_Per_Year", "mean"),
    emissions=("Carbon_Emissions_tCO2e", "mean"),
    water=("Water_Usage_Million_Litres", "mean"),
    waste=("Landfill_Waste_Tonnes", "mean"),
    worker_wage=("Avg_Worker_Wage_USD", "mean"),
    env_cost=("Env_Cost_Index", "mean"),
    sustainability_score=("Sustainability_Score", "mean"),
    transparency=("Transparency_Index", "mean"),
    compliance=("Compliance_Score", "mean"),
    ethical_rating=("Ethical_Rating", "mean"),
).reset_index()

brand_lookup = products[["brand", "brand_clean"]].drop_duplicates()
cost_brand = cost_brand.merge(brand_lookup, on="brand_clean", how="left").dropna(subset=["brand"])

sustain_cols_bad = ["emissions", "water", "waste", "env_cost"]
sustain_cols_good = ["sustainability_score", "transparency", "compliance", "ethical_rating", "worker_wage"]

sustain_work = cost_brand.copy()
for col in sustain_cols_bad:
    sustain_work[col] = -sustain_work[col]

score_cols = sustain_cols_bad + sustain_cols_good
sustain_scaled = StandardScaler().fit_transform(sustain_work[score_cols])

sustain_pca = PCA(n_components=2)
sustain_pcs = sustain_pca.fit_transform(sustain_scaled)

cost_brand["sustain_pc1"] = sustain_pcs[:, 0]
cost_brand["sustain_pc2"] = sustain_pcs[:, 1]

cost_brand["sustainability_index"] = StandardScaler().fit_transform(
    sustain_work[score_cols]
).mean(axis=1)

min_val = cost_brand["sustainability_index"].min()
max_val = cost_brand["sustainability_index"].max()
cost_brand["sustainability_score_relative"] = (
    (cost_brand["sustainability_index"] - min_val) / (max_val - min_val) * 100
)


# =========================
# LABOR BENCHMARK OVERLAY
# =========================
costs_wage = costs.merge(
    wages,
    left_on="Country",
    right_on="country",
    how="left"
)

drop_countries = ["USA", "UK", "Germany"]
costs_wage = costs_wage[~costs_wage["Country"].isin(drop_countries)].copy()

costs_wage["wage_gap_usd"] = costs_wage["Avg_Worker_Wage_USD"] - costs_wage["monthly_livable_usd"]
costs_wage["wage_ratio"] = costs_wage["Avg_Worker_Wage_USD"] / costs_wage["monthly_livable_usd"]
costs_wage["meets_living_wage"] = costs_wage["wage_ratio"] >= 1

labor_risk = costs_wage.groupby(["brand_clean", "Country"]).agg(
    avg_worker_wage=("Avg_Worker_Wage_USD", "mean"),
    livable_wage=("monthly_livable_usd", "mean"),
    wage_gap_usd=("wage_gap_usd", "mean"),
    wage_ratio=("wage_ratio", "mean"),
    meets_living_wage_rate=("meets_living_wage", "mean"),
).reset_index()

labor_risk = labor_risk.merge(brand_lookup, on="brand_clean", how="left").dropna(subset=["brand"])
labor_risk["meets_living_wage_rate"] *= 100

labor_summary = labor_risk.groupby("brand").agg(
    avg_wage_ratio=("wage_ratio", "mean"),
    avg_wage_gap_usd=("wage_gap_usd", "mean"),
    living_wage_compliance_rate=("meets_living_wage_rate", "mean"),
).reset_index()


# =========================
# EXPORTS
# =========================
brand_strategy.round(4).to_csv("brand_strategy_table.csv", index=False)
pca_loadings.round(4).to_csv("pca_loadings.csv", index=False)
product_return_risk.round(4).to_csv("product_return_risk.csv", index=False)
feature_importance.round(5).to_csv("feature_importance.csv", index=False)
cost_brand.round(4).to_csv("sustainability_overlay.csv", index=False)
labor_risk.round(4).to_csv("labor_risk.csv", index=False)
labor_summary.round(4).to_csv("labor_summary.csv", index=False)

frontend_data = {
    "brand_strategy": brand_strategy.round(4).replace({np.nan: None}).to_dict(orient="records"),
    "sustainability": cost_brand.round(4).replace({np.nan: None}).to_dict(orient="records"),
    "labor_summary": labor_summary.round(4).replace({np.nan: None}).to_dict(orient="records"),
    "model_metrics": model_metrics,
    "top_features": feature_importance.head(15).round(5).to_dict(orient="records"),
}

with open("frontend_data.json", "w") as f:
    json.dump(frontend_data, f, indent=2)

print("✅ Done. Exported:")
print("- brand_strategy_table.csv")
print("- product_return_risk.csv")
print("- sustainability_overlay.csv")
print("- labor_risk.csv")
print("- labor_summary.csv")
print("- frontend_data.json")
print("Model metrics:", model_metrics)