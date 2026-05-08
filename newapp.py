import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ======================================================
# PAGE CONFIG + STYLE
# ======================================================
st.set_page_config(page_title="Fashion Brand Intelligence", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #fff7fb 0%, #ffffff 45%, #fff1f6 100%);
}
[data-testid="stSidebar"] {
    background-color: #fff0f6;
}
.big-title {
    font-size: 46px;
    font-weight: 850;
    color: #2b2b2b;
    margin-bottom: 0px;
}
.subtitle {
    font-size: 18px;
    color: #606060;
    margin-bottom: 24px;
}
.note {
    background-color: #fff0f6;
    padding: 16px;
    border-radius: 14px;
    border-left: 6px solid #e75480;
    margin-bottom: 12px;
}
.good {
    background-color: #edf9f0;
    padding: 16px;
    border-radius: 14px;
    border-left: 6px solid #2ecc71;
    margin-bottom: 12px;
}
.warn {
    background-color: #fff8e8;
    padding: 16px;
    border-radius: 14px;
    border-left: 6px solid #f5a623;
    margin-bottom: 12px;
}
.bad {
    background-color: #ffecec;
    padding: 16px;
    border-radius: 14px;
    border-left: 6px solid #ff4d4d;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_outputs():
    brand = pd.read_csv("brand_strategy_table.csv")
    pca_loadings = pd.read_csv("pca_loadings.csv")
    product_risk = pd.read_csv("product_return_risk.csv")
    feature_importance = pd.read_csv("feature_importance.csv")
    sustain = pd.read_csv("sustainability_overlay.csv")
    labor_summary = pd.read_csv("labor_summary.csv")
    labor_risk = pd.read_csv("labor_risk.csv")
    products = pd.read_csv("fashion_boutique_dataset.csv")

    return brand, pca_loadings, product_risk, feature_importance, sustain, labor_summary, labor_risk, products


brand, pca_loadings, product_risk, feature_importance, sustain, labor_summary, labor_risk, products = load_outputs()


# ======================================================
# RETURN RISK MODEL
# ======================================================
@st.cache_resource
def train_return_model(products):
    df = products.copy()

    df["is_returned"] = df["is_returned"].astype(int)
    df["price_gap"] = df["original_price"] - df["current_price"]
    df["discount_flag"] = (df["markdown_percentage"] > 0).astype(int)

    model_cols = [
        "brand", "category", "season", "size", "color",
        "original_price", "current_price", "markdown_percentage",
        "price_gap", "discount_flag", "stock_quantity"
    ]

    df_model = df[model_cols + ["is_returned"]].dropna().copy()

    X = pd.get_dummies(df_model[model_cols], drop_first=False)
    y = df_model["is_returned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=450,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "auc": roc_auc_score(y_test, prob)
    }

    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return model, list(X.columns), metrics, importances


return_model, model_columns, model_metrics, live_importance = train_return_model(products)


# ======================================================
# HELPERS
# ======================================================
def prepare_model_matrix(df):
    model_cols = [
        "brand", "category", "season", "size", "color",
        "original_price", "current_price", "markdown_percentage",
        "price_gap", "discount_flag", "stock_quantity"
    ]

    X = pd.get_dummies(df[model_cols], drop_first=False)
    X = X.reindex(columns=model_columns, fill_value=0)
    return X


def simulate_return_risk(products, selected_brand, price_change, markdown_change, stock_change):
    baseline = products.copy()
    simulated = products.copy()

    for df in [baseline, simulated]:
        df["is_returned"] = df["is_returned"].astype(int)
        df["price_gap"] = df["original_price"] - df["current_price"]
        df["discount_flag"] = (df["markdown_percentage"] > 0).astype(int)

    mask = simulated["brand"] == selected_brand

    simulated.loc[mask, "current_price"] *= (1 + price_change)
    simulated.loc[mask, "original_price"] *= (1 + price_change)
    simulated.loc[mask, "markdown_percentage"] *= (1 + markdown_change)
    simulated.loc[mask, "stock_quantity"] *= (1 + stock_change)

    simulated["price_gap"] = simulated["original_price"] - simulated["current_price"]
    simulated["discount_flag"] = (simulated["markdown_percentage"] > 0).astype(int)

    X_base = prepare_model_matrix(baseline)
    X_sim = prepare_model_matrix(simulated)

    baseline["sim_return_risk"] = return_model.predict_proba(X_base)[:, 1]
    simulated["sim_return_risk"] = return_model.predict_proba(X_sim)[:, 1]

    return baseline, simulated


def build_strategy_map(brand_table):
    df = brand_table.copy()

    df["discount_dependency"] = df["avg_markdown"]
    df["customer_appeal"] = (
        df["avg_rating"] / df["avg_rating"].max() * 100
    ) * (1 - df["return_rate"])

    return df


def simulate_strategy_map(brand_table, selected_brand, price_change, markdown_change, stock_change):
    baseline = build_strategy_map(brand_table)
    simulated = build_strategy_map(brand_table)

    mask = simulated["brand"] == selected_brand

    simulated.loc[mask, "avg_price"] *= (1 + price_change)
    simulated.loc[mask, "avg_markdown"] *= (1 + markdown_change)
    simulated.loc[mask, "avg_stock"] *= (1 + stock_change)

    simulated["discount_dependency"] = simulated["avg_markdown"]

    # Directional PM logic: markdown pressure hurts appeal slightly, price premium can help slightly, inventory over-expansion hurts slightly.
    simulated.loc[mask, "customer_appeal"] = (
        simulated.loc[mask, "customer_appeal"]
        + (price_change * 8)
        - (markdown_change * 10)
        - (max(stock_change, 0) * 4)
    )

    simulated["customer_appeal"] = simulated["customer_appeal"].clip(0, 100)

    return baseline, simulated


def classify_strategy(row, median_discount, median_appeal):
    if row["discount_dependency"] >= median_discount and row["customer_appeal"] >= median_appeal:
        return "Mass Appeal / Discount-Led"
    if row["discount_dependency"] < median_discount and row["customer_appeal"] >= median_appeal:
        return "Premium / Healthy Demand"
    if row["discount_dependency"] >= median_discount and row["customer_appeal"] < median_appeal:
        return "High Discount / Weak Appeal"
    return "Low Appeal / Low Promotion"


def simulate_labor(labor_risk, selected_brand, wage_increase):
    base = labor_risk[labor_risk["brand"] == selected_brand].copy()
    sim = base.copy()

    if base.empty:
        return base, sim

    sim["avg_worker_wage"] = sim["avg_worker_wage"] * (1 + wage_increase)
    sim["wage_gap_usd"] = sim["avg_worker_wage"] - sim["livable_wage"]
    sim["wage_ratio"] = sim["avg_worker_wage"] / sim["livable_wage"]
    sim["meets_living_wage_rate"] = (sim["wage_ratio"] >= 1).astype(int) * 100

    return base, sim


def brand_snapshot_text(row):
    return (
        f"{row['brand']} has an average rating of {row['avg_rating']:.2f}, "
        f"a return rate of {row['return_rate'] * 100:.1f}%, "
        f"and an average markdown of {row['avg_markdown']:.1f}%."
    )


def assign_priority(row, product_risk, sustain, labor_summary):
    brand_name = row["brand"]

    avg_ml_risk = product_risk[
        product_risk["brand"] == brand_name
    ]["predicted_return_risk"].mean()

    return_risk_score = row["return_rate"] * 100
    discount_risk_score = row["avg_markdown"]
    ml_risk_score = avg_ml_risk * 100

    sustainability_risk = None
    if brand_name in sustain["brand"].values:
        s = sustain[sustain["brand"] == brand_name].iloc[0]
        sustainability_risk = 100 - s["sustainability_score_relative"]

    labor_risk_score = None
    if brand_name in labor_summary["brand"].values:
        l = labor_summary[labor_summary["brand"] == brand_name].iloc[0]
        labor_risk_score = 100 - l["living_wage_compliance_rate"]

    risk_components = [return_risk_score, discount_risk_score, ml_risk_score]

    if sustainability_risk is not None:
        risk_components.append(sustainability_risk)

    if labor_risk_score is not None:
        risk_components.append(labor_risk_score)

    overall_priority = sum(risk_components) / len(risk_components)

    if overall_priority >= 60:
        priority = "High Priority"
    elif overall_priority >= 40:
        priority = "Medium Priority"
    else:
        priority = "Low Priority"

    return pd.Series({
        "brand": brand_name,
        "Return Risk": return_risk_score,
        "Discount Risk": discount_risk_score,
        "ML Return Risk": ml_risk_score,
        "Sustainability Risk": sustainability_risk,
        "Labor Risk": labor_risk_score,
        "Overall Priority Score": overall_priority,
        "Priority": priority
    })


# ======================================================
# SIDEBAR
# ======================================================
selected_brand = st.sidebar.selectbox("Select brand", sorted(brand["brand"].unique()))

st.sidebar.markdown("### Commercial Strategy Sliders")
price_change = st.sidebar.slider("Price Change (%)", -30, 30, 0, 5) / 100
markdown_change = st.sidebar.slider("Markdown Change (%)", -50, 50, 0, 5) / 100
stock_change = st.sidebar.slider("Inventory Change (%)", -50, 50, 0, 5) / 100

st.sidebar.markdown("### Labor Strategy Slider")
wage_increase = st.sidebar.slider("Worker Wage Increase (%)", 0, 80, 0, 5) / 100


baseline_products, simulated_products = simulate_return_risk(
    products,
    selected_brand,
    price_change,
    markdown_change,
    stock_change
)

strategy_base, strategy_sim = simulate_strategy_map(
    brand,
    selected_brand,
    price_change,
    markdown_change,
    stock_change
)

median_discount = strategy_base["discount_dependency"].median()
median_appeal = strategy_base["customer_appeal"].median()

strategy_base["strategy_zone"] = strategy_base.apply(
    lambda x: classify_strategy(x, median_discount, median_appeal),
    axis=1
)

strategy_sim["strategy_zone"] = strategy_sim.apply(
    lambda x: classify_strategy(x, median_discount, median_appeal),
    axis=1
)

labor_base, labor_sim = simulate_labor(labor_risk, selected_brand, wage_increase)


# ======================================================
# HEADER
# ======================================================
st.markdown('<div class="big-title">Fashion Brand Intelligence Simulator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">An interactive tool to simulate how pricing, inventory, and labor decisions shape brand strategy, return risk, and sustainability outcomes.</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Strategy Map",
    "Return Risk Simulator",
    "Labor Wage Simulator",
    "Sustainability Lens",
    "Risk Quadrant",
    "Executive Recommendation"
])


# ======================================================
# TAB 1: STRATEGY MAP
# ======================================================
with tab1:
    st.subheader("Interactive Brand Strategy Map")

    current_row = strategy_base[strategy_base["brand"] == selected_brand].iloc[0]
    sim_row = strategy_sim[strategy_sim["brand"] == selected_brand].iloc[0]

    st.markdown(f"""
    <div class="note">
    <b>Brand snapshot:</b> {brand_snapshot_text(brand[brand["brand"] == selected_brand].iloc[0])}
    <br><br>
    Strategy zone moves from <b>{current_row["strategy_zone"]}</b> to <b>{sim_row["strategy_zone"]}</b>.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Zone", current_row["strategy_zone"])
    c2.metric("Simulated Zone", sim_row["strategy_zone"])
    c3.metric("Customer Appeal", f"{sim_row['customer_appeal']:.1f}", f"{sim_row['customer_appeal'] - current_row['customer_appeal']:.2f}")

    fig = go.Figure()

    # Quadrant backgrounds
    fig.add_shape(type="rect", x0=0, x1=median_discount, y0=median_appeal, y1=100,
                  fillcolor="rgba(187, 222, 251, 0.35)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=median_discount, x1=max(strategy_base["discount_dependency"].max(), strategy_sim["discount_dependency"].max()) + 8,
                  y0=median_appeal, y1=100,
                  fillcolor="rgba(255, 204, 128, 0.35)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=0, x1=median_discount, y0=0, y1=median_appeal,
                  fillcolor="rgba(200, 230, 201, 0.35)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=median_discount, x1=max(strategy_base["discount_dependency"].max(), strategy_sim["discount_dependency"].max()) + 8,
                  y0=0, y1=median_appeal,
                  fillcolor="rgba(255, 205, 210, 0.35)", line=dict(width=0), layer="below")

    # Quadrant labels
    fig.add_annotation(x=median_discount / 2, y=96, text="Premium / Healthy Demand", showarrow=False, font=dict(size=13))
    fig.add_annotation(x=median_discount + 5, y=96, text="Mass Appeal / Discount-Led", showarrow=False, font=dict(size=13))
    fig.add_annotation(x=median_discount / 2, y=5, text="Low Appeal / Low Promotion", showarrow=False, font=dict(size=13))
    fig.add_annotation(x=median_discount + 5, y=5, text="High Discount / Weak Appeal", showarrow=False, font=dict(size=13))

    # Baseline brands as clean black points
    fig.add_trace(go.Scatter(
        x=strategy_base["discount_dependency"],
        y=strategy_base["customer_appeal"],
        mode="markers+text",
        text=strategy_base["brand"],
        textposition="top center",
        marker=dict(size=11, color="black"),
        name="Brands",
        hovertemplate="<b>%{text}</b><br>Discount Dependency=%{x:.1f}<br>Customer Appeal=%{y:.1f}<extra></extra>"
    ))

    # Selected baseline
    fig.add_trace(go.Scatter(
        x=[current_row["discount_dependency"]],
        y=[current_row["customer_appeal"]],
        mode="markers",
        marker=dict(size=18, color="#e75480", symbol="circle"),
        name="Selected baseline",
        hovertemplate="Selected baseline<extra></extra>"
    ))

    # Selected simulated
    fig.add_trace(go.Scatter(
        x=[sim_row["discount_dependency"]],
        y=[sim_row["customer_appeal"]],
        mode="markers",
        marker=dict(size=20, color="#7e57c2", symbol="star"),
        name="Selected simulated",
        hovertemplate="Selected simulated<extra></extra>"
    ))

    # Arrow
    fig.add_annotation(
        x=sim_row["discount_dependency"],
        y=sim_row["customer_appeal"],
        ax=current_row["discount_dependency"],
        ay=current_row["customer_appeal"],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.4,
        arrowwidth=2.2,
        arrowcolor="#333333"
    )

    fig.add_vline(x=median_discount, line_width=1, line_dash="dash", line_color="gray")
    fig.add_hline(y=median_appeal, line_width=1, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Brand Strategy Map: Discount Dependency vs Customer Appeal",
        xaxis_title="Discount Dependency",
        yaxis_title="Customer Appeal",
        height=650,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=70, b=90)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Before vs After Movement")

    movement_df = pd.DataFrame({
        "State": ["Baseline", "Simulated"],
        "Discount Dependency": [current_row["discount_dependency"], sim_row["discount_dependency"]],
        "Customer Appeal": [current_row["customer_appeal"], sim_row["customer_appeal"]],
        "Strategy Zone": [current_row["strategy_zone"], sim_row["strategy_zone"]]
    })

    st.dataframe(movement_df.round(3), use_container_width=True)

    st.markdown("### PCA Methodology Layer")

    st.markdown("""
    <div class="note">
    The main map uses interpretable axes for storytelling. PCA is still used as the technical layer to validate brand positioning from the original feature space: price, markdown, rating, return rate, stock, product count, and discount behavior.
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(pca_loadings.round(3), use_container_width=True)


# ======================================================
# TAB 2: RETURN RISK SIMULATOR
# ======================================================
with tab2:
    st.subheader("Return Risk Simulator")

    base_brand = baseline_products[baseline_products["brand"] == selected_brand]
    sim_brand = simulated_products[simulated_products["brand"] == selected_brand]

    base_risk = base_brand["sim_return_risk"].mean()
    sim_risk = sim_brand["sim_return_risk"].mean()
    risk_delta = sim_risk - base_risk

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Return Risk", f"{base_risk * 100:.1f}%")
    c2.metric("Simulated Return Risk", f"{sim_risk * 100:.1f}%", f"{risk_delta * 100:.2f}%")
    c3.metric("Products Simulated", len(sim_brand))

    if risk_delta < -0.005:
        st.markdown('<div class="good"><b>Improvement:</b> This scenario lowers predicted return risk.</div>', unsafe_allow_html=True)
    elif risk_delta > 0.005:
        st.markdown('<div class="warn"><b>Risk increase:</b> This scenario raises predicted return risk.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="note"><b>Neutral:</b> This scenario has limited predicted impact on return risk.</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(base_brand["sim_return_risk"], bins=20, alpha=0.50, label="Baseline")
    ax.hist(sim_brand["sim_return_risk"], bins=20, alpha=0.50, label="Simulated")
    ax.set_xlabel("Predicted Return Risk")
    ax.set_ylabel("Number of Products")
    ax.set_title(f"{selected_brand}: Product Return Risk Distribution")
    ax.legend()
    st.pyplot(fig)

    display_sim = sim_brand.copy()
    display_sim["risk_change"] = display_sim["sim_return_risk"] - base_brand["sim_return_risk"].values

    st.markdown("### Highest-Risk Products After Simulation")
    st.dataframe(
        display_sim[
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
                "sim_return_risk",
                "risk_change"
            ]
        ].sort_values("sim_return_risk", ascending=False).head(15).round(3),
        use_container_width=True
    )

    st.markdown("### Random Forest Feature Importance")
    top_features = live_importance.head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_features["feature"], top_features["importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top Return-Risk Drivers")
    st.pyplot(fig)

    st.dataframe(top_features.round(4), use_container_width=True)


# ======================================================
# TAB 3: LABOR WAGE SIMULATOR
# ======================================================
with tab3:
    st.subheader("Labor Wage Gap Simulator")

    if labor_base.empty:
        st.warning("Labor benchmark data is only available for matched fast-fashion brands.")
    else:
        base_compliance = labor_base["meets_living_wage_rate"].mean()
        sim_compliance = labor_sim["meets_living_wage_rate"].mean()

        base_avg_ratio = labor_base["wage_ratio"].mean()
        sim_avg_ratio = labor_sim["wage_ratio"].mean()

        base_avg_gap = labor_base["wage_gap_usd"].mean()
        sim_avg_gap = labor_sim["wage_gap_usd"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Wage Gap", f"${sim_avg_gap:,.0f}", f"${sim_avg_gap - base_avg_gap:,.0f}")
        c2.metric("Avg Wage Ratio", f"{sim_avg_ratio:.2f}", f"{sim_avg_ratio - base_avg_ratio:.2f}")
        c3.metric("Living Wage Compliance", f"{sim_compliance:.1f}%", f"{sim_compliance - base_compliance:.1f}%")
        c4.metric("Countries Benchmarked", len(labor_sim))

        st.markdown(f"""
        <div class="note">
        <b>Average wage gap:</b> Under the current simulation, {selected_brand}'s average worker wage moves from
        <b>${base_avg_gap:,.0f}</b> relative to the living wage benchmark to <b>${sim_avg_gap:,.0f}</b>.
        Negative values mean workers are below the living wage benchmark; positive values mean they exceed it.
        </div>
        """, unsafe_allow_html=True)

        x = np.arange(len(labor_base["Country"]))
        width = 0.38

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, labor_base["wage_ratio"], width, label="Baseline")
        ax.bar(x + width / 2, labor_sim["wage_ratio"], width, label="After Wage Increase")
        ax.axhline(1, linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labor_base["Country"], rotation=45)
        ax.set_ylabel("Wage Ratio")
        ax.set_title(f"{selected_brand}: Wage Ratio by Production Country")
        ax.legend()
        st.pyplot(fig)

        wage_display = labor_base[[
            "brand",
            "Country",
            "avg_worker_wage",
            "livable_wage",
            "wage_gap_usd",
            "wage_ratio",
            "meets_living_wage_rate"
        ]].copy()

        wage_display = wage_display.rename(columns={
            "avg_worker_wage": "Baseline Worker Wage",
            "livable_wage": "Living Wage Benchmark",
            "wage_gap_usd": "Baseline Wage Gap",
            "wage_ratio": "Baseline Wage Ratio",
            "meets_living_wage_rate": "Baseline Compliance"
        })

        wage_display["Simulated Worker Wage"] = labor_sim["avg_worker_wage"].values
        wage_display["Simulated Wage Gap"] = labor_sim["wage_gap_usd"].values
        wage_display["Simulated Wage Ratio"] = labor_sim["wage_ratio"].values
        wage_display["Simulated Compliance"] = labor_sim["meets_living_wage_rate"].values

        st.dataframe(wage_display.round(3), use_container_width=True)


# ======================================================
# TAB 4: SUSTAINABILITY LENS
# ======================================================
with tab4:
    st.subheader("Sustainability Lens")

    if selected_brand not in sustain["brand"].values:
        st.warning("This brand does not have matched sustainability data. Sustainability overlay is available for matched fast-fashion brands only.")
    else:
        s = sustain[sustain["brand"] == selected_brand].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Relative Sustainability Score", round(s["sustainability_score_relative"], 1))
        c2.metric("Ethical Rating", round(s["ethical_rating"], 1))
        c3.metric("Transparency", round(s["transparency"], 1))

        st.markdown("""
        <div class="note">
        Relative sustainability is scaled from 0 to 100 across the matched brands. A score of 0 means lowest within this comparison group, not zero sustainability overall.
        Ethical rating and transparency are taken from the source dataset and shown as supporting indicators.
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(sustain["production"], sustain["sustainability_score_relative"], s=120, alpha=0.65)

        selected_s = sustain[sustain["brand"] == selected_brand].iloc[0]
        ax.scatter(selected_s["production"], selected_s["sustainability_score_relative"], s=300, marker="*", label=selected_brand)

        for _, r in sustain.iterrows():
            ax.text(r["production"], r["sustainability_score_relative"], r["brand"], fontsize=9)

        ax.set_xlabel("Production Scale")
        ax.set_ylabel("Relative Sustainability")
        ax.set_title("Production Scale vs Sustainability")
        ax.grid(alpha=0.25)
        ax.legend()
        st.pyplot(fig)

        st.dataframe(
            sustain[
                [
                    "brand",
                    "production",
                    "release_cycles",
                    "emissions",
                    "water",
                    "waste",
                    "worker_wage",
                    "sustainability_score_relative",
                    "ethical_rating",
                    "transparency",
                    "compliance"
                ]
            ].round(3),
            use_container_width=True
        )


# ======================================================
# TAB 5: RISK QUADRANT
# ======================================================
with tab5:
    st.subheader("Risk Quadrant")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        brand["avg_markdown"],
        brand["return_rate"] * 100,
        s=brand["product_count"] * 3,
        alpha=0.65
    )

    selected = brand[brand["brand"] == selected_brand].iloc[0]

    ax.scatter(
        selected["avg_markdown"],
        selected["return_rate"] * 100,
        s=300,
        marker="*",
        label=selected_brand
    )

    median_markdown = brand["avg_markdown"].median()
    median_return = brand["return_rate"].median() * 100

    ax.axvline(median_markdown, linestyle="--", linewidth=1)
    ax.axhline(median_return, linestyle="--", linewidth=1)

    for _, r in brand.iterrows():
        ax.text(r["avg_markdown"] + 0.2, r["return_rate"] * 100 + 0.2, r["brand"], fontsize=9)

    ax.set_xlabel("Average Markdown (%)")
    ax.set_ylabel("Return Rate (%)")
    ax.set_title("Markdown Dependence vs Return Risk")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)

    priority_table = brand.apply(
        lambda x: assign_priority(x, product_risk, sustain, labor_summary),
        axis=1
    )

    matched_brands = set(sustain["brand"].dropna()).intersection(set(labor_summary["brand"].dropna()))
    fast_fashion_priority = priority_table[priority_table["brand"].isin(matched_brands)].copy()
    other_priority = priority_table[~priority_table["brand"].isin(matched_brands)].copy()

    st.markdown("### Full Priority Score: Brands With Sustainability + Labor Data")
    st.markdown("""
    <div class="note">
    This table includes commercial risk, ML return risk, sustainability risk, and labor risk.
    Sustainability and labor scores are only included for brands with matched ethical-impact data.
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(
        fast_fashion_priority.sort_values("Overall Priority Score", ascending=False).round(2),
        use_container_width=True
    )

    st.markdown("### Commercial Priority Score: Brands Without Sustainability + Labor Data")
    st.markdown("""
    <div class="note">
    These brands did not have matched sustainability or labor data, so their priority score uses only return risk,
    discount risk, and ML-predicted return risk.
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(
        other_priority[
            ["brand", "Return Risk", "Discount Risk", "ML Return Risk", "Overall Priority Score", "Priority"]
        ].sort_values("Overall Priority Score", ascending=False).round(2),
        use_container_width=True
    )


# ======================================================
# TAB 6: EXECUTIVE RECOMMENDATION
# ======================================================
with tab6:
    st.subheader("Executive Recommendation")

    r = brand[brand["brand"] == selected_brand].iloc[0]
    selected_risk_mean = sim_brand["sim_return_risk"].mean()

    recs = []

    if current_row["strategy_zone"] != sim_row["strategy_zone"]:
        recs.append(
            f"Your selected strategy shifts {selected_brand} from '{current_row['strategy_zone']}' toward '{sim_row['strategy_zone']}'."
        )

    if r["return_rate"] > brand["return_rate"].median():
        recs.append("Return rate is above peer median. Prioritize fit, quality, and category-level return reduction.")

    if r["avg_markdown"] > brand["avg_markdown"].median():
        recs.append("Markdown dependence is above peer median. Test pricing discipline before expanding inventory.")

    if selected_risk_mean > baseline_products["sim_return_risk"].mean():
        recs.append("ML-predicted return risk is above dataset average. Review high-risk product patterns before growth.")

    if risk_delta > 0.005:
        recs.append("Current commercial simulation increases predicted return risk. Reconsider aggressive pricing, markdown, or inventory changes.")
    elif risk_delta < -0.005:
        recs.append("Current commercial simulation reduces predicted return risk. This scenario may be operationally safer.")

    if not labor_base.empty:
        if sim_compliance > base_compliance:
            recs.append(f"Labor investment improves living wage compliance by {sim_compliance - base_compliance:.1f} percentage points.")
        elif base_compliance < 50:
            recs.append("Labor benchmark risk is high. Growth strategy should account for wage compliance gaps.")

    if selected_brand in sustain["brand"].values:
        s = sustain[sustain["brand"] == selected_brand].iloc[0]
        if s["sustainability_score_relative"] < 50:
            recs.append("Sustainability risk is elevated. Pair commercial moves with emissions, waste, or transparency improvements.")

    if not recs:
        recs.append("No major risk flags detected. Brand appears relatively balanced across commercial, return-risk, labor, and sustainability dimensions.")

    for rec in recs:
        st.markdown(f"""
        <div class="note">
        <b>{selected_brand}:</b> {rec}
        </div>
        """, unsafe_allow_html=True)