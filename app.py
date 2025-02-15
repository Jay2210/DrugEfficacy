import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# If you have these packages:
import shap
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ===============================
# LOAD & PREPARE THE DATA
# ===============================
file_path = "UNMATCHED_PATIENTS.csv"
df = pd.read_csv(file_path)

# Remove columns with all null values
df = df.dropna(axis=1, how='all')

# Map drug_s: 0 → 'drug_d', 1 → 'drug_s'
df['drug_s'] = df['drug_s'].map({0: 'drug_d', 1: 'drug_s'})

# Set the page config (optional)
st.set_page_config(page_title="Two-Page EDA & Modeling", layout="wide")

# Define a custom color palette
color_palette = [
    '#636EFA',  # 0 - Blue
    '#EF553B',  # 1 - Red/Orange
    '#00CC96',  # 2 - Green
    '#AB63FA',  # 3 - Purple
    '#FFA15A',  # 4 - Peach
    '#19D3F3',  # 5 - Light Blue
    '#FF6692'   # 6 - Pink
]

# ==========================================
# CREATE TABS: EDA (Page 1) & MODELING (Page 2)
# ==========================================
st.title("Two-Page App: EDA and SMOTE + Modeling")

tab1, tab2 = st.tabs(["EDA Page", "SMOTE & Modeling Page"])

# ==========================================
# TAB 1: EDA PAGE
# ==========================================
with tab1:
    st.header("Page 1: Exploratory Data Analysis (EDA)")

    st.write("Here, we focus on **comorbidities**, **charges**, and **exacerbations** "
             "across different drug groups.")

    # 1) Comorbidity Distribution
    st.subheader("Comorbidities by Drug Usage")
    comorbidities = [
        'pneumonia', 'sinusitis', 'acute_bronchitis',
        'acute_laryngitis', 'upper_respiratory_infection',
        'gerd', 'rhinitis'
    ]
    drug_comorbidities = df.groupby('drug_s')[comorbidities].sum().T
    fig_comorb = px.bar(
        drug_comorbidities,
        barmode='group',
        title="Comorbidities by Drug Usage",
        color_discrete_sequence=color_palette[:2]
    )
    st.plotly_chart(fig_comorb)

    # 2) Distribution of Charges by Drug Usage
    st.subheader("Distribution of Charges by Drug Usage")

    fig_log = px.box(
        df, x='drug_s', y='log_charges',
        title="Log Charges by Drug Usage",
        color_discrete_sequence=[color_palette[3]]
    )
    st.plotly_chart(fig_log)

    fig_pre = px.box(
        df, x='drug_s', y='total_pre_index_charge',
        title="Total Pre-Index Charge by Drug Usage",
        color_discrete_sequence=[color_palette[5]]
    )
    st.plotly_chart(fig_pre)

    fig_asthma_charge = px.box(
        df, x='drug_s', y='log_asthma_charge',
        title="Log Asthma Charge by Drug Usage",
        color_discrete_sequence=[color_palette[6]]
    )
    st.plotly_chart(fig_asthma_charge)

    # 3) Exacerbations Analysis
    st.subheader("Exacerbations Analysis by Drug Usage")
    fig_exac_hist = px.histogram(
        df, x='post_index_exacerbations365', color='drug_s',
        nbins=20,
        title="Post-Index Exacerbations Distribution by Drug Usage",
        color_discrete_sequence=color_palette
    )
    st.plotly_chart(fig_exac_hist)

    fig_exac_box = px.box(
        df, x='drug_s', y='post_index_exacerbations365',
        title="Exacerbations by Drug Usage",
        color_discrete_sequence=[color_palette[2]]
    )
    st.plotly_chart(fig_exac_box)

    st.write("### EDA Insights:")
    st.write("- **Drug usage** significantly affects comorbidities and medical charges.")
    st.write("- Some drug groups (e.g., `'drug_s'`) might correlate with higher post-index exacerbations.")
    st.write("- **Financial burdens** (charges) vary among different drug treatments.")


# ==========================================
# TAB 2: SMOTE & MODELING PAGE
# ==========================================
with tab2:
    st.header("Page 2: SMOTE Balancing & Propensity Score Estimation")

    st.write("This page demonstrates **SMOTE** to handle class imbalance and **XGBoost** "
             "to estimate **propensity scores** and feature importance using **SHAP**.")

    # Define covariates for modeling
    covariates = [
        "index_age", "previous_asthma_drugs", "pneumonia", "sinusitis",
        "acute_bronchitis", "acute_laryngitis", "upper_respiratory_infection",
        "gerd", "rhinitis", "female", "adherence", "total_pre_index_charge",
        "pre_asthma_days", "pre_asthma_charge", "pre_asthma_pharma_charge"
    ]

    # Prepare X, y
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[covariates])

    y = df["drug_s"].map({'drug_d': 0, 'drug_s': 1})

    # Train XGBoost model
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_scaled, y)

    # SHAP Feature Importance (If SHAP installed)
    st.subheader("SHAP Feature Importance")
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X_scaled)
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        "Feature": covariates,
        "Importance": shap_importance
    }).sort_values(by="Importance", ascending=False)

    fig_shap = px.bar(
        feature_importance_df, x="Importance", y="Feature",
        orientation='h',
        title="Feature Importance (SHAP)",
        color_discrete_sequence=[color_palette[2]]
    )
    st.plotly_chart(fig_shap)

    # SMOTE balancing
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Class Distribution Before & After SMOTE
    st.subheader("Class Distribution Before & After SMOTE")
    fig_before = px.histogram(
        y,
        title="Class Distribution Before SMOTE",
        color_discrete_sequence=[color_palette[0]]
    )
    st.plotly_chart(fig_before)

    fig_after = px.histogram(
        y_resampled,
        title="Class Distribution After SMOTE",
        color_discrete_sequence=[color_palette[1]]
    )
    st.plotly_chart(fig_after)

    # Train-test split on balanced data
    X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # Retrain XGB after SMOTE
    xgb_model.fit(X_train_bal, y_train_bal)

    # Predict probabilities
    y_pred_probs = xgb_model.predict_proba(X_test_bal)[:, 1]

    # Precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test_bal, y_pred_probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)

    # Align lengths (sometimes there's an off-by-one mismatch)
    min_len = min(len(thresholds), len(f1_scores))
    thresholds = thresholds[:min_len]
    f1_scores = f1_scores[:min_len]

    st.subheader("Optimal Classification Threshold Based on F1 Score")
    fig_thr = px.line(
        x=thresholds, y=f1_scores,
        labels={'x': 'Threshold', 'y': 'F1 Score'},
        title="Optimal Classification Threshold",
        color_discrete_sequence=[color_palette[4]]
    )
    st.plotly_chart(fig_thr)

    # Compute & store propensity scores
    df["propensity_score_xgb"] = xgb_model.predict_proba(X_scaled)[:, 1]
    df["iptw_weight_xgb"] = (
        (y / df["propensity_score_xgb"]) +
        ((1 - y) / (1 - df["propensity_score_xgb"]))
    )

    # Propensity Score Distribution
    st.subheader("Propensity Score Distribution")
    if "propensity_score_xgb" in df.columns:
        fig_prop = px.histogram(
            df,
            x="propensity_score_xgb",
            title="Propensity Score Distribution",
            color_discrete_sequence=[color_palette[3]]
        )
        st.plotly_chart(fig_prop)
    else:
        st.warning("No 'propensity_score_xgb' found. Check if the model has been fit properly.")

    # Meaningful Insights
    st.subheader("Modeling Insights")
    st.write("- **SHAP** analysis shows which covariates most strongly predict drug assignment.")
    st.write("- **SMOTE** effectively balances the classes, potentially reducing model bias.")
    st.write("- The **F1 score** helps identify an optimal threshold for classification, rather than default 0.5.")
    st.write("- **Propensity scores** (and IPTW weights) allow for causal interpretation in further analyses.")
