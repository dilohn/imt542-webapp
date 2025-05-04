import sys
from streamlit import runtime
from streamlit.web import cli as stcli
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

DATA_URLS = {
    "Red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "White": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
}


@st.cache_data
def load_data(url):
    return pd.read_csv(url, sep=";")


def show_exploration(df, wine_type):
    st.write(f"### {wine_type} Wine Data ({df.shape[0]} samples)")
    st.dataframe(df.head())

    st.sidebar.header("Chart Type")
    plot_type = st.sidebar.selectbox(
        "Plot type", ["Histogram", "Scatter", "Correlation"]
    )
    if plot_type == "Histogram":
        feat = st.sidebar.selectbox("Feature", df.columns[:-1])
        bins = st.sidebar.slider("Bins", 5, 100, 20)
        fig, ax = plt.subplots()
        ax.hist(df[feat], bins=bins)
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif plot_type == "Scatter":
        x = st.sidebar.selectbox("X-axis", df.columns[:-1], index=0)
        y = st.sidebar.selectbox("Y-axis", df.columns[:-1], index=1)
        fig, ax = plt.subplots()
        sc = ax.scatter(df[x], df[y], c=df["quality"], cmap="viridis", alpha=0.6)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.colorbar(sc, ax=ax, label="Quality")
        st.pyplot(fig)

    else:
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)


def train_and_store_model(df, wine_type):
    X = df.drop("quality", axis=1)
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    # st.write(f"**R² score:** {r2_score(y_test, preds):.2f}")
    # st.write(f"**Mean Absolute Error:** {mean_absolute_error(y_test, preds):.2f}")
    st.session_state.model = model
    st.session_state["trained_wine_type"] = wine_type
    st.success("Adjust values below to predict quality.")


def prediction_ui(df, wine_type):
    st.header("Predict wine quality")

    if (
        "model" not in st.session_state
        or st.session_state.get("trained_wine_type") != wine_type
    ):
        if st.button("Train Model"):
            train_and_store_model(df, wine_type)
            st.rerun()
    else:
        inputs = {}
        for col in df.columns[:-1]:
            inputs[col] = st.number_input(
                label=col,
                min_value=float(df[col].min()),
                max_value=float(df[col].max()),
                value=float(df[col].mean()),
            )
        if st.button("Predict"):
            pred = st.session_state.model.predict(pd.DataFrame([inputs]))[0]
            st.success(f"Predicted Quality: {pred:.2f}")


def main():
    st.title("Explore Wine Quality and Predict")
    wine_type = st.sidebar.selectbox("Wine type", list(DATA_URLS.keys()))
    df = load_data(DATA_URLS[wine_type])

    if st.session_state.get("trained_wine_type") != wine_type:
        st.session_state.pop("model", None)
        st.session_state["trained_wine_type"] = wine_type

    show_exploration(df, wine_type)
    prediction_ui(df, wine_type)


if __name__ == "__main__":
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
