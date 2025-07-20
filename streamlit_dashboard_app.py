# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

st.set_page_config(page_title="CO₂ Emissions Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("owid-co2-data.csv")
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    return df

df = load_data()

st.title("CO₂ Emissions Explorer")
st.markdown("Track and forecast CO₂ emissions by country, sector, and economic impact.")

# Sidebar filters
countries = df['country'].dropna().unique()
country_selection = st.sidebar.selectbox("Select Country", sorted(countries))
start_year, end_year = st.sidebar.slider("Select Year Range", 
                                         int(df['year'].dt.year.min()), 
                                         int(df['year'].dt.year.max()), 
                                         (1990, 2021))

filtered = df[(df['country'] == country_selection) & 
              (df['year'].dt.year >= start_year) & 
              (df['year'].dt.year <= end_year)]

# Trend Analysis
st.subheader("📈 CO₂ Emissions Over Time")
fig_trend = px.line(filtered, x="year", y="co2", title=f"Total CO₂ Emissions for {country_selection}")
st.plotly_chart(fig_trend, use_container_width=True)

# Sectoral CO₂ Comparison
st.subheader("🏭 Sectoral Emissions")
sectors = ['cement_co2', 'coal_co2', 'oil_co2', 'gas_co2']
fig_sectors = px.area(filtered, x='year', y=sectors, title=f"{country_selection} - Sectoral CO₂ Emissions")
st.plotly_chart(fig_sectors, use_container_width=True)

# Economic Correlation
st.subheader("📊 CO₂ vs GDP")
fig_corr = px.scatter(filtered, x='gdp', y='co2', size='population',
                      title=f"{country_selection} - CO₂ Emissions vs GDP",
                      labels={'gdp': 'GDP', 'co2': 'CO₂ Emissions'})
st.plotly_chart(fig_corr, use_container_width=True)

# Forecasting CO₂ Emissions
st.subheader("🔮 Forecasting Future CO₂ Emissions")

if filtered[['year', 'co2']].dropna().shape[0] > 2:
    forecast_df = filtered[['year', 'co2']].dropna().rename(columns={"year": "ds", "co2": "y"})
    model = Prophet()
    model.fit(forecast_df)
    future = model.make_future_dataframe(periods=10, freq='Y')
    forecast = model.predict(future)

    fig_forecast = px.line(forecast, x='ds', y='yhat', title=f"{country_selection} - CO₂ Forecast")
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.info("Not enough data to forecast CO₂ emissions for this country.")

# Raw Data Section
with st.expander("🔍 View Raw Data"):
    st.dataframe(filtered)

st.markdown("---")
st.markdown("🚀 Built with [Streamlit](https://streamlit.io) | 📊 Data: Our World In Data")
