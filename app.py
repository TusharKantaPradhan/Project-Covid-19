import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# Page config
st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")

st.title("🌍 COVID-19 Analysis & Prediction Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("covid_19_clean_complete.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Sidebar
st.sidebar.header("Filters")
country = st.sidebar.selectbox("Select Country", df['Country/Region'].unique())

# Filter data
country_df = df[df['Country/Region'] == country]

# Aggregate
data = country_df.groupby('Date')[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()

# KPI Metrics
st.subheader(f"📊 {country} Overview")
col1, col2, col3 = st.columns(3)

col1.metric("Total Confirmed", int(data['Confirmed'].max()))
col2.metric("Total Deaths", int(data['Deaths'].max()))
col3.metric("Total Recovered", int(data['Recovered'].max()))

# Line Chart
st.subheader("📈 Trend Over Time")
fig = px.line(data, x='Date', y=['Confirmed', 'Deaths', 'Recovered'])
st.plotly_chart(fig, use_container_width=True)

# Daily cases
data['Daily Cases'] = data['Confirmed'].diff()

st.subheader("📅 Daily New Cases")
fig2 = px.line(data, x='Date', y='Daily Cases')
st.plotly_chart(fig2, use_container_width=True)

# Global Map
st.subheader("🌍 Global Spread Map")
latest = df[df['Date'] == df['Date'].max()]

fig_map = px.choropleth(
    latest,
    locations="Country/Region",
    locationmode="country names",
    color="Confirmed",
    hover_name="Country/Region",
    color_continuous_scale="Reds"
)

st.plotly_chart(fig_map, use_container_width=True)

# Prediction Section
st.subheader("🔮 Future Prediction (Next 7 Days)")

prophet_df = data[['Date', 'Confirmed']]
prophet_df.columns = ['ds', 'y']

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)

fig_forecast = px.line(forecast, x='ds', y='yhat', title="Predicted Cases")
st.plotly_chart(fig_forecast, use_container_width=True)

# Footer
st.write("📌 Built using Streamlit, Plotly & Prophet")