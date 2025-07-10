import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# App title
st.title("🛢️ Brent Crude Oil Forecast Tool")
st.markdown("Forecast **Average Price**, **Global Demand**, and **Global Supply** for future years.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Clean Excel File", type=["xlsx"])

# Year range selector
n_years = st.slider("📆 Select number of future years to predict", min_value=1, max_value=20, value=10)

if uploaded_file:
    try:
        # Read Excel and create datetime column
        df = pd.read_excel(uploaded_file)
        df['ds'] = pd.to_datetime(df['Year'].astype(str) + '-01-01')

        # ----------- PRICE FORECAST -----------
        st.subheader("📊 Forecast: Avg Price ($/bbl)")
        price_df = df[['ds', 'Avg Price ($/bbl)']].rename(columns={'Avg Price ($/bbl)': 'y'})
        price_model = Prophet()
        price_model.fit(price_df)
        future = price_model.make_future_dataframe(periods=n_years, freq='Y')
        price_forecast = price_model.predict(future)
        st.pyplot(price_model.plot(price_forecast))

        # ----------- DEMAND FORECAST -----------
        st.subheader("📈 Forecast: Global Demand (mb/d)")
        demand_df = df[['ds', 'Global Demand (mb/d)']].rename(columns={'Global Demand (mb/d)': 'y'})
        demand_model = Prophet()
        demand_model.fit(demand_df)
        demand_forecast = demand_model.predict(future)
        st.pyplot(demand_model.plot(demand_forecast))

        # ----------- SUPPLY FORECAST -----------
        st.subheader("📉 Forecast: Global Supply (mb/d)")
        supply_df = df[['ds', 'Global Supply (mb/d)']].rename(columns={'Global Supply (mb/d)': 'y'})
        supply_model = Prophet()
        supply_model.fit(supply_df)
        supply_forecast = supply_model.predict(future)
        st.pyplot(supply_model.plot(supply_forecast))

        # ----------- COMPARISON CHART -----------
        st.subheader("📈 Comparison: Price vs Demand vs Supply")
        comparison_df = pd.DataFrame({
            'Year': future['ds'].dt.year,
            'Avg Price ($/bbl)': price_forecast['yhat'],
            'Global Demand (mb/d)': demand_forecast['yhat'],
            'Global Supply (mb/d)': supply_forecast['yhat']
        })

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(comparison_df['Year'], comparison_df['Avg Price ($/bbl)'], label='Price ($/bbl)', color='blue')
        ax.plot(comparison_df['Year'], comparison_df['Global Demand (mb/d)'], label='Demand (mb/d)', color='green', linestyle='--')
        ax.plot(comparison_df['Year'], comparison_df['Global Supply (mb/d)'], label='Supply (mb/d)', color='red', linestyle=':')
        ax.set_title("Comparison Forecast")
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # ----------- DOWNLOAD FORECAST -----------
        st.subheader("⬇️ Download Forecast CSV")
        st.download_button(
            label="Download Data as CSV",
            data=comparison_df.to_csv(index=False),
            file_name="brent_forecast_comparison.csv",
            mime="text/csv"
        )

        st.success("✅ Forecast and comparison completed successfully!")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("👈 Upload your clean Excel file with columns: Year, Avg Price ($/bbl), Global Demand (mb/d), Global Supply (mb/d)")
