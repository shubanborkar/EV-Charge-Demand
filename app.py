import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime

# === Page Configuration (must be the first Streamlit command) ===
st.set_page_config(
    page_title="EV Adoption Forecaster",
    page_icon="⚡️",
    layout="wide"
)

# === New "Professional Dark" Theming via CSS ===
st.markdown("""
    <style>
        /* Main app background */
        .stApp {
            background-color: #0E1117; /* Dark background */
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: #161B22; /* Slightly lighter dark for sidebar */
        }

        /* Font colors */
        body, .stTextInput, .stSelectbox, .stMultiselect, .stNumberInput {
            color: #FAFAFA; /* Light gray for text */
        }

        /* Title and header colors */
        h1, h2, h3 {
            color: #3B82F6; /* A bright, professional blue */
        }
        
        /* Metric styling for a "card" look */
        .stMetric {
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 10px;
            padding: 15px;
        }

        /* Removing the default top padding for the main block container */
        .block-container {
            padding-top: 2rem;
        }

    </style>
""", unsafe_allow_html=True)


# === Model and Data Loading (with caching) ===
@st.cache_data
def load_data_and_model():
    """Loads the preprocessed data and the forecasting model."""
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    model = joblib.load('forecasting_ev_model.pkl')
    return df, model

df, model = load_data_and_model()
county_list = sorted(df['County'].dropna().unique().tolist())
FORECAST_HORIZON = 36 # 3 years

# === Core Forecasting Logic (Refactored into a function) ===
def generate_forecast(county_df, model, forecast_horizon):
    """
    Generates an EV adoption forecast for a given county.
    
    Args:
        county_df (pd.DataFrame): The historical data for a single county.
        model: The trained forecasting model.
        forecast_horizon (int): Number of months to forecast into the future.

    Returns:
        pd.DataFrame: A dataframe containing both historical and forecasted cumulative EV counts.
    """
    # Initialize variables from the last available data
    historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    cumulative_ev = list(np.cumsum(historical_ev))
    months_since_start = county_df['months_since_start'].max()
    latest_date = county_df['Date'].max()

    future_predictions = []

    # Iteratively predict for each future month
    for i in range(1, forecast_horizon + 1):
        # Create features for the model
        lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
        roll_mean = np.mean([lag1, lag2, lag3])
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        recent_cumulative = cumulative_ev[-6:]
        ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0
        
        feature_row = {
            'months_since_start': months_since_start + i,
            'county_encoded': county_df['county_encoded'].iloc[0],
            'ev_total_lag1': lag1, 'ev_total_lag2': lag2, 'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean, 'ev_total_pct_change_1': pct_change_1,
            'ev_total_pct_change_3': pct_change_3, 'ev_growth_slope': ev_growth_slope
        }

        # Predict and store the result
        prediction = model.predict(pd.DataFrame([feature_row]))[0]
        forecast_date = latest_date + pd.DateOffset(months=i)
        future_predictions.append({"Date": forecast_date, "Predicted EV Total": round(prediction)})

        # Update the history for the next iteration's feature engineering
        historical_ev.append(prediction)
        historical_ev.pop(0)
        cumulative_ev.append(cumulative_ev[-1] + prediction)
        cumulative_ev.pop(0)

    # Combine historical and forecasted data for plotting
    historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
    historical_cum['Source'] = 'Historical'
    historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

    forecast_df = pd.DataFrame(future_predictions)
    forecast_df['Source'] = 'Forecast'
    forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + (historical_cum['Cumulative EV'].iloc[-1] if not historical_cum.empty else 0)

    return pd.concat([historical_cum, forecast_df], ignore_index=True)


# === Sidebar for User Inputs ===
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Select an analysis mode and choose the counties to forecast.")

# App mode selection
app_mode = st.sidebar.selectbox("Choose Mode", ["Single County Forecast", "Compare Counties"])


# === Main Panel Display ===
st.markdown("<h1 style='text-align: center;'>Electric Vehicle Adoption Forecaster</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #D1D5DB;'>Analyzing EV trends in the State of Washington</p>", unsafe_allow_html=True)
st.image("professional_background.jpg", use_container_width=True) # Recommended to find a new professional/clean image


# === Single County Forecast Mode ===
if app_mode == "Single County Forecast":
    st.header("Single County Deep Dive")

    # New robust code
    # This block determines the default county to show
    default_index = 0  # Default to the first county in the list
    if "King" in county_list:
        default_index = county_list.index("King")

    # The selectbox should be in the sidebar, but controlled by this logic block
    county = st.sidebar.selectbox("Select a County", county_list, index=default_index)

    # This block runs only after a county has been selected
    if county:
        county_df = df[df['County'] == county].sort_values("Date")
        combined_df = generate_forecast(county_df, model, FORECAST_HORIZON)
        
        # --- Metrics and Plot Display ---
        st.subheader(f"📈 3-Year Forecast for {county} County")
        col1, col2, col3 = st.columns(3)
        
        historical_total = combined_df[combined_df['Source'] == 'Historical']['Cumulative EV'].iloc[-1]
        forecasted_total = combined_df['Cumulative EV'].iloc[-1]
        growth_pct = ((forecasted_total - historical_total) / historical_total) * 100 if historical_total > 0 else 0
        
        col1.metric(label="Current Cumulative EVs", value=f"{int(historical_total):,}")
        col2.metric(label="Forecasted EVs (in 3 Yrs)", value=f"{int(forecasted_total):,}")
        col3.metric(label="Projected Growth", value=f"{growth_pct:.2f}%")

        st.markdown("---")
        
        # --- Plotly Chart ---
        fig = px.line(combined_df, x='Date', y='Cumulative EV', color='Source',
                      title=f"Historical vs. Forecasted EV Adoption in {county}",
                      labels={'Cumulative EV': 'Cumulative EV Count', 'Date': 'Date'},
                      color_discrete_map={'Historical': '#3B82F6', 'Forecast': '#F97316'}, # Blue for historical, orange for forecast
                      markers=True)
        
        fig.update_layout(
            plot_bgcolor='#161B22', paper_bgcolor='#161B22', font_color='#FAFAFA',
            legend_title_text='', legend=dict(x=0.01, y=0.98),
            xaxis=dict(gridcolor='#30363D'),
            yaxis=dict(gridcolor='#30363D'),
        )
        st.plotly_chart(fig, use_container_width=True)
        
# === Compare Counties Mode ===
if app_mode == "Compare Counties":
    st.header("County Comparison")
    # New robust code
    # Define the list of counties you would ideally like to be the default
    ideal_defaults = ["King", "Snohomish", "Pierce"]

    # Create a new list containing only the ideal defaults that actually exist in your county_list
    valid_defaults = [county for county in ideal_defaults if county in county_list]

    # Use this "safe" list as the default for the widget
    multi_counties = st.sidebar.multiselect(
        "Select up to 3 counties",
        county_list,
        default=valid_defaults,
        max_selections=3
    )

    if multi_counties:
        comparison_data = []
        growth_summaries = {}
        
        for cty in multi_counties:
            cty_df = df[df['County'] == cty].sort_values("Date")
            combined_cty_df = generate_forecast(cty_df, model, FORECAST_HORIZON)
            combined_cty_df['County'] = cty
            comparison_data.append(combined_cty_df)

            # Calculate growth for summary
            hist_total = combined_cty_df[combined_cty_df['Source'] == 'Historical']['Cumulative EV'].iloc[-1]
            fc_total = combined_cty_df['Cumulative EV'].iloc[-1]
            growth_pct = ((fc_total - hist_total) / hist_total) * 100 if hist_total > 0 else 0
            growth_summaries[cty] = f"{growth_pct:.2f}%"

        comp_df = pd.concat(comparison_data, ignore_index=True)
        
        # --- Plot and Summary ---
        st.subheader("📊 Comparison of Cumulative EV Adoption Trends")
        fig = px.line(comp_df, x='Date', y='Cumulative EV', color='County',
                      title="EV Adoption Trends: Historical + 3-Year Forecast",
                      labels={'Cumulative EV': 'Cumulative EV Count', 'Date': 'Date'},
                      color_discrete_sequence=px.colors.qualitative.Vivid, # A color sequence that works well on dark
                      markers=False)
        
        fig.update_layout(
            plot_bgcolor='#161B22', paper_bgcolor='#161B22', font_color='#FAFAFA',
            legend_title_text='County',
            xaxis=dict(gridcolor='#30363D'),
            yaxis=dict(gridcolor='#30363D'),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecasted Growth Summary")
        with st.expander("Click to see the 3-year forecasted growth for each county"):
            for county, growth in growth_summaries.items():
                st.markdown(f"- **{county}:** {growth}")


st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Prepared for the AICTE Internship Cycle 2 by Shuban Borkar</p>", unsafe_allow_html=True)