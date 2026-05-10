import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

# 2. Load the cleaned data
@st.cache_data
def load_data():
    # Loading the EDA version which contains your engineered features
    df = pd.read_csv("eda_india_housing_prices.csv")
    return df

df = load_data()

# 3. Sidebar for Navigation & Filtering
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Market Insights (EDA)", "Investment Logic"])

st.sidebar.header("Filter Properties")
selected_city = st.sidebar.multiselect("Select City", options=df['City'].unique(), default=df['City'].unique())
selected_type = st.sidebar.multiselect("Property Type", options=df['Property_Type'].unique(), default=df['Property_Type'].unique())

# Filtered Data
filtered_df = df[(df['City'].isin(selected_city)) & (df['Property_Type'].isin(selected_type))]

# 4. Main Page Content
if page == "Market Insights (EDA)":
    st.title("📊 Real Estate Market Insights")
    st.markdown("Explore trends and distributions across the Indian housing market.") 
    # Row 1: Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Properties", len(filtered_df))
    col2.metric("Avg Price (Lakhs)", f"₹{round(filtered_df['Price_in_Lakhs'].mean(), 2)}")
    col3.metric("Avg Price/SqFt", f"₹{round(filtered_df['Price_per_SqFt'].mean(), 2)}")

    # Row 2: Charts [cite: 50, 52, 146]
    st.subheader("Price Distribution by Property Type")
    fig_box = px.box(filtered_df, x="Property_Type", y="Price_in_Lakhs", color="Property_Type")
    st.plotly_chart(fig_box, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("City-wise Average Pricing") 
        city_avg = filtered_df.groupby('City')['Price_in_Lakhs'].mean().reset_index()
        fig_bar = px.bar(city_avg, x='City', y='Price_in_Lakhs')
        st.plotly_chart(fig_bar)

    with col_b:
        st.subheader("Size vs Price Relationship") 
        fig_scatter = px.scatter(filtered_df, x="Size_in_SqFt", y="Price_in_Lakhs", trendline="ols")
        st.plotly_chart(fig_scatter)

elif page == "Investment Logic":
    st.title("🏠 Rule-Based Investment Advisor")
    st.write("This section uses domain-based rules to identify potential 'Good Investments' and forecast future value.")

    if not filtered_df.empty:
        # 1. Define 'Good Investment' Logic
        # Calculate median for the filtered set to avoid index mismatch
        current_city_median = filtered_df.groupby('City')['Price_per_SqFt'].transform('median')
        
        # Rule: Price < median AND High Transport Accessibility
        recommendations = filtered_df[
            (filtered_df['Price_per_SqFt'].values < current_city_median.values) & 
            (filtered_df['Public_Transport_Accessibility'] == 'High')
        ].copy() # Using .copy() to avoid SettingWithCopy warnings

        if not recommendations.empty:
            st.success(f"Found {len(recommendations)} properties matching the 'Good Investment' criteria!")
            
            # 2. 5-Year Price Forecast Logic (Regression Baseline)
            st.subheader("📈 City-Specific 5-Year Price Forecast")
            
            growth_map = {
                'Bangalore': 0.12, 'Hyderabad': 0.12, 'Pune': 0.10,
                'Mumbai': 0.08, 'Delhi': 0.08, 'Chennai': 0.07, 'Kolkata': 0.06
            }

            def calculate_future_val(row):
                rate = growth_map.get(row['City'], 0.06) 
                return row['Price_in_Lakhs'] * ((1 + rate) ** 5)

            # Applying the forecast to the recommendations found
            recommendations['Projected_Price_2031'] = recommendations.apply(calculate_future_val, axis=1)
            recommendations['Estimated_Gain'] = recommendations['Projected_Price_2031'] - recommendations['Price_in_Lakhs']

            # Display the Final Result Table
            st.write("Projected returns based on regional appreciation trends:")
            st.dataframe(recommendations[[
                'City', 'Locality', 'Price_in_Lakhs', 'Projected_Price_2031', 'Estimated_Gain'
            ]].style.format({
                "Price_in_Lakhs": "₹{:.2f}L",
                "Projected_Price_2031": "₹{:.2f}L", 
                "Estimated_Gain": "₹{:.2f}L"
            }))
        else:
            st.warning("No properties match the 'Good Investment' criteria in this selection.")
    else:
        st.error("Please select at least one city in the sidebar to see recommendations.")