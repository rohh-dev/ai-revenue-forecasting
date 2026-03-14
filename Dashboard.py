from tensorflow.keras import models
import streamlit as st
from streamlit_option_menu import option_menu
import os
import pandas as pd
import plotly.express as px
import hydralit_components as hc
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input,LSTM, Dense
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide",initial_sidebar_state='auto')

user_data_file = "user_data.csv"

# create a file if not already
if not os.path.exists(user_data_file):
    df = pd.DataFrame(columns=["Username","Password"])
    df.to_csv(user_data_file,index=False)
else:
    df = pd.read_csv(user_data_file,dtype={'Username':str,'Password':str})

# ensure all values are treated as strings and remove accidental space
df["Username"] = df["Username"].astype(str).str.strip()
df["Password"] = df["Password"].astype(str).str.strip()

# set up authentication and user_id initially for default
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# check state of session --- from URL
query_params = st.query_params
if "auth" in query_params and query_params["auth"] == "true":
    st.session_state.authenticated = True
    st.session_state.user_id = query_params.get("user","")

with st.sidebar:
    if not st.session_state.authenticated:
        selected = option_menu("AI-Driven Revenue Forecasting and Trend Analysis for Business Growth", ["Register", "Login"],
                               default_index=0, icons=["person", "lock"])
        menu_id = False
    else:
        selected = False

if selected == "Register":
    st.header("Register Now")
    reg_username = st.text_input("Enter Username").strip()
    reg_password = st.text_input("Enter Password",type="password").strip()
    reg_button = st.button("Register")

    if reg_button:
        if reg_username and reg_password:
            if reg_username in df["Username"].values:
                st.error("Username Already Exists. Please Login")
            else:
                new_entry = pd.DataFrame({"Username":[reg_username],"Password":[reg_password]})
                df = pd.concat([df,new_entry],ignore_index=True)
                df.to_csv(user_data_file,index=False)
                st.success("Registration successful! You can now login")
        else:
            st.error("Please enter both username and password")

if selected == "Login":
    st.header("Login Now")
    username = st.text_input("Enter Username")
    password = st.text_input("Enter Password",type="password")
    login_button = st.button("Login")

    if login_button:
        username = str(username).strip()
        password = str(password).strip()

        user = df[(df["Username"] == username) & (df["Password"] == password)]

        if not user.empty:
            st.session_state.authenticated = True
            st.session_state.user_id = username
            st.query_params.update(auth="true",user=username)
            st.success("Login Successfully")
            st.rerun()

        else:
            st.error("Invalid Username or Password")


if st.session_state.authenticated:

    df = pd.read_csv("sample_sales_data.csv")

    menu_data = [
        {'icon': "bi bi-info-circle", 'label': "Overview"},
        {'icon': "bi bi-bar-chart-fill", 'label': "Sales Performance"},
        {'icon': "bi bi-activity", 'label': "Customer Insights"},
        {'icon': "far fa-chart-bar", 'label': "Revenue Forecasting"},
        {'icon': "bi bi-search", 'label': "Custom Filter"},
        {'icon': "bi bi-box-arrow-right", 'label': "Logout"},
    ]

    over_theme = {'txc_inactive': '#FFFFFF'}
    menu_id = hc.nav_bar(
        menu_definition=menu_data,
        override_theme=over_theme,
        hide_streamlit_markers=False,
        sticky_nav=True,
        sticky_mode='pinned',
    )

if menu_id == "Overview":

    revenue = int(sum(df["Amount"]))
    orders = len(df["Order ID"].unique())
    AOV = revenue/orders

    top_categories = df.groupby("Category")["Amount"].sum().sort_values(ascending=False).head(5)
    top_sku = df.groupby("SKU")["Amount"].sum().sort_values(ascending=False).head(10)
    df["Date"] = pd.to_datetime(df["Date"])
    revenue_trend = df.groupby("Date")["Amount"].sum()

    filter_df = {
        "category": top_categories.index,
        "values": top_categories.values,
    }

    filter_df2 = {
        "sku": top_sku.index,
        "amount": top_sku.values
    }

    col1, col2, col3, col4,col5 = st.columns(5)

    col1.metric(label="Revenue", value=f"₹{revenue}", delta="+8%")
    col2.metric(label="Orders", value=f"{orders}", delta="-2%")
    col3.metric(label="Average Order Value", value=f"{AOV:.2f}", delta="+0.5%")
    col4.metric(label="Top Selling Category",value=top_categories.index[0])
    col5.metric(label="Top Selling Product", value=top_sku.index[0])

    #REVENUE TRENDS OVER TIME
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=revenue_trend.index, y=revenue_trend.values, mode="lines", name="Close Price"))

    fig.update_layout(title=f"Revenue Trends Over Time", xaxis_title="Date", yaxis_title="Price",
                      template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


    #TOP SELLING CATEGORIES AND SKU
    col1, col2 = st.columns(2)

    with col1:
        fig_exp = px.bar(filter_df, x="category", y="values", color="category", title="Top Selling Categories")
        st.plotly_chart(fig_exp, use_container_width=True)
    with col2:
        fig_exp = px.bar(filter_df2, x="sku", y="amount", color="amount", title="Top Selling SKU")
        st.plotly_chart(fig_exp, use_container_width=True)


if menu_id == "Sales Performance":

    total_revenue = df["Amount"].sum()
    total_orders = df["Order ID"].nunique()
    total_skus = df["SKU"].nunique()
    top_channel = df.groupby("Fulfilment")["Amount"].sum().idxmax()
    top_state = df.groupby("ship-state")["Amount"].sum().idxmax()
    average_order_value = total_revenue / total_orders

    cancelled_orders = df[df['Status'].isin(['Cancelled', 'Shipped - Lost in Transit'])]
    returned_orders = df[df['Status'].isin(
        ['Shipped - Returned to Seller', 'Shipped - Returning to Seller', 'Shipped - Rejected by Buyer',
         'Shipped - Damaged'])]
    total_cancelled = len(cancelled_orders)
    total_returned = len(returned_orders)


    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧾 Total Revenue", f"₹{total_revenue:,.2f}")
    col2.metric("📦 Total Orders", f"{total_orders}")
    col3.metric("🛍 Total SKUs Sold", f"{total_skus}")
    col4.metric("📈 Avg Order Value", f"{average_order_value:,.2f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("🏆 Top Sales Channel", f"{top_channel}")
    col6.metric("📍 Top State (India)", f"{top_state}")
    col7.metric("🚫 Cancelled Orders",f"{total_cancelled}")
    col8.metric("🔄 Returned Orders",f"{total_returned}")


    #MONTHLY REVENUE TRENDS

    time = st.selectbox("Select the Time for Revenue",options=["Daily","Monthly"])

    if time == "Daily":
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year-Month"] = df["Date"].dt.to_period("D").astype(str)
        monthly_revenue = df.groupby("Year-Month")["Amount"].sum()
    if time == "Monthly":
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year-Month"] = df["Date"].dt.to_period("M").astype(str)
        monthly_revenue = df.groupby("Year-Month")["Amount"].sum()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=monthly_revenue.index, y=monthly_revenue.values, mode="lines+markers", name="Close Price"))

    fig.update_layout(title=f"{time} Revenue Over Time", xaxis_title="Date", yaxis_title="Price",
                      template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)



    # TOP CATEGORY AND TOP SKU REVENUE

    col1, col2 = st.columns(2)

    with col1:
        category_rev = df.groupby("Category")["Amount"].sum().reset_index().sort_values(by="Amount", ascending=False)
        fig_cat = px.bar(category_rev, x="Category", y="Amount", title="Revenue by Category",
                         color="Category")
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        sku_rev = df.groupby("SKU")["Amount"].sum().reset_index().sort_values(by="Amount", ascending=False).head(10)
        fig_sku = px.bar(sku_rev, x="SKU", y="Amount", title="Top 10 SKUs by Revenue",
                         color="SKU")
        st.plotly_chart(fig_sku, use_container_width=True)




    #SALES CHANNEL ANALYSIS

    sales_channel_revenue = df.groupby("Fulfilment")["Amount"].sum().sort_values(ascending=False)

    filter_df={
        "sales_channel":sales_channel_revenue.index,
        "values":sales_channel_revenue.values
    }

    total_revenue = sales_channel_revenue.values.sum()
    sales_channel_revenue["Percentage"] = (sales_channel_revenue.values / total_revenue) * 100


    col1,col2 = st.columns(2)

    with col1:
        fig_exp = px.bar(filter_df, x="sales_channel", y="values", color="values",
                         title="Sales Channel Analysis",color_continuous_scale = "RdBu_r")
        st.plotly_chart(fig_exp, use_container_width=True)

    with col2:
        fig_sales_channel = px.pie(
            sales_channel_revenue,
            names=sales_channel_revenue.index,
            values=sales_channel_revenue.values,
            title="Percentage contribution of each sales channel",
            hole=0.2,

        )
        st.plotly_chart(fig_sales_channel, use_container_width=True)



    #REVENUE BY STATE BAR CHART

    df["ship-state"] = df["ship-state"].str.title()
    state_revenue = df.groupby("ship-state")["Amount"].sum().reset_index()
    fig_state = px.bar(
        state_revenue,
        x="ship-state",
        y="Amount",
        title="Revenue by State",
        labels={"Amount": "Revenue ($)"},
        color="ship-state",
        # color_continuous_scale = "RdBu_r" # Cividis, RdBu_r, Viridius,
    )
    st.plotly_chart(fig_state,use_container_width=True)



    #COURIER STATUS ANALYSIS

    col1, col2 = st.columns(2)

    with col1:
        courier_orders = df["Courier Status"].value_counts()
        fig1 = px.pie(courier_orders,names=courier_orders.index,
                values=courier_orders.values,title="Orders by Courier Status",color_discrete_sequence=px.colors.sequential.RdBu,hole=0.2)
        st.plotly_chart(fig1)

    with col2:    # Revenue by Courier Status
        courier_rev = df.groupby("Courier Status")["Amount"].sum().reset_index()
        fig2 = px.bar(courier_rev, x="Courier Status", y="Amount",
                      title="Revenue by Courier Status", color="Courier Status")
        st.plotly_chart(fig2)




    #SALES BY REGION MAP

    df["ship-state"] = df["ship-state"].str.title()
    state_revenue = df.groupby("ship-state")["Amount"].sum().reset_index()

    st.markdown("### 🌐 Sales by Region")
    st.write("")
    fig = px.choropleth(
        state_revenue,
        geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
        featureidkey='properties.ST_NM',
        locations='ship-state',
        color='Amount',
        color_continuous_scale="viridis_r"
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})  # scale map

    st.plotly_chart(fig,use_container_width=True)

    df["DayOfWeek"] = df["Date"].dt.day_name()

    df["Is_Weekend"] = df["DayOfWeek"].isin(["Saturday", "Sunday"])
    weekend_revenue = df[df["Is_Weekend"]]["Amount"].sum()
    weekday_revenue = df[~df["Is_Weekend"]]["Amount"].sum()

    weekend_weekday_data = pd.DataFrame({
        "Type": ["Weekend", "Weekday"],
        "Revenue": [weekend_revenue, weekday_revenue]
    })

    fig_pie = px.pie(
        weekend_weekday_data,
        names="Type",
        values="Revenue",
        title="Weekend vs. Weekday Revenue Share",
        hole=0.2,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    avg_price_by_category = df.groupby('Category')['Amount'].mean()
    avg_price_by_category = avg_price_by_category.sort_values(ascending=False)

    avg_price_by_category = pd.DataFrame({
        "Category":avg_price_by_category.index,
        "Amount": avg_price_by_category
    })

    #TOP 10 PRODUCTS
    top_products = df.groupby(["SKU", "Category"])["Amount"].sum().reset_index().sort_values(by="Amount",
                                                                                           ascending=False,ignore_index=True).head(10)
    st.write("")
    st.markdown("### 🏆 Top 10 Products by Revenue")
    st.dataframe(top_products, use_container_width=True,hide_index=1)


    #AVERAGE PRICE BY CATEGORY
    st.write("")
    st.markdown("### 🏷️ Average Price by Category")
    st.dataframe(avg_price_by_category, use_container_width=True,hide_index=1)

if menu_id == "Custom Filter":

    #SKU FILTER
    st.markdown("### 📦 Category-wise Performance")

    category = st.multiselect("Select Category", options=df["Category"].unique())

    qty_by_category = df.groupby("Category")["Qty"].sum()
    revenue_by_category = df.groupby("Category")["Amount"].sum()

    category_table = pd.DataFrame({
        "Qty_Sold": qty_by_category,
        "Revenue": revenue_by_category
    }).reset_index()
    st.dataframe(category_table[category_table["Category"].isin(category)], use_container_width=True)



    #STATE PERFORMANCE FILTER
    st.markdown("### 🌍 State-wise Performance")

    state = st.multiselect("Select State Name",options=df["ship-state"].unique())

    revenue_by_state = df.groupby("ship-state")["Amount"].sum()
    orders_by_state = df.groupby("ship-state")["Order ID"].nunique()

    state_data = pd.DataFrame({
        "Total Revenue": revenue_by_state,
        "Total Orders": orders_by_state
    }).reset_index()

    state_data["Avg Revenue per Order"] = (state_data["Total Revenue"] / state_data["Total Orders"]).round(2)
    st.dataframe(state_data[state_data["ship-state"].isin(state)], use_container_width=True)


    #CITY PERFORMANCE FILTER
    st.markdown("### 🌐 City-wise Performance")

    state = st.selectbox("Select State Name", options=df["ship-state"].unique(), key="1")

    city_df = df["ship-city"].where(df["ship-state"] == state)
    city = st.multiselect("Select City Name", options=city_df.unique(), key="2")

    revenue_by_state = df.groupby("ship-city")["Amount"].sum()
    orders_by_state = df.groupby("ship-city")["Order ID"].nunique()

    state_data = pd.DataFrame({
        "Total Revenue": revenue_by_state,
        "Total Orders": orders_by_state
    }).reset_index()

    state_data["Avg Revenue per Order"] = (state_data["Total Revenue"] / state_data["Total Orders"]).round(2)
    st.dataframe(state_data[state_data["ship-city"].isin(city)], use_container_width=True)



if menu_id == "Customer Insights":

    df["Date"] = pd.to_datetime(df["Date"])
    df["Hour"] = df["Date"].dt.hour
    df["Month"] = df["Date"].dt.month_name()
    df["DayOfWeek"] = df["Date"].dt.day_name()

    peak_day = df.groupby("DayOfWeek")["Amount"].sum().idxmax()

    peak_month = df.groupby("Month")["Amount"].sum().idxmax()

    df["Is_Weekend"] = df["DayOfWeek"].isin(["Saturday", "Sunday"])
    weekend_revenue = df[df["Is_Weekend"]]["Amount"].sum()
    weekday_revenue = df[~df["Is_Weekend"]]["Amount"].sum()

    fulfillment_counts = df["ship-service-level"].value_counts().reset_index()
    fastest_method = fulfillment_counts["ship-service-level"][0]

    average_item_per_order = round(df["Qty"].sum() / df["Order ID"].nunique(), 2)

    b2b_orders = len(df[df["B2B"] == 1].value_counts())

    b2c_orders = len(df[df["B2B"] == 0].value_counts())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Peak Sales Day", peak_day)
    col2.metric("📆 Peak Sales Month", peak_month)
    col3.metric("🏖️ Weekend Revenue", f"₹{weekend_revenue:,.0f}")
    col4.metric("📈 Weekday Revenue", f"₹{weekday_revenue:,.0f}")
    col1.metric("⚡ Fastest Ship-Service Method",fastest_method)
    col2.metric("📦 Avg Items per Order",average_item_per_order)
    col3.metric("💼 B2B Orders",b2b_orders)
    col4.metric("🧑‍🤝‍🧑 B2C Orders",b2c_orders)


    col1, col2 = st.columns(2)

    with col1:
        #REPEAT VS NEW CUSTOMERS
        b2b_summary = df["B2B"].value_counts().reset_index()
        b2b_summary.columns = ["Customer Type", "Orders"]
        b2b_summary["Customer Type"] = b2b_summary["Customer Type"].replace({True: "B2B (Repeat)", False: "B2C (New)"})
        fig = px.pie(b2b_summary, values="Orders", names="Customer Type", title="Repeat vs. New Customers",hole=0.2)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        #ORDER SIZE DISTRIBUTION
        order_size = df.groupby("Order ID")["Qty"].sum().reset_index()
        fig = px.histogram(order_size, x="Qty", nbins=20, title="Order Size Distribution (Qty per Order)")
        st.plotly_chart(fig, use_container_width=True)


    #HEAT MAP
    heatmap_data = df.groupby(["DayOfWeek", "Hour"])["Amount"].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index="DayOfWeek", columns="Hour", values="Amount")

    fig = px.imshow(heatmap_pivot,
                    labels=dict(x="Hour of Day", y="Day of Week", color="Revenue"),
                    title="🕒 Revenue Heatmap by Hour & Day")
    st.plotly_chart(fig, use_container_width=True)



    #DATE TIME ANALYSIS
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month_name()
    df["Week"] = df["Date"].dt.isocalendar().week
    df["DayOfWeek"] = df["Date"].dt.day_name()

    option = st.selectbox("🔎 View Sales By:", ["Day of Week", "Month", "Week Number"])

    if option == "Day of Week":
        revenue_by_day = df.groupby("DayOfWeek")["Amount"].sum().reset_index().sort_values(by="Amount", ascending=False)
        fig = px.bar(revenue_by_day, x="DayOfWeek", y="Amount", title="📅 Revenue by Day of Week",
                     color="DayOfWeek", labels={"Amount": "Revenue"})
        st.plotly_chart(fig, use_container_width=True)

    elif option == "Month":
        revenue_by_month = df.groupby("Month")["Amount"].sum().reset_index()
        fig = px.bar(revenue_by_month, x="Month", y="Amount", title="📅 Revenue by Month",
                     color="Month", labels={"Amount": "Revenue"})
        st.plotly_chart(fig, use_container_width=True)

    elif option == "Week Number":
        revenue_by_week = df.groupby("Week")["Amount"].sum().reset_index()
        fig = px.line(revenue_by_week, x="Week", y="Amount", markers=True, title="📈 Weekly Revenue Trend",
                      labels={"Amount": "Revenue"})
        st.plotly_chart(fig, use_container_width=True)


    # QUANTITY SOLD BY DAY OF WEEK
    quantity_by_day = df.groupby("DayOfWeek")["Qty"].sum().reset_index().sort_values(by="Qty", ascending=False)
    fig_qty = px.bar(quantity_by_day, x="DayOfWeek", y="Qty", title="📦 Quantity Sold by Day of Week",
                     color="Qty", labels={"Qty": "Quantity"})
    st.plotly_chart(fig_qty, use_container_width=True)


    #CUSTOMER LOCATION
    state_revenue = df.groupby("ship-state")["Amount"].sum().reset_index()
    state_revenue.columns = ["State", "Revenue"]
    st.markdown("### 📍 Customer Locations")
    location_table = df["ship-state"].value_counts().reset_index()
    location_table.columns = ["State", "Orders"]
    location_table = location_table.merge(state_revenue, on="State")
    location_table = location_table.sort_values(by="Revenue", ascending=False)
    st.dataframe(location_table, use_container_width=True,hide_index=1)


    #POPULAR CATEGORY BY STATE
    popular_category_by_state = df.groupby(['ship-state', 'Category'])['Qty'].sum().reset_index()
    popular_category_by_state = popular_category_by_state.sort_values(['ship-state', 'Qty'],
                                                                      ascending=[True,False])
    popular_category_by_state = popular_category_by_state.drop_duplicates(subset=['ship-state'])
    st.markdown("### 🗺️ Popular Category by State")
    st.dataframe(popular_category_by_state,use_container_width=True,hide_index=1)


if menu_id == "Revenue Forecasting":

    st.markdown("### 🔮 Revenue Forecasting")

    # REVENUE PREDICTION LSTM

    df["Year-Month"] = pd.to_datetime(df["Date"]).dt.to_period("D").astype(str)
    daily_revenue = df.groupby("Year-Month")["Amount"].sum().reset_index()
    daily_revenue["Year-Month"] = pd.to_datetime(daily_revenue["Year-Month"])

    daily_revenue = daily_revenue.sort_values("Year-Month")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(daily_revenue["Amount"].values.reshape(-1, 1))

    def create_sequences(data, seq_len=30):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i])
            y.append(data[i])
        return np.array(X), np.array(y)


    sequence_length = 30
    X, y = create_sequences(scaled_data, sequence_length)

    # Build LSTM model
    model = models.Sequential([
        Input(shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)

    n_days = st.slider("📅 Forecast Revenue For Next N Days", min_value=7, max_value=90, value=30)

    # Predict future revenue
    last_sequence = scaled_data[-sequence_length:]
    future_preds = []
    current_seq = last_sequence.reshape(1, sequence_length, 1)

    for _ in range(n_days):
        pred = model.predict(current_seq, verbose=0)[0][0]
        future_preds.append(pred)
        current_seq = np.append(current_seq[:, 1:, :], [[[pred]]], axis=1)

    # Inverse scale predictions
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()


    last_date = daily_revenue["Year-Month"].max()
    future_dates = [last_date + timedelta(days=i + 1) for i in range(n_days)]

    forecast_df = pd.DataFrame({
        "Year-Month": future_dates,
        "Predicted_Revenue": future_preds
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast_df["Year-Month"],
        y=forecast_df["Predicted_Revenue"],
        mode='lines',
        name='Predicted Revenue',
        line=dict(dash='dot'),
    ))

    fig.add_trace(go.Scatter(
        x=daily_revenue["Year-Month"],
        y=daily_revenue["Amount"],
        mode='lines',
        name='Actual Revenue',
        # line=dict(color='lightgray', width=2)
    ))

    fig.update_layout(
        title=f"💹 Revenue Forecast For Next {n_days} Days",
        xaxis_title="Date",
        yaxis_title="Revenue",
        legend_title="Legend",
        hovermode="x unified",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)



    # ORDER VOLUME PREDICTION

    df["Year-Month"] = pd.to_datetime(df["Year-Month"])
    df_daily = df.groupby("Year-Month")["Qty"].sum().reset_index()

    df_daily["Day"] = df_daily["Year-Month"].dt.day
    df_daily["Month"] = df_daily["Year-Month"].dt.month
    df_daily["Weekday"] = df_daily["Year-Month"].dt.weekday

    X = df_daily[["Day", "Month", "Weekday"]]
    y = df_daily["Qty"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    last_date = df_daily["Year-Month"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, n_days)]
    future_df = pd.DataFrame({"Date": future_dates})
    future_df["Day"] = future_df["Date"].dt.day
    future_df["Month"] = future_df["Date"].dt.month
    future_df["Weekday"] = future_df["Date"].dt.weekday

    future_df["Forecast_Qty"] = model.predict(future_df[["Day", "Month", "Weekday"]])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_daily["Year-Month"], y=df_daily["Qty"],
        mode="lines", name="Historical Orders"
    ))

    fig.add_trace(go.Scatter(
        x=future_df["Date"], y=future_df["Forecast_Qty"],
        mode="lines+markers", name="Forecasted Orders"
    ))

    fig.update_layout(
        title=f"📦 Order Volume Forecast For Next {n_days} Days",
        xaxis_title="Date",
        yaxis_title="Order Volume",
        template="plotly_dark"
    )

    st.plotly_chart(fig,use_container_width=True)



    #CATEGORY LEVEL REVENUE PREDICTION

    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Category", "Amount"]]

    daily_cat_rev = df.groupby(["Date", "Category"])["Amount"].sum().reset_index()

    daily_cat_rev["Day"] = daily_cat_rev["Date"].dt.day
    daily_cat_rev["Month"] = daily_cat_rev["Date"].dt.month
    daily_cat_rev["Weekday"] = daily_cat_rev["Date"].dt.weekday

    categories = daily_cat_rev["Category"].unique()
    forecast_data = []


    cat = st.selectbox("Select the Category",options=categories)

    cat_df = daily_cat_rev[daily_cat_rev["Category"] == cat].copy()

    X = cat_df[["Day", "Month", "Weekday"]]
    y = cat_df["Amount"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    last_date = cat_df["Date"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, n_days + 1)]

    future_df = pd.DataFrame({"Date": future_dates})
    future_df["Day"] = future_df["Date"].dt.day
    future_df["Month"] = future_df["Date"].dt.month
    future_df["Weekday"] = future_df["Date"].dt.weekday
    future_df["Forecast_Revenue"] = model.predict(future_df[["Day", "Month", "Weekday"]])
    future_df["Category"] = cat


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=future_df["Date"],
        y=future_df["Forecast_Revenue"],
        mode="lines",
        stackgroup="one",
        name=cat
    ))

    fig.update_layout(
        title=f"📈 Category-Level Revenue Forecast For Next {n_days} Days",
        xaxis_title="Date",
        yaxis_title="Forecasted Revenue",
        template="plotly_dark"
    )

    st.plotly_chart(fig,use_container_width=True)



    # SKU LEVEL DEMAND PREDICTION

    df = pd.read_csv("sample_sales_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "SKU", "Qty"]]

    sku_daily = df.groupby(["Date", "SKU"])["Qty"].sum().reset_index()

    top_skus = sku_daily.groupby("SKU")["Qty"].sum().nlargest(5).index.tolist()
    sku_daily = sku_daily[sku_daily["SKU"].isin(top_skus)]

    sku_daily["Day"] = sku_daily["Date"].dt.day
    sku_daily["Month"] = sku_daily["Date"].dt.month
    sku_daily["Weekday"] = sku_daily["Date"].dt.weekday

    future_days = n_days
    forecast_list = []

    for sku in top_skus:
        sku_df = sku_daily[sku_daily["SKU"] == sku].copy()

        X = sku_df[["Day", "Month", "Weekday"]]
        y = sku_df["Qty"]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        last_date = sku_df["Date"].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]

        future_df = pd.DataFrame({"Date": future_dates})
        future_df["Day"] = future_df["Date"].dt.day
        future_df["Month"] = future_df["Date"].dt.month
        future_df["Weekday"] = future_df["Date"].dt.weekday
        future_df["Forecast_Qty"] = model.predict(future_df[["Day", "Month", "Weekday"]])
        future_df["SKU"] = sku

        forecast_list.append(future_df)

    forecast_df = pd.concat(forecast_list)

    fig = go.Figure()

    for sku in top_skus:
        data = forecast_df[forecast_df["SKU"] == sku]
        fig.add_trace(go.Scatter(
            x=data["Date"],
            y=data["Forecast_Qty"],
            mode="lines",
            name=sku
        ))

    fig.update_layout(
        title=f"🏷️ SKU-Level Demand Forecast For Next {n_days} Days",
        xaxis_title="Date",
        yaxis_title="Forecasted Quantity Sold",
        template="plotly_white"
    )

    st.plotly_chart(fig,use_container_width=True)


if menu_id == "Logout":
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.query_params.clear()
    st.success("Logout Successfully")
    st.rerun()
