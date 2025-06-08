import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Ahva Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📦"
)

# ========== CSS Styling ==========
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .kpi-container {
        display: flex;
        gap: 15px;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        flex: 1;
        min-width: 200px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    .kpi-blue { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .kpi-green { background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); }
    .kpi-red { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); }
    .kpi-purple { background: linear-gradient(135deg, #8360c3 0%, #2ebf91 100%); }
    .kpi-orange { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); }
    .kpi-title { 
        font-size: 14px; 
        margin-bottom: 10px; 
        opacity: 0.9; 
        font-weight: 500;
    }
    .kpi-value { 
        font-size: 28px; 
        font-weight: bold; 
        margin: 10px 0; 
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .kpi-subtext { 
        font-size: 12px; 
        opacity: 0.8; 
    }
    .sidebar-title { 
        color: #2e4057; 
        margin-bottom: 20px; 
        font-weight: bold;
        text-align: center;
    }
    h1, h2, h3 {
        color: #2e4057;
        font-weight: 600;
    }
    hr {
        margin: 1rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    .forecast-highlight {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #2d3436;
        font-weight: 500;
    }
    @media (max-width: 768px) {
        .kpi-container { flex-direction: column; }
        .kpi-card { min-width: 100%; }
        .kpi-value { font-size: 24px; }
    }
</style>
""", unsafe_allow_html=True)

# ========== Data Cleaning Functions ==========
@st.cache_data
def clean_data(df):
    """ניקוי וטיפול בנתונים פגומים"""
    df_clean = df.copy()
    
    # טיפול בתאריכים
    if 'Date' in df_clean.columns:
        def convert_date(date_val):
            if pd.isna(date_val):
                return pd.NaT
            if isinstance(date_val, (int, float)):
                try:
                    return pd.to_datetime('1899-12-30') + pd.Timedelta(days=date_val)
                except:
                    return pd.NaT
            try:
                return pd.to_datetime(date_val)
            except:
                return pd.NaT
        
        df_clean['Date'] = df_clean['Date'].apply(convert_date)
        invalid_dates = df_clean['Date'].isna().sum()
        if invalid_dates > 0:
            df_clean = df_clean.dropna(subset=['Date'])
    
    # תיקון שמות קטגוריות לעקביות
    if 'Category' in df_clean.columns:
        category_mapping = {
            'חלוה': 'Halva', 'חלווה': 'Halva', 'halva': 'Halva', 'HALVA': 'Halva',
            'טחינה': 'Tahini', 'TAHINI': 'Tahini', 'tahini': 'Tahini',
            'חטיפים': 'Snacks', 'SNACKS': 'Snacks', 'snacks': 'Snacks',
            'עוגות': 'Cakes', 'CAKES': 'Cakes', 'cakes': 'Cakes',
            'עוגיות': 'Cookies', 'COOKIES': 'Cookies', 'cookies': 'Cookies',
            'מאפים': 'Pastries', 'PASTRIES': 'Pastries', 'pastries': 'Pastries',
            'סירופ': 'Syrup', 'SYRUP': 'Syrup', 'syrup': 'Syrup'
        }
        df_clean['Category'] = df_clean['Category'].replace(category_mapping)
        df_clean['Category'] = df_clean['Category'].str.title()
    
    # הסרת שורות חסרות מידע חיוני
    critical_columns = [col for col in ['Product', 'Category', 'UnitsSold', 'Stock'] if col in df_clean.columns]
    before_cleaning = len(df_clean)
    df_clean = df_clean.dropna(subset=critical_columns)
    after_cleaning = len(df_clean)
    
    if before_cleaning != after_cleaning:
        st.info(f"🧹 Data Cleaning: Removed {before_cleaning - after_cleaning} rows with missing critical data")
    
    # טיפול בערכים שליליים
    numeric_columns = ['UnitsSold', 'Stock', 'עלות ליחידה (₪)', 'מחיר ליחידה (₪)']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            negative_count = (df_clean[col] < 0).sum()
            if negative_count > 0:
                df_clean[col] = df_clean[col].abs()
    
    if 'Category' in df_clean.columns:
        unique_categories = sorted(df_clean['Category'].unique())
        st.success(f"✅ Categories standardized: {', '.join(unique_categories)}")
    
    return df_clean

@st.cache_data
def prepare_forecast_data(df):
    """הכנת נתונים לחיזוי"""
    if len(df) == 0:
        return df
    
    df_forecast = df.copy()
    
    # פיצ'רי זמן
    df_forecast['Year'] = df_forecast['Date'].dt.year
    df_forecast['Month'] = df_forecast['Date'].dt.month
    df_forecast['DayOfWeek'] = df_forecast['Date'].dt.dayofweek
    df_forecast['WeekOfYear'] = df_forecast['Date'].dt.isocalendar().week
    df_forecast['Quarter'] = df_forecast['Date'].dt.quarter
    df_forecast['IsWeekend'] = df_forecast['DayOfWeek'].isin([5, 6]).astype(int)
    
    # פיצ'רי מוצר
    df_forecast['Product_encoded'] = pd.Categorical(df_forecast['Product']).codes
    df_forecast['Category_encoded'] = pd.Categorical(df_forecast['Category']).codes
    
    # פיצ'רי מלאי ומחיר
    df_forecast['PricePerUnit'] = pd.to_numeric(df_forecast.get('מחיר ליחידה (₪)', 0), errors='coerce').fillna(0)
    df_forecast['WeightPerUnit'] = pd.to_numeric(df_forecast.get('משקל יחידה (גרם)', 0), errors='coerce').fillna(0)
    
    # פיצ'רי מכירות היסטוריות
    df_forecast = df_forecast.sort_values(['Product', 'Date'])
    df_forecast['Sales_MA_7'] = df_forecast.groupby('Product')['UnitsSold'].transform(
        lambda x: x.rolling(window=min(7, len(x)), min_periods=1).mean()
    )
    df_forecast['Sales_MA_30'] = df_forecast.groupby('Product')['UnitsSold'].transform(
        lambda x: x.rolling(window=min(30, len(x)), min_periods=1).mean()
    )
    
    return df_forecast

def build_forecast_model(df_forecast):
    """בניית מודל חיזוי"""
    if len(df_forecast) < 10:
        raise ValueError("Not enough data for forecasting")
    
    features = [
        'Month', 'DayOfWeek', 'WeekOfYear', 'Quarter', 'IsWeekend',
        'Product_encoded', 'Category_encoded', 'Stock', 'PricePerUnit',
        'WeightPerUnit', 'Sales_MA_7', 'Sales_MA_30'
    ]
    
    X = df_forecast[features].fillna(0)
    y = df_forecast['UnitsSold']
    
    test_size = min(0.2, max(0.1, len(df_forecast) // 10))
    
    if len(df_forecast) > 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
    
    n_estimators = min(100, max(10, len(X_train) // 2))
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, features, mae, rmse

# ========== Navigation ==========
st.sidebar.markdown("<h2 class='sidebar-title'>🧭 Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to:", ["🏠 HOME", "📊 Analysis", "📈 Seasonality", "🔮 Forecasting"])

# ========== Session State ==========
if "df" not in st.session_state:
    st.session_state.df = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

# ========== HOME PAGE ==========
if page == "🏠 HOME":
    st.markdown("""
    <h1 style='margin-bottom: 10px; text-align: center;'>📦 Ahva Inventory Dashboard</h1>
    <p style='text-align: center; font-size: 18px; color: #666;'>Advanced Analytics & Sales Forecasting Platform</p>
    <hr>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📁 Upload Excel File", 
        type=["xlsx", "xls"],
        help="Upload your Ahva sales data file (Excel format)"
    )

    if uploaded_file is not None:
        try:
            with st.spinner("📊 Loading and analyzing your data..."):
                df = pd.read_excel(uploaded_file)
                st.session_state.df = df
                df_clean = clean_data(df)
                st.session_state.df_clean = df_clean
            
            st.success("✅ File uploaded and processed successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**📋 Raw Data Overview:**")
                st.write(f"- Original rows: {len(df):,}")
                st.write(f"- Columns: {len(df.columns)}")
                st.write(f"- File size: {uploaded_file.size / 1024:.1f} KB")
                
            with col2:
                st.write("**✨ Cleaned Data Overview:**")
                st.write(f"- Processed rows: {len(df_clean):,}")
                st.write(f"- Data quality: {(len(df_clean)/len(df)*100):.1f}%")
                st.write(f"- Ready for analysis: ✅")
            
            with st.expander("👀 Preview Your Data", expanded=False):
                st.write("**Sample of your cleaned data:**")
                st.dataframe(df_clean.head(10))
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean

        st.markdown("---")
        st.subheader("📅 Date Range Filter")
        
        if 'Date' in df.columns and not df['Date'].isna().all():
            min_date = df['Date'].min().date()
            max_date = df['Date'].max().date()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
            
            filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
            
            if len(filtered_df) == 0:
                st.warning("⚠️ No data found in the selected date range.")
                filtered_df = df
        else:
            filtered_df = df

        # KPI CALCULATIONS
        st.markdown("---")
        st.subheader("📊 Key Performance Indicators")
        
        total_products = filtered_df['Product'].nunique() if 'Product' in filtered_df.columns else 0
        total_stock = int(filtered_df['Stock'].sum()) if 'Stock' in filtered_df.columns else 0
        total_demand = int(filtered_df['UnitsSold'].sum()) if 'UnitsSold' in filtered_df.columns else 0
        
        if 'UnitsSold' in filtered_df.columns and 'Stock' in filtered_df.columns:
            shortages = (filtered_df['UnitsSold'] > filtered_df['Stock']).sum()
            filtered_df["ShortageQty"] = (filtered_df["UnitsSold"] - filtered_df["Stock"]).clip(lower=0)
            missing_units = int(filtered_df["ShortageQty"].sum())
        else:
            shortages = 0
            missing_units = 0

        efficiency = (total_demand / total_stock) * 100 if total_stock > 0 else 0

        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card kpi-blue">
                <div class="kpi-title">❌ Missing Units</div>
                <div class="kpi-value">{missing_units:,}</div>
                <div class="kpi-subtext">Total Shortage Quantity</div>
            </div>
            <div class="kpi-card kpi-green">
                <div class="kpi-title">📦 Total Stock</div>
                <div class="kpi-value">{total_stock:,}</div>
                <div class="kpi-subtext">Available Inventory</div>
            </div>
            <div class="kpi-card kpi-red">
                <div class="kpi-title">⚠️ Shortage Events</div>
                <div class="kpi-value">{shortages:,}</div>
                <div class="kpi-subtext">Times Demand > Stock</div>
            </div>
            <div class="kpi-card kpi-purple">
                <div class="kpi-title">📊 Total Demand</div>
                <div class="kpi-value">{total_demand:,}</div>
                <div class="kpi-subtext">Units Sold</div>
            </div>
            <div class="kpi-card kpi-orange">
                <div class="kpi-title">📈 Efficiency</div>
                <div class="kpi-value">{efficiency:.1f}%</div>
                <div class="kpi-subtext">Demand/Stock Ratio</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ========== ANALYSIS PAGE ==========
elif page == "📊 Analysis":
    st.markdown("<h1>📊 Sales & Demand Analysis</h1><hr>", unsafe_allow_html=True)

    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean.copy()

        required_cols = ['Category', 'UnitsSold']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        else:
            # Sales by Category with Plotly Charts
            st.subheader("🏷️ Sales Distribution by Category")
            category_sales = df.groupby("Category")["UnitsSold"].agg(['sum', 'mean', 'count']).reset_index()
            category_sales.columns = ['Category', 'Total_Sales', 'Avg_Sales', 'Records']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Interactive Bar Chart
                fig_bar = px.bar(
                    category_sales, 
                    x="Category", 
                    y="Total_Sales",
                    color="Total_Sales",
                    title="Total Units Sold per Category",
                    labels={"Total_Sales": "Total Units Sold"},
                    color_continuous_scale="Blues",
                    text="Total_Sales"
                )
                fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                fig_bar.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Interactive Pie Chart
                fig_pie = px.pie(
                    category_sales, 
                    values="Total_Sales", 
                    names="Category",
                    title="Sales Distribution (%)",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Category performance table
            st.write("**📋 Category Performance Summary:**")
            category_sales['Avg_Sales'] = category_sales['Avg_Sales'].round(1)
            st.dataframe(category_sales, use_container_width=True)

            # Time-based analysis
            if 'Date' in df.columns and not df['Date'].isna().all():
                st.markdown("---")
                st.subheader("📈 Sales Trends Over Time")
                
                # Daily sales trend
                daily_sales = df.groupby('Date')['UnitsSold'].sum().reset_index()
                fig_trend = px.line(
                    daily_sales, 
                    x='Date', 
                    y='UnitsSold',
                    title='Daily Sales Trend',
                    labels={'UnitsSold': 'Units Sold'}
                )
                fig_trend.update_traces(line_color='#1f77b4', line_width=2)
                fig_trend.update_layout(height=400)
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Sales Pattern Analysis
                st.markdown("---")
                st.subheader("📊 Sales Pattern Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Day of week analysis
                    df['DayName'] = df['Date'].dt.day_name()
                    daily_pattern = df.groupby('DayName')['UnitsSold'].sum().reset_index()
                    
                    # Order days
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    daily_pattern['DayName'] = pd.Categorical(daily_pattern['DayName'], categories=day_order, ordered=True)
                    daily_pattern = daily_pattern.sort_values('DayName')
                    
                    fig_daily = px.bar(
                        daily_pattern,
                        x='DayName',
                        y='UnitsSold',
                        title="Sales by Day of Week",
                        color='UnitsSold',
                        color_continuous_scale='Blues'
                    )
                    fig_daily.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_daily, use_container_width=True)
                
                with col2:
                    # Product velocity
                    product_velocity = df.groupby('Product')['UnitsSold'].agg(['sum', 'mean']).reset_index()
                    product_velocity.columns = ['Product', 'Total_Sales', 'Avg_Daily_Sales']
                    top_products = product_velocity.nlargest(10, 'Total_Sales')
                    
                    fig_products = px.bar(
                        top_products,
                        x='Total_Sales',
                        y='Product',
                        orientation='h',
                        title='Top 10 Products by Sales',
                        color='Total_Sales',
                        color_continuous_scale='Viridis'
                    )
                    fig_products.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                    st.plotly_chart(fig_products, use_container_width=True)

    else:
        st.warning("📁 Please upload a file in the HOME page first.")

# ========== SEASONALITY PAGE ==========
elif page == "📈 Seasonality":
    st.markdown("<h1>📈 Seasonality Analysis</h1><hr>", unsafe_allow_html=True)

    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean.copy()
        
        required_cols = ['Product', 'UnitsSold', 'Date']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        elif df['Date'].isna().all():
            st.error("❌ Date column contains no valid dates")
        else:
            # Product selector
            products = df['Product'].unique()
            selected_product = st.selectbox("🏷️ Select Product for Analysis:", products)
            
            product_data = df[df['Product'] == selected_product].copy()
            
            if len(product_data) == 0:
                st.warning("⚠️ No data found for selected product.")
            else:
                st.subheader(f"📅 Seasonality Analysis for {selected_product}")
                
                # Monthly analysis with plotly
                product_data['Month'] = product_data['Date'].dt.month
                product_data['MonthName'] = product_data['Date'].dt.month_name()
                monthly_sales = product_data.groupby(['Month', 'MonthName'])['UnitsSold'].sum().reset_index()
                monthly_sales.columns = ['Month', 'MonthName', 'Total_Sales']
                
                # Interactive line chart
                fig_monthly = px.line(
                    monthly_sales, 
                    x='MonthName', 
                    y='Total_Sales',
                    markers=True,
                    title=f"Monthly Sales Pattern for {selected_product}",
                    labels={'Total_Sales': 'Total Units Sold', 'MonthName': 'Month'}
                )
                fig_monthly.update_traces(line_color='#1e88e5', marker_size=8, line_width=3)
                fig_monthly.update_layout(height=400)
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Sales", f"{product_data['UnitsSold'].sum():,.0f}")
                with col2:
                    peak_month = monthly_sales.loc[monthly_sales['Total_Sales'].idxmax(), 'MonthName']
                    st.metric("Peak Month", peak_month)
                with col3:
                    avg_monthly = monthly_sales['Total_Sales'].mean()
                    st.metric("Avg Monthly", f"{avg_monthly:.1f}")
                with col4:
                    peak_ratio = monthly_sales['Total_Sales'].max() / monthly_sales['Total_Sales'].mean()
                    st.metric("Seasonality Index", f"{peak_ratio:.1f}x")
                
                # Weekly pattern
                st.markdown("---")
                st.subheader("📊 Weekly Sales Pattern")
                
                product_data['DayOfWeek'] = product_data['Date'].dt.day_name()
                weekly_sales = product_data.groupby('DayOfWeek')['UnitsSold'].sum().reset_index()
                
                # Order days
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekly_sales['DayOfWeek'] = pd.Categorical(weekly_sales['DayOfWeek'], categories=day_order, ordered=True)
                weekly_sales = weekly_sales.sort_values('DayOfWeek')
                
                fig_weekly = px.bar(
                    weekly_sales,
                    x='DayOfWeek',
                    y='UnitsSold',
                    title=f"Weekly Sales Pattern for {selected_product}",
                    color='UnitsSold',
                    color_continuous_scale='Blues'
                )
                fig_weekly.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_weekly, use_container_width=True)

    else:
        st.warning("📁 Please upload a file in the HOME page first.")

# ========== FORECASTING PAGE ==========
elif page == "🔮 Forecasting":
    st.markdown("<h1>🔮 Advanced Sales Forecasting</h1><hr>", unsafe_allow_html=True)
    
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean.copy()
        
        st.subheader("🤖 Machine Learning Sales Prediction")
        
        if len(df) < 10:
            st.error("❌ Insufficient data for forecasting. Need at least 10 records.")
        else:
            with st.spinner("🧠 Building advanced forecast model..."):
                try:
                    # Prepare data for forecasting
                    df_forecast = prepare_forecast_data(df)
                    
                    # Build model
                    model, features, mae, rmse = build_forecast_model(df_forecast)
                    
                    # Display model performance
                    st.success("✅ Advanced ML model built successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model MAE", f"{mae:.2f}", help="Mean Absolute Error")
                    with col2:
                        st.metric("Model RMSE", f"{rmse:.2f}", help="Root Mean Square Error")
                    with col3:
                        avg_sales = df['UnitsSold'].mean()
                        accuracy_pct = max(0, (1 - mae/avg_sales) * 100)
                        st.metric("Accuracy", f"{accuracy_pct:.1f}%", help="Model accuracy percentage")
                    
                    # Product selection for forecasting
                    st.markdown("---")
                    st.subheader("📊 Generate Product Forecast")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_product = st.selectbox("🏷️ Select Product:", df['Product'].unique())
                    with col2:
                        forecast_days = st.slider("📅 Forecast Period (days):", 7, 90, 30)
                    
                    if st.button("🔮 Generate Advanced Forecast", type="primary"):
                        try:
                            # Get product data
                            product_data = df[df['Product'] == selected_product].iloc[-1]
                            
                            # Create future dates
                            last_date = df['Date'].max()
                            future_dates = pd.date_range(
                                start=last_date + pd.Timedelta(days=1), 
                                periods=forecast_days, 
                                freq='D'
                            )
                            
                            # Prepare future data
                            future_data = []
                            for date in future_dates:
                                row = {
                                    'Date': date,
                                    'Product': selected_product,
                                    'Month': date.month,
                                    'DayOfWeek': date.dayofweek,
                                    'WeekOfYear': date.isocalendar().week,
                                    'Quarter': date.quarter,
                                    'IsWeekend': 1 if date.dayofweek >= 5 else 0,
                                    'Product_encoded': pd.Categorical([selected_product], 
                                                                    categories=df['Product'].unique()).codes[0],
                                    'Category_encoded': pd.Categorical([product_data['Category']], 
                                                                     categories=df['Category'].unique()).codes[0],
                                    'Stock': product_data['Stock'],
                                    'PricePerUnit': product_data.get('מחיר ליחידה (₪)', 0),
                                    'WeightPerUnit': product_data.get('משקל יחידה (גרם)', 0),
                                    'Sales_MA_7': df[df['Product'] == selected_product]['UnitsSold'].tail(7).mean(),
                                    'Sales_MA_30': df[df['Product'] == selected_product]['UnitsSold'].tail(30).mean()
                                }
                                future_data.append(row)
                            
                            future_df = pd.DataFrame(future_data)
                            
                            # Generate predictions
                            X_future = future_df[features].fillna(0)
                            predictions = model.predict(X_future)
                            predictions = np.maximum(predictions, 0)  # No negative predictions
                            
                            future_df['Predicted_Sales'] = predictions
                            
                            # Create advanced forecast visualization with plotly
                            st.markdown("---")
                            st.subheader(f"📈 Advanced Forecast Results for {selected_product}")
                            
                            # Historical vs Forecast chart
                            historical_data = df[df['Product'] == selected_product].tail(60)  # Last 60 days
                            
                            fig = go.Figure()
                            
                            # Historical data
                            fig.add_trace(go.Scatter(
                                x=historical_data['Date'],
                                y=historical_data['UnitsSold'],
                                mode='lines+markers',
                                name='Historical Sales',
                                line=dict(color='#2E86AB', width=3),
                                marker=dict(size=6),
                                hovertemplate='<b>Historical</b><br>Date: %{x}<br>Sales: %{y}<extra></extra>'
                            ))
                            
                            # Forecast data
                            fig.add_trace(go.Scatter(
                                x=future_df['Date'],
                                y=future_df['Predicted_Sales'],
                                mode='lines+markers',
                                name='ML Forecast',
                                line=dict(color='#F24236', width=3, dash='dash'),
                                marker=dict(size=6),
                                hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Predicted: %{y:.1f}<extra></extra>'
                            ))
                            
                            # Add confidence interval (simple estimation)
                            uncertainty = predictions * 0.15  # 15% uncertainty
                            
                            # Upper bound
                            fig.add_trace(go.Scatter(
                                x=future_df['Date'],
                                y=predictions + uncertainty,
                                fill=None,
                                mode='lines',
                                line_color='rgba(242, 66, 54, 0)',
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            
                            # Lower bound
                            fig.add_trace(go.Scatter(
                                x=future_df['Date'],
                                y=predictions - uncertainty,
                                fill='tonexty',
                                mode='lines',
                                line_color='rgba(242, 66, 54, 0)',
                                name='Confidence Interval',
                                fillcolor='rgba(242, 66, 54, 0.2)',
                                hoverinfo='skip'
                            ))
                            
                            # Add vertical line to separate historical from forecast
                            fig.add_vline(
                                x=last_date,
                                line_dash="dot",
                                line_color="gray",
                                annotation_text="Forecast Start",
                                annotation_position="top"
                            )
                            
                            fig.update_layout(
                                title=f'Advanced ML Sales Forecast for {selected_product}',
                                xaxis_title='Date',
                                yaxis_title='Units Sold',
                                height=500,
                                showlegend=True,
                                hovermode='x unified',
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Forecast summary metrics
                            st.subheader("📊 Forecast Summary")
                            
                            total_forecast = future_df['Predicted_Sales'].sum()
                            avg_daily_forecast = future_df['Predicted_Sales'].mean()
                            max_daily_forecast = future_df['Predicted_Sales'].max()
                            min_daily_forecast = future_df['Predicted_Sales'].min()
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Forecast", f"{total_forecast:.0f}")
                            with col2:
                                st.metric("Daily Average", f"{avg_daily_forecast:.1f}")
                            with col3:
                                st.metric("Peak Day", f"{max_daily_forecast:.0f}")
                            with col4:
                                st.metric("Low Day", f"{min_daily_forecast:.0f}")
                            
                            # Business recommendations
                            st.markdown("---")
                            st.subheader("💡 AI-Powered Business Recommendations")
                            
                            current_stock = product_data['Stock']
                            recommended_stock = total_forecast * 1.2  # 20% safety buffer
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if current_stock < recommended_stock:
                                    st.markdown("""
                                    <div class="recommendation-box">
                                        <h4>⚠️ Stock Alert</h4>
                                        <p><strong>Current Stock:</strong> {:.0f} units</p>
                                        <p><strong>Forecasted Demand:</strong> {:.0f} units</p>
                                        <p><strong>Recommended Stock:</strong> {:.0f} units</p>
                                        <p><strong>Additional Stock Needed:</strong> {:.0f} units</p>
                                        <p><em>🎯 Action: Order additional inventory</em></p>
                                    </div>
                                    """.format(current_stock, total_forecast, recommended_stock, recommended_stock - current_stock), 
                                    unsafe_allow_html=True)
                                else:
                                    st.markdown("""
                                    <div class="forecast-highlight">
                                        <h4>✅ Stock Status: Optimal</h4>
                                        <p><strong>Current Stock:</strong> {:.0f} units</p>
                                        <p><strong>Forecasted Demand:</strong> {:.0f} units</p>
                                        <p><strong>Buffer Available:</strong> {:.0f} units</p>
                                        <p><em>🎯 Status: No action needed</em></p>
                                    </div>
                                    """.format(current_stock, total_forecast, current_stock - total_forecast), 
                                    unsafe_allow_html=True)
                            
                            with col2:
                                # Peak days identification
                                peak_days = future_df.nlargest(3, 'Predicted_Sales')[['Date', 'Predicted_Sales']]
                                peak_dates = [d.strftime('%Y-%m-%d') for d in peak_days['Date']]
                                
                                st.markdown(f"""
                                <div class="forecast-highlight">
                                    <h4>📈 Peak Sales Days</h4>
                                    <p><strong>Expected peak sales on:</strong></p>
                                    <ul>
                                        {"".join([f"<li>{date} - {sales:.0f} units</li>" for date, sales in zip(peak_dates, peak_days['Predicted_Sales'])])}
                                    </ul>
                                    <p><em>🎯 Action: Ensure adequate staffing and inventory</em></p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Detailed forecast table
                            with st.expander("📋 Detailed Forecast Table", expanded=False):
                                display_df = future_df[['Date', 'Predicted_Sales']].copy()
                                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                                display_df['Predicted_Sales'] = display_df['Predicted_Sales'].round(1)
                                display_df.columns = ['Date', 'Predicted Sales']
                                
                                st.dataframe(display_df, use_container_width=True)
                                
                                # Download option
                                csv = display_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Forecast CSV",
                                    data=csv,
                                    file_name=f'ml_forecast_{selected_product}_{datetime.now().strftime("%Y%m%d")}.csv',
                                    mime='text/csv'
                                )
                            
                        except Exception as e:
                            st.error(f"❌ Error generating forecast: {str(e)}")
                    
                    # Feature importance analysis
                    st.markdown("---")
                    st.subheader("🎯 ML Model Feature Importance Analysis")
                    
                    feature_importance = pd.DataFrame({
                        'Feature': features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    # Rename features for better readability
                    feature_names = {
                        'Month': 'Month of Year',
                        'DayOfWeek': 'Day of Week', 
                        'WeekOfYear': 'Week of Year',
                        'Quarter': 'Quarter',
                        'IsWeekend': 'Weekend Flag',
                        'Product_encoded': 'Product Type',
                        'Category_encoded': 'Category',
                        'Stock': 'Available Stock',
                        'PricePerUnit': 'Unit Price',
                        'WeightPerUnit': 'Unit Weight',
                        'Sales_MA_7': '7-Day Sales Average',
                        'Sales_MA_30': '30-Day Sales Average'
                    }
                    
                    feature_importance['Feature_Name'] = feature_importance['Feature'].map(feature_names)
                    
                    # Interactive feature importance chart
                    fig_importance = px.bar(
                        feature_importance.head(10),
                        x='Importance',
                        y='Feature_Name',
                        orientation='h',
                        title='Top 10 Most Important Factors for Sales Prediction',
                        labels={'Importance': 'Feature Importance Score', 'Feature_Name': 'Factor'},
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    fig_importance.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=500
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Feature importance insights
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🔍 Key Insights:**")
                        top_feature = feature_importance.iloc[0]
                        st.write(f"• Most important factor: **{top_feature['Feature_Name']}** ({top_feature['Importance']:.3f})")
                        
                        temporal_features = feature_importance[feature_importance['Feature'].isin(['Month', 'DayOfWeek', 'Quarter', 'IsWeekend'])]
                        temporal_importance = temporal_features['Importance'].sum()
                        st.write(f"• Seasonal factors influence: **{temporal_importance:.1%}** of prediction")
                        
                        business_features = feature_importance[feature_importance['Feature'].isin(['Stock', 'PricePerUnit'])]
                        business_importance = business_features['Importance'].sum()
                        st.write(f"• Business factors influence: **{business_importance:.1%}** of prediction")
                    
                    with col2:
                        st.write("**💡 Actionable Recommendations:**")
                        if temporal_importance > 0.3:
                            st.write("• Focus on seasonal planning and inventory management")
                        if business_importance > 0.2:
                            st.write("• Monitor stock levels and pricing strategies closely")
                        if top_feature['Feature'] in ['Sales_MA_7', 'Sales_MA_30']:
                            st.write("• Historical sales patterns are strong predictors")
                        
                        st.info("""
                        **🔍 Model Interpretation:**
                        - Higher importance = stronger influence on sales predictions
                        - Use these insights to focus on key business drivers
                        - Historical patterns often show highest importance
                        """)
                    
                except Exception as e:
                    st.error(f"❌ Error building forecast model: {str(e)}")
                    st.write("**Possible issues:**")
                    st.write("- Insufficient data (need at least 10 records)")
                    st.write("- Missing required columns")
                    st.write("- Data quality issues")
    
    else:
        st.warning("📁 Please upload and clean your data in the HOME page first.")
        st.info("""
        **To use the advanced forecasting feature:**
        1. Go to the HOME page
        2. Upload your Excel file with sales data
        3. Ensure your data contains: Product, Date, UnitsSold, Stock
        4. Return to this page to generate ML-powered forecasts
        """)

# ========== Sidebar Additional Features ==========
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data Tools")

if st.session_state.df_clean is not None:
    # Export options
    if st.sidebar.button("📥 Export Cleaned Data"):
        csv = st.session_state.df_clean.to_csv(index=False)
        st.sidebar.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name=f"ahva_cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    # Data quality report
    if st.sidebar.button("📋 Data Quality Report"):
        with st.expander("📊 Data Quality Summary", expanded=True):
            df = st.session_state.df_clean
            
            st.write("**Dataset Overview:**")
            st.write(f"- Total records: {len(df):,}")
            if 'Date' in df.columns and not df['Date'].isna().all():
                st.write(f"- Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
            if 'Product' in df.columns:
                st.write(f"- Unique products: {df['Product'].nunique()}")
            if 'Category' in df.columns:
                st.write(f"- Categories: {', '.join(df['Category'].unique())}")
            
            if 'UnitsSold' in df.columns:
                st.write("**Sales Statistics:**")
                st.write(f"- Total units sold: {df['UnitsSold'].sum():,}")
                st.write(f"- Average daily sales: {df['UnitsSold'].mean():.1f}")
                st.write(f"- Peak day sales: {df['UnitsSold'].max():,}")
            
            st.write("**Data Quality:**")
            missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
            if missing_pct.sum() == 0:
                st.write("✅ No missing data!")
            else:
                st.write("Missing data by column:")
                for col, pct in missing_pct[missing_pct > 0].items():
                    st.write(f"  - {col}: {pct}%")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**📦 Ahva Dashboard v2.0**")
st.sidebar.markdown("*Advanced ML Analytics Platform*")
st.sidebar.markdown("Built with ❤️ using Streamlit & Python")

# Debug mode
if st.sidebar.checkbox("🔧 Debug Mode"):
    st.sidebar.write("**Session State:**")
    st.sidebar.write(f"Raw data loaded: {st.session_state.df is not None}")
    st.sidebar.write(f"Clean data loaded: {st.session_state.df_clean is not None}")
    
    if st.session_state.df_clean is not None:
        st.sidebar.write(f"Rows: {len(st.session_state.df_clean)}")
        st.sidebar.write(f"Columns: {len(st.session_state.df_clean.columns)}")
        st.sidebar.write("**Available columns:**")
        for col in st.session_state.df_clean.columns:
            st.sidebar.write(f"  - {col}")
