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
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
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
                    # המרה ממספר סידורי של Excel
                    return pd.to_datetime('1899-12-30') + pd.Timedelta(days=date_val)
                except:
                    return pd.NaT
            try:
                return pd.to_datetime(date_val)
            except:
                return pd.NaT
        
        df_clean['Date'] = df_clean['Date'].apply(convert_date)
        
        # הסרת תאריכים לא תקינים
        invalid_dates = df_clean['Date'].isna().sum()
        if invalid_dates > 0:
            df_clean = df_clean.dropna(subset=['Date'])
    
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
            # החלפת ערכים שליליים בערך מוחלט
            negative_count = (df_clean[col] < 0).sum()
            if negative_count > 0:
                df_clean[col] = df_clean[col].abs()
    
    # טיפול בערכים חריגים במכירות
    if 'UnitsSold' in df_clean.columns and len(df_clean) > 10:
        Q1 = df_clean['UnitsSold'].quantile(0.25)
        Q3 = df_clean['UnitsSold'].quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold = Q3 + 3 * IQR  # גישה מתונה יותר
        
        outliers_count = (df_clean['UnitsSold'] > outlier_threshold).sum()
        if outliers_count > 0:
            # הסרת ערכים חריגים קיצוניים בלבד
            extreme_threshold = outlier_threshold * 2
            df_clean = df_clean[df_clean['UnitsSold'] <= extreme_threshold]
    
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
    df_forecast['PricePerUnit'] = df_forecast.get('מחיר ליחידה (₪)', 0)
    df_forecast['WeightPerUnit'] = df_forecast.get('משקל יחידה (גרם)', 0)
    
    # מילוי ערכים חסרים
    df_forecast['PricePerUnit'] = pd.to_numeric(df_forecast['PricePerUnit'], errors='coerce').fillna(0)
    df_forecast['WeightPerUnit'] = pd.to_numeric(df_forecast['WeightPerUnit'], errors='coerce').fillna(0)
    
    # פיצ'רי מכירות היסטוריות (חלון נע)
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
        raise ValueError("Not enough data for forecasting (minimum 10 records required)")
    
    features = [
        'Month', 'DayOfWeek', 'WeekOfYear', 'Quarter', 'IsWeekend',
        'Product_encoded', 'Category_encoded', 'Stock', 'PricePerUnit',
        'WeightPerUnit', 'Sales_MA_7', 'Sales_MA_30'
    ]
    
    # הכנת הנתונים
    X = df_forecast[features].fillna(0)
    y = df_forecast['UnitsSold']
    
    # וידוא שיש מספיק נתונים לחלוקה
    test_size = min(0.2, max(0.1, len(df_forecast) // 10))
    
    if len(df_forecast) > 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
    
    # בניית המודל
    n_estimators = min(100, max(10, len(X_train) // 2))  # התאמת מספר עצים לכמות הנתונים
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # הערכת המודל
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
                
                # ניקוי אוטומטי של הנתונים
                df_clean = clean_data(df)
                st.session_state.df_clean = df_clean
            
            st.success("✅ File uploaded and processed successfully!")
            
            # הצגת נתונים בסיסיים
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
            
            # הצגת דוגמה מהנתונים
            with st.expander("👀 Preview Your Data", expanded=False):
                st.write("**Sample of your cleaned data:**")
                st.dataframe(df_clean.head(10))
                
                st.write("**Column Information:**")
                col_info = pd.DataFrame({
                    'Column': df_clean.columns,
                    'Data Type': df_clean.dtypes,
                    'Non-Null Count': df_clean.count(),
                    'Sample Value': [str(df_clean[col].dropna().iloc[0]) if len(df_clean[col].dropna()) > 0 else 'N/A' for col in df_clean.columns]
                })
                st.dataframe(col_info)
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.write("Please ensure your Excel file contains the required columns: Product, Date, UnitsSold, Stock")

    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean

        # Date filter
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
            
            # Apply filter
            filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
            
            if len(filtered_df) == 0:
                st.warning("⚠️ No data found in the selected date range.")
                filtered_df = df  # Fall back to all data
        else:
            filtered_df = df
            st.info("ℹ️ Date column not available or contains invalid dates. Showing all data.")

        # KPI CALCULATIONS
        st.markdown("---")
        st.subheader("📊 Key Performance Indicators")
        
        # Calculate KPIs safely
        total_products = filtered_df['Product'].nunique() if 'Product' in filtered_df.columns else 0
        total_stock = int(filtered_df['Stock'].sum()) if 'Stock' in filtered_df.columns else 0
        total_demand = int(filtered_df['UnitsSold'].sum()) if 'UnitsSold' in filtered_df.columns else 0
        
        # Calculate shortages
        if 'UnitsSold' in filtered_df.columns and 'Stock' in filtered_df.columns:
            shortages = (filtered_df['UnitsSold'] > filtered_df['Stock']).sum()
            filtered_df["ShortageQty"] = (filtered_df["UnitsSold"] - filtered_df["Stock"]).clip(lower=0)
            missing_units = int(filtered_df["ShortageQty"].sum())
        else:
            shortages = 0
            missing_units = 0

        # Display KPIs
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
                <div class="kpi-title">🏷️ Products</div>
                <div class="kpi-value">{total_products:,}</div>
                <div class="kpi-subtext">Unique Items</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ========== ANALYSIS PAGE ==========
elif page == "📊 Analysis":
    st.markdown("<h1>📊 Sales & Demand Analysis</h1><hr>", unsafe_allow_html=True)

    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean.copy()

        # Check required columns
        required_cols = ['Category', 'UnitsSold']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.info("Available columns: " + ", ".join(df.columns))
        else:
            # Sales by Category
            st.subheader("🏷️ Sales Distribution by Category")
            category_sales = df.groupby("Category")["UnitsSold"].agg(['sum', 'mean', 'count']).reset_index()
            category_sales.columns = ['Category', 'Total_Sales', 'Avg_Sales', 'Records']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_cat = px.bar(
                    category_sales, 
                    x="Category", 
                    y="Total_Sales",
                    color="Total_Sales",
                    title="Total Units Sold per Category",
                    labels={"Total_Sales": "Total Units Sold"},
                    color_continuous_scale="Blues"
                )
                fig_cat.update_layout(showlegend=False)
                st.plotly_chart(fig_cat, use_container_width=True)
            
            with col2:
                fig_pie = px.pie(
                    category_sales, 
                    values="Total_Sales", 
                    names="Category",
                    title="Sales Distribution (%)"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Category statistics table
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
                fig_trend.update_traces(line_color='#667eea')
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Heatmap analysis
                if len(df) > 0:
                    st.subheader("🔥 Sales Heatmap: Day vs Month")
                    df['Day'] = df['Date'].dt.day_name()
                    df['Month'] = df['Date'].dt.month_name()
                    
                    pivot = df.pivot_table(index='Day', columns='Month', values='UnitsSold', aggfunc='sum', fill_value=0)
                    
                    # Order days of week
                    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    pivot = pivot.reindex([day for day in ordered_days if day in pivot.index])
                    
                    if not pivot.empty:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.heatmap(pivot, cmap='OrRd', annot=True, fmt='.0f', linewidths=0.5, ax=ax)
                        plt.title('Sales Heatmap: Peak Demand by Day and Month')
                        st.pyplot(fig)
                    else:
                        st.info("Not enough data for heatmap visualization")

            # Top performing products
            if 'Product' in df.columns:
                st.markdown("---")
                st.subheader("🏆 Top Performing Products")
                
                product_performance = df.groupby('Product')['UnitsSold'].agg(['sum', 'mean', 'count']).reset_index()
                product_performance.columns = ['Product', 'Total_Sales', 'Avg_Sales', 'Records']
                product_performance = product_performance.sort_values('Total_Sales', ascending=False).head(10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_top = px.bar(
                        product_performance,
                        x='Total_Sales',
                        y='Product',
                        orientation='h',
                        title='Top 10 Products by Total Sales',
                        labels={'Total_Sales': 'Total Units Sold'}
                    )
                    fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_top, use_container_width=True)
                
                with col2:
                    st.write("**📊 Top Products Table:**")
                    product_performance['Avg_Sales'] = product_performance['Avg_Sales'].round(1)
                    st.dataframe(product_performance, use_container_width=True)

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
            
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_product = st.selectbox("🏷️ Select Product for Analysis:", products)
            with col2:
                analysis_type = st.selectbox("📊 Analysis Type:", ["Monthly", "Weekly", "Daily"])
            
            # Filter data for selected product
            product_data = df[df['Product'] == selected_product].copy()
            
            if len(product_data) == 0:
                st.warning("⚠️ No data found for selected product.")
            else:
                # Monthly seasonality
                if analysis_type == "Monthly":
                    st.subheader(f"📅 Monthly Sales Pattern for {selected_product}")
                    
                    product_data['Month'] = product_data['Date'].dt.month
                    product_data['MonthName'] = product_data['Date'].dt.month_name()
                    monthly_sales = product_data.groupby(['Month', 'MonthName'])['UnitsSold'].agg(['sum', 'mean', 'count']).reset_index()
                    monthly_sales.columns = ['Month', 'MonthName', 'Total_Sales', 'Avg_Sales', 'Records']
                    
                    fig = px.line(
                        monthly_sales, 
                        x='MonthName', 
                        y='Total_Sales',
                        markers=True,
                        title=f"Monthly Sales Trend for {selected_product}",
                        labels={'Total_Sales': 'Total Units Sold', 'MonthName': 'Month'}
                    )
                    fig.update_traces(line_color='#667eea', marker_size=8)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Sales", f"{product_data['UnitsSold'].sum():,.0f}")
                    with col2:
                        st.metric("Peak Month", monthly_sales.loc[monthly_sales['Total_Sales'].idxmax(), 'MonthName'])
                    with col3:
                        st.metric("Avg Monthly", f"{monthly_sales['Total_Sales'].mean():.1f}")
                    with col4:
                        peak_ratio = monthly_sales['Total_Sales'].max() / monthly_sales['Total_Sales'].mean()
                        st.metric("Seasonality Index", f"{peak_ratio:.1f}x")
                
                # Weekly seasonality
                elif analysis_type == "Weekly":
                    st.subheader(f"📊 Weekly Sales Pattern for {selected_product}")
                    
                    product_data['DayOfWeek'] = product_data['Date'].dt.day_name()
                    weekly_sales = product_data.groupby('DayOfWeek')['UnitsSold'].agg(['sum', 'mean']).reset_index()
                    weekly_sales.columns = ['DayOfWeek', 'Total_Sales', 'Avg_Sales']
                    
                    # Order days
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    weekly_sales['DayOfWeek'] = pd.Categorical(weekly_sales['DayOfWeek'], categories=day_order, ordered=True)
                    weekly_sales = weekly_sales.sort_values('DayOfWeek')
                    
                    fig = px.bar(
                        weekly_sales,
                        x='DayOfWeek',
                        y='Total_Sales',
                        title=f"Weekly Sales Pattern for {selected_product}",
                        labels={'Total_Sales': 'Total Units Sold', 'DayOfWeek': 'Day of Week'},
                        color='Total_Sales',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Daily trend
                else:  # Daily
                    st.subheader(f"📈 Daily Sales Trend for {selected_product}")
                    
                    daily_sales = product_data.groupby('Date')['UnitsSold'].sum().reset_index()
                    
                    fig = px.scatter(
                        daily_sales,
                        x='Date',
                        y='UnitsSold',
                        title=f"Daily Sales Scatter for {selected_product}",
                        labels={'UnitsSold': 'Units Sold'},
                        trendline="ols"
                    )
                    fig.update_traces(marker_color='#667eea')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Seasonal insights
                st.markdown("---")
                st.subheader("🔍 Seasonal Insights")
                
                if 'Season' in product_data.columns:
                    seasonal_sales = product_data.groupby('Season')['UnitsSold'].agg(['sum', 'mean']).reset_index()
                    seasonal_sales.columns = ['Season', 'Total_Sales', 'Avg_Sales']
                    
                    fig_seasonal = px.pie(
                        seasonal_sales,
                        values='Total_Sales',
                        names='Season',
                        title=f"Seasonal Distribution for {selected_product}"
                    )
                    st.plotly_chart(fig_seasonal, use_container_width=True)
                else:
                    # Create seasons based on months
                    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                                 3: 'Spring', 4: 'Spring', 5: 'Spring',
                                 6: 'Summer', 7: 'Summer', 8: 'Summer',
                                 9: 'Fall', 10: 'Fall', 11: 'Fall'}
                    
                    product_data['Season'] = product_data['Date'].dt.month.map(season_map)
                    seasonal_sales = product_data.groupby('Season')['UnitsSold'].sum().reset_index()
                    
                    fig_seasonal = px.bar(
                        seasonal_sales,
                        x='Season',
                        y='UnitsSold',
                        title=f"Seasonal Sales for {selected_product}",
                        color='UnitsSold',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_seasonal, use_container_width=True)

    else:
        st.warning("📁 Please upload a file in the HOME page first.")

# ========== FORECASTING PAGE ==========
elif page == "🔮 Forecasting":
    st.markdown("<h1>🔮 Sales Forecasting</h1><hr>", unsafe_allow_html=True)
    
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean.copy()
        
        st.subheader("🤖 Machine Learning Sales Prediction")
        
        # Check if we have enough data
        if len(df) < 10:
            st.error("❌ Insufficient data for forecasting. Need at least 10 records.")
        else:
            with st.spinner("🧠 Building forecast model..."):
                try:
                    # Prepare data for forecasting
                    df_forecast = prepare_forecast_data(df)
                    
                    # Build model
                    model, features, mae, rmse = build_forecast_model(df_forecast)
                    
                    # Display model performance
                    st.success("✅ Forecast model built successfully!")
                    
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
                    
                    if st.button("🔮 Generate Forecast", type="primary"):
                        try:
                            # Get product data
                            product_data = df[df['Product'] == selected_product].iloc[-1]
                            
                            # Create future dates
                            last_date = df['Date'].max()
                            future_dates = pd.date_range(
                                start=last_date + timedelta(days=1), 
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
                            
                            # Create forecast visualization
                            st.markdown("---")
                            st.subheader(f"📈 Forecast Results for {selected_product}")
                            
                            # Historical vs Forecast chart
                            historical_data = df[df['Product'] == selected_product].tail(60)  # Last 60 days
                            
                            fig = go.Figure()
                            
                            # Historical data
                            fig.add_trace(go.Scatter(
                                x=historical_data['Date'],
                                y=historical_data['UnitsSold'],
                                mode='lines+markers',
                                name='Historical Sales',
                                line=dict(color='#2E86AB', width=2),
                                marker=dict(size=4)
                            ))
                            
                            # Forecast data
                            fig.add_trace(go.Scatter(
                                x=future_df['Date'],
                                y=future_df['Predicted_Sales'],
                                mode='lines+markers',
                                name='Forecast',
                                line=dict(color='#F24236', width=2, dash='dash'),
                                marker=dict(size=4)
                            ))
                            
                            # Add vertical line to separate historical from forecast
                            fig.add_vline(
                                x=last_date,
                                line_dash="dot",
                                line_color="gray",
                                annotation_text="Forecast Start"
                            )
                            
                            fig.update_layout(
                                title=f'Sales Forecast for {selected_product}',
                                xaxis_title='Date',
                                yaxis_title='Units Sold',
                                height=500,
                                showlegend=True,
                                hovermode='x unified'
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
                            st.subheader("💡 Business Recommendations")
                            
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
                                    </div>
                                    """.format(current_stock, total_forecast, recommended_stock, recommended_stock - current_stock), 
                                    unsafe_allow_html=True)
                                else:
                                    st.markdown("""
                                    <div class="forecast-highlight">
                                        <h4>✅ Stock Status: Good</h4>
                                        <p><strong>Current Stock:</strong> {:.0f} units</p>
                                        <p><strong>Forecasted Demand:</strong> {:.0f} units</p>
                                        <p><strong>Buffer Available:</strong> {:.0f} units</p>
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
                                        {"".join([f"<li>{date}</li>" for date in peak_dates])}
                                    </ul>
                                    <p><em>Consider increasing marketing efforts or ensuring adequate staff during these periods.</em></p>
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
                                    file_name=f'forecast_{selected_product}_{datetime.now().strftime("%Y%m%d")}.csv',
                                    mime='text/csv'
                                )
                            
                        except Exception as e:
                            st.error(f"❌ Error generating forecast: {str(e)}")
                    
                    # Feature importance analysis
                    st.markdown("---")
                    st.subheader("🎯 Model Feature Importance")
                    
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
                    
                    fig_importance = px.bar(
                        feature_importance.head(10),
                        x='Importance',
                        y='Feature_Name',
                        orientation='h',
                        title='Top 10 Most Important Factors for Sales Prediction',
                        labels={'Importance': 'Feature Importance', 'Feature_Name': 'Factor'},
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Interpretation
                    st.info("""
                    **🔍 How to interpret Feature Importance:**
                    - Higher values indicate more influence on sales predictions
                    - Historical sales averages typically have the highest importance
                    - Seasonal factors (month, quarter) often show significant impact
                    - Stock levels and pricing can be key business levers
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Error building forecast model: {str(e)}")
                    st.write("**Possible issues:**")
                    st.write("- Insufficient data (need at least 10 records)")
                    st.write("- Missing required columns")
                    st.write("- Data quality issues")
                    
                    if st.checkbox("🔧 Show debug information"):
                        st.write("**Available columns:**", list(df.columns))
                        st.write("**Data shape:**", df.shape)
                        st.write("**Data types:**", df.dtypes.to_dict())
    
    else:
        st.warning("📁 Please upload and clean your data in the HOME page first.")
        st.info("""
        **To use the forecasting feature:**
        1. Go to the HOME page
        2. Upload your Excel file with sales data
        3. Ensure your data contains: Product, Date, UnitsSold, Stock
        4. Return to this page to generate forecasts
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
st.sidebar.markdown("*Advanced Analytics Platform*")
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
