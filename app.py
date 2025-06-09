import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    .main > div { padding-top: 2rem; }
    .kpi-container {
        display: flex; gap: 15px; margin: 20px 0; flex-wrap: wrap;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 20px; border-radius: 10px; text-align: center;
        flex: 1; min-width: 200px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 15px rgba(0,0,0,0.2); }
    .kpi-blue { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .kpi-green { background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); }
    .kpi-red { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); }
    .kpi-purple { background: linear-gradient(135deg, #8360c3 0%, #2ebf91 100%); }
    .kpi-orange { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); }
    .kpi-title { font-size: 14px; margin-bottom: 10px; opacity: 0.9; font-weight: 500; }
    .kpi-value { font-size: 28px; font-weight: bold; margin: 10px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3); }
    .kpi-subtext { font-size: 12px; opacity: 0.8; }
    .sidebar-title { color: #2e4057; margin-bottom: 20px; font-weight: bold; text-align: center; }
    h1, h2, h3 { color: #2e4057; font-weight: 600; }
    hr { margin: 1rem 0; border: none; height: 2px; background: linear-gradient(90deg, #667eea, #764ba2); }
    .forecast-highlight {
        background: #f8f9fa; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #28a745; margin: 1rem 0;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem; border-radius: 8px; margin: 1rem 0;
        color: #2d3436; font-weight: 500;
    }
    @media (max-width: 768px) {
        .kpi-container { flex-direction: column; }
        .kpi-card { min-width: 100%; }
        .kpi-value { font-size: 24px; }
    }
</style>
""", unsafe_allow_html=True)

# ========== Enhanced Data Cleaning Functions ==========
@st.cache_data
def clean_data(df):
    """ניקוי וטיפול בנתונים פגומים - גרסה משופרת"""
    df_clean = df.copy()
    
    # טיפול בתאריכים - שיטה מחוזקת
    if 'Date' in df_clean.columns:
        def convert_date(date_val):
            if pd.isna(date_val):
                return pd.NaT
            
            # אם זה מספר (Excel serial number)
            if isinstance(date_val, (int, float)) and not pd.isna(date_val):
                try:
                    if 1 <= date_val <= 100000:
                        base_date = pd.to_datetime('1899-12-30')
                        # FIXED: Use pd.Timedelta instead of direct addition
                        return base_date + pd.Timedelta(days=int(date_val))
                except:
                    pass
            
            # ניסיון המרה רגילה
            try:
                converted = pd.to_datetime(date_val, errors='coerce')
                if pd.isna(converted):
                    return pd.NaT
                current_year = datetime.now().year
                if 2000 <= converted.year <= current_year + 1:
                    return converted
                else:
                    return pd.NaT
            except:
                return pd.NaT
        
        df_clean['Date'] = df_clean['Date'].apply(convert_date)
        
        invalid_dates = df_clean['Date'].isna().sum()
        if invalid_dates > 0:
            df_clean = df_clean.dropna(subset=['Date'])
            st.info(f"🧹 Removed {invalid_dates} rows with invalid dates")
    
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
        df_clean['Category'] = df_clean['Category'].replace(category_mapping).str.title()
    
    # הסרת שורות חסרות מידע חיוני
    critical_columns = [col for col in ['Product', 'Category', 'UnitsSold', 'Stock'] if col in df_clean.columns]
    before_cleaning = len(df_clean)
    df_clean = df_clean.dropna(subset=critical_columns)
    after_cleaning = len(df_clean)
    
    if before_cleaning != after_cleaning:
        st.info(f"🧹 Data Cleaning: Removed {before_cleaning - after_cleaning} rows with missing critical data")
    
    # טיפול מתקדם בערכים מספריים
    numeric_columns = ['UnitsSold', 'Stock', 'עלות ליחידה (₪)', 'מחיר ליחידה (₪)']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            negative_count = (df_clean[col] < 0).sum()
            if negative_count > 0:
                df_clean[col] = df_clean[col].abs()
                
            if col in ['UnitsSold', 'Stock']:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                extreme_high = df_clean[col] > upper_bound
                if extreme_high.sum() > 0:
                    df_clean.loc[extreme_high, col] = upper_bound
    
    if 'Category' in df_clean.columns:
        unique_categories = sorted(df_clean['Category'].unique())
        st.success(f"✅ Categories standardized: {', '.join(unique_categories)}")
    
    return df_clean

@st.cache_data
def prepare_forecast_data_enhanced(df):
    """הכנת נתונים לחיזוי - גרסה משופרת"""
    if len(df) == 0:
        return df
    
    df_forecast = df.copy()
    df_forecast = df_forecast.dropna(subset=['Date'])
    df_forecast = df_forecast.sort_values('Date')
    
    # פיצ'רי זמן מתקדמים
    df_forecast['Year'] = df_forecast['Date'].dt.year
    df_forecast['Month'] = df_forecast['Date'].dt.month
    df_forecast['DayOfWeek'] = df_forecast['Date'].dt.dayofweek
    df_forecast['WeekOfYear'] = df_forecast['Date'].dt.isocalendar().week
    df_forecast['Quarter'] = df_forecast['Date'].dt.quarter
    df_forecast['DayOfMonth'] = df_forecast['Date'].dt.day
    df_forecast['IsWeekend'] = df_forecast['DayOfWeek'].isin([5, 6]).astype(int)
    df_forecast['IsMonthStart'] = df_forecast['Date'].dt.is_month_start.astype(int)
    df_forecast['IsMonthEnd'] = df_forecast['Date'].dt.is_month_end.astype(int)
    
    # פיצ'רי מוצר מתקדמים
    df_forecast['Product_encoded'] = pd.Categorical(df_forecast['Product']).codes
    df_forecast['Category_encoded'] = pd.Categorical(df_forecast['Category']).codes
    
    # פיצ'רי מחיר ומשקל
    df_forecast['PricePerUnit'] = pd.to_numeric(df_forecast.get('מחיר ליחידה (₪)', 0), errors='coerce').fillna(0)
    df_forecast['WeightPerUnit'] = pd.to_numeric(df_forecast.get('משקל יחידה (גרם)', 0), errors='coerce').fillna(0)
    
    # פיצ'רי מכירות היסטוריות מתקדמים
    df_forecast = df_forecast.sort_values(['Product', 'Date'])
    
    for window in [3, 7, 14, 30]:
        df_forecast[f'Sales_MA_{window}'] = df_forecast.groupby('Product')['UnitsSold'].transform(
            lambda x: x.rolling(window=min(window, len(x)), min_periods=1).mean()
        )
    
    df_forecast['Sales_Trend_7'] = df_forecast.groupby('Product')['UnitsSold'].transform(
        lambda x: x.rolling(window=min(7, len(x)), min_periods=2).apply(
            lambda vals: np.polyfit(range(len(vals)), vals, 1)[0] if len(vals) > 1 else 0, raw=False
        )
    )
    
    df_forecast['Stock_Sales_Ratio'] = df_forecast['Stock'] / (df_forecast['UnitsSold'] + 1)
    
    category_avg = df_forecast.groupby('Category')['UnitsSold'].transform('mean')
    df_forecast['Product_vs_Category_Performance'] = df_forecast['UnitsSold'] / (category_avg + 1)
    
    return df_forecast

def build_enhanced_forecast_model(df_forecast):
    """בניית מודל חיזוי משופר"""
    if len(df_forecast) < 15:
        raise ValueError("Need at least 15 records for reliable forecasting")
    
    features = [
        'Month', 'DayOfWeek', 'WeekOfYear', 'Quarter', 'DayOfMonth',
        'IsWeekend', 'IsMonthStart', 'IsMonthEnd',
        'Product_encoded', 'Category_encoded', 
        'Stock', 'PricePerUnit', 'WeightPerUnit',
        'Sales_MA_3', 'Sales_MA_7', 'Sales_MA_14', 'Sales_MA_30',
        'Sales_Trend_7', 'Stock_Sales_Ratio', 'Product_vs_Category_Performance'
    ]
    
    available_features = [f for f in features if f in df_forecast.columns]
    
    X = df_forecast[available_features].fillna(0)
    y = df_forecast['UnitsSold']
    
    test_size = min(0.25, max(0.15, len(df_forecast) // 8))
    
    if len(df_forecast) > 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
    
    n_estimators = min(200, max(50, len(X_train) // 3))
    max_depth = min(15, max(5, len(X_train) // 10))
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=max(2, len(X_train) // 50),
        min_samples_leaf=max(1, len(X_train) // 100),
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, available_features, mae, rmse, r2

def simple_forecast_backup(df, product_name, days=30):
    """חיזוי פשוט לגיבוי במקרה של כשל במודל ML - FIXED DATE ISSUE"""
    product_data = df[df['Product'] == product_name].copy()
    
    if len(product_data) == 0:
        return None
    
    product_data = product_data.sort_values('Date')
    
    if len(product_data) >= 7:
        recent_sales = product_data['UnitsSold'].tail(7).mean()
        older_sales = product_data['UnitsSold'].head(7).mean()
        if older_sales > 0:
            growth_rate = (recent_sales - older_sales) / older_sales
        else:
            growth_rate = 0
    else:
        growth_rate = 0
    
    base_forecast = product_data['UnitsSold'].tail(14).mean()
    
    forecast_data = []
    last_date = product_data['Date'].max()
    
    # FIXED: Use string conversion to avoid date arithmetic issues
    for i in range(1, days + 1):
        # Convert to timestamp, add days, then back to datetime
        future_date = pd.Timestamp(last_date) + pd.Timedelta(days=i)
        
        predicted_sales = base_forecast * (1 + growth_rate * i / 30)
        
        day_of_week = future_date.dayofweek
        month = future_date.month
        
        if day_of_week in [5, 6]:
            predicted_sales *= 0.85
        elif day_of_week in [1, 2]:
            predicted_sales *= 1.1
        
        seasonal_factors = {
            1: 0.9, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.05, 6: 1.1,
            7: 1.15, 8: 1.1, 9: 1.05, 10: 1.0, 11: 0.95, 12: 1.2
        }
        predicted_sales *= seasonal_factors.get(month, 1.0)
        
        predicted_sales = max(0, predicted_sales)
        
        forecast_data.append({
            'Date': future_date,
            'Predicted_Sales': predicted_sales
        })
    
    return pd.DataFrame(forecast_data)

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

    uploaded_file = st.file_uploader("📁 Upload Excel File", type=["xlsx", "xls"], help="Upload your Ahva sales data file")

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

        if 'Category' not in df.columns or 'UnitsSold' not in df.columns:
            st.error("❌ Missing required columns: Category, UnitsSold")
        else:
            # Sales by Category with Interactive Plotly Charts
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
                    title="📊 Total Units Sold per Category",
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
                    title="🥧 Sales Distribution (%)",
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
                    title='📈 Daily Sales Trend',
                    labels={'UnitsSold': 'Units Sold'}
                )
                fig_trend.update_traces(line_color='#1f77b4', line_width=3)
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
                        title="📅 Sales by Day of Week",
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
                        title='🏆 Top 10 Products by Sales',
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
        
        if 'Product' not in df.columns or 'UnitsSold' not in df.columns or 'Date' not in df.columns:
            st.error("❌ Missing required columns: Product, UnitsSold, Date")
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
                    title=f"📈 Monthly Sales Pattern for {selected_product}",
                    labels={'Total_Sales': 'Total Units Sold', 'MonthName': 'Month'}
                )
                fig_monthly.update_traces(line_color='#1e88e5', marker_size=10, line_width=4)
                fig_monthly.update_layout(height=400)
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Sales", f"{product_data['UnitsSold'].sum():,.0f}")
                with col2:
                    if len(monthly_sales) > 0:
                        peak_month = monthly_sales.loc[monthly_sales['Total_Sales'].idxmax(), 'MonthName']
                        st.metric("Peak Month", peak_month)
                with col3:
                    avg_monthly = monthly_sales['Total_Sales'].mean()
                    st.metric("Avg Monthly", f"{avg_monthly:.1f}")
                with col4:
                    if len(monthly_sales) > 0:
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
                    title=f"📊 Weekly Sales Pattern for {selected_product}",
                    color='UnitsSold',
                    color_continuous_scale='Blues'
                )
                fig_weekly.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_weekly, use_container_width=True)

    else:
        st.warning("📁 Please upload a file in the HOME page first.")

# ========== ENHANCED FORECASTING PAGE ==========
elif page == "🔮 Forecasting":
    st.markdown("<h1>🔮 Enhanced ML Sales Forecasting</h1><hr>", unsafe_allow_html=True)
    
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean.copy()
        
        st.subheader("🤖 Advanced Machine Learning Prediction Engine")
        
        if len(df) < 15:
            st.error("❌ Insufficient data for reliable ML forecasting. Need at least 15 records.")
            st.info("💡 Try uploading more historical data for better predictions.")
        else:
            # Model selection
            st.write("**🎯 Select Forecasting Method:**")
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox("Choose Model:", 
                    ["🤖 Advanced ML (Recommended)", "📊 Statistical Backup"],
                    help="Advanced ML uses Random Forest with 20+ features. Statistical backup uses trend analysis."
                )
            
            with col2:
                confidence_level = st.selectbox("Confidence Level:", 
                    ["High (±10%)", "Medium (±15%)", "Low (±20%)"],
                    index=1,
                    help="Higher confidence = narrower prediction bands"
                )
            
            # Extract confidence percentage
            confidence_pct = {"High (±10%)": 0.10, "Medium (±15%)": 0.15, "Low (±20%)": 0.20}[confidence_level]
            
            if model_type == "🤖 Advanced ML (Recommended)":
                with st.spinner("🧠 Building enhanced Random Forest model with 20+ features..."):
                    try:
                        # Prepare enhanced data
                        df_forecast = prepare_forecast_data_enhanced(df)
                        
                        # Build enhanced model
                        model, features, mae, rmse, r2 = build_enhanced_forecast_model(df_forecast)
                        
                        # Display enhanced model performance
                        st.success("✅ Enhanced Random Forest model trained successfully!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🎯 MAE", f"{mae:.2f}", help="Mean Absolute Error")
                        with col2:
                            st.metric("📊 RMSE", f"{rmse:.2f}", help="Root Mean Square Error")
                        with col3:
                            st.metric("🔥 R² Score", f"{r2:.3f}", help="Explained Variance (closer to 1 = better)")
                        with col4:
                            avg_sales = df['UnitsSold'].mean()
                            accuracy_pct = max(0, (1 - mae/avg_sales) * 100)
                            st.metric("✅ Accuracy", f"{accuracy_pct:.1f}%", help="Prediction Accuracy")
                        
                        # Model quality assessment
                        if r2 > 0.8:
                            st.success("🏆 Excellent model quality! High confidence in predictions.")
                        elif r2 > 0.6:
                            st.info("✅ Good model quality. Reliable predictions expected.")
                        elif r2 > 0.4:
                            st.warning("⚠️ Moderate model quality. Use predictions with caution.")
                        else:
                            st.error("❌ Poor model quality. Consider using Statistical Backup method.")
                        
                        use_ml_model = True
                        
                    except Exception as e:
                        st.error(f"❌ ML model failed: {str(e)}")
                        st.info("🔄 Falling back to Statistical method...")
                        use_ml_model = False
            else:
                use_ml_model = False
                st.info("📊 Using Statistical forecasting method with trend analysis.")
            
            # Product selection for forecasting
            st.markdown("---")
            st.subheader("📊 Generate Forecast")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_product = st.selectbox("🏷️ Select Product:", df['Product'].unique())
            with col2:
                forecast_days = st.slider("📅 Forecast Period (days):", 7, 120, 30)
            with col3:
                show_confidence = st.checkbox("📈 Show Confidence Bands", value=True)
            
            if st.button("🔮 Generate Enhanced Forecast", type="primary"):
                try:
                    # Product data validation
                    product_data = df[df['Product'] == selected_product]
                    if len(product_data) < 5:
                        st.error(f"❌ Insufficient data for {selected_product}. Need at least 5 records.")
                        st.stop()
                    
                    product_info = product_data.iloc[-1]
                    
                    # Show debug info
                    st.info(f"🔍 Debug: Found {len(product_data)} records for {selected_product}")
                    
                    if use_ml_model and 'model' in locals():
                        # Advanced ML Forecasting
                        st.markdown("### 🤖 Advanced ML Forecast Results")
                        
                        try:
                            # Create future dates - MULTIPLE METHODS TO TRY
                            last_date = df['Date'].max()
                            st.info(f"🔍 Last date in data: {last_date}")
                            
                            # Method 1: Simple range
                            try:
                                future_dates = pd.date_range(
                                    start=last_date,
                                    periods=forecast_days + 1,
                                    freq='D'
                                )[1:]  # Skip first date (last_date)
                                st.success("✅ Method 1: pd.date_range worked")
                            except Exception as e1:
                                st.warning(f"⚠️ Method 1 failed: {e1}")
                                
                                # Method 2: Manual creation
                                try:
                                    future_dates = []
                                    base_date = pd.to_datetime(last_date)
                                    for i in range(1, forecast_days + 1):
                                        future_dates.append(base_date + pd.Timedelta(days=i))
                                    st.success("✅ Method 2: Manual creation worked")
                                except Exception as e2:
                                    st.error(f"❌ Method 2 also failed: {e2}")
                                    st.stop()
                            
                            # Prepare future data for ML model
                            future_data = []
                            for i, date in enumerate(future_dates):
                                try:
                                    row = {
                                        'Date': date,
                                        'Product': selected_product,
                                        'Month': date.month,
                                        'DayOfWeek': date.dayofweek,
                                        'WeekOfYear': date.isocalendar().week,
                                        'Quarter': date.quarter,
                                        'DayOfMonth': date.day,
                                        'IsWeekend': 1 if date.dayofweek >= 5 else 0,
                                        'IsMonthStart': 1 if date.day == 1 else 0,
                                        'IsMonthEnd': 1 if date.is_month_end else 0,
                                        'Product_encoded': pd.Categorical([selected_product], categories=df['Product'].unique()).codes[0],
                                        'Category_encoded': pd.Categorical([product_info['Category']], categories=df['Category'].unique()).codes[0],
                                        'Stock': product_info['Stock'],
                                        'PricePerUnit': product_info.get('מחיר ליחידה (₪)', 0),
                                        'WeightPerUnit': product_info.get('משקל יחידה (גרם)', 0),
                                        'Sales_MA_3': product_data['UnitsSold'].tail(3).mean(),
                                        'Sales_MA_7': product_data['UnitsSold'].tail(7).mean(),
                                        'Sales_MA_14': product_data['UnitsSold'].tail(14).mean(),
                                        'Sales_MA_30': product_data['UnitsSold'].tail(30).mean(),
                                        'Sales_Trend_7': 0,
                                        'Stock_Sales_Ratio': product_info['Stock'] / (product_data['UnitsSold'].tail(7).mean() + 1),
                                        'Product_vs_Category_Performance': 1.0
                                    }
                                    future_data.append(row)
                                except Exception as e3:
                                    st.error(f"❌ Error creating row {i}: {e3}")
                                    st.stop()
                            
                            future_df = pd.DataFrame(future_data)
                            st.success(f"✅ Created {len(future_df)} future data points")
                            
                            # Generate ML predictions
                            X_future = future_df[features].fillna(0)
                            predictions = model.predict(X_future)
                            predictions = np.maximum(predictions, 0)  # No negative predictions
                            
                        except Exception as date_error:
                            st.error(f"❌ Date creation error: {date_error}")
                            st.info("🔄 Falling back to Statistical method...")
                            use_ml_model = False
                        
                        # DEBUG: Let's see what the model is getting
                        st.markdown("### 🔍 Debug Information")
                        
                        # Show some of the future data the model sees
                        with st.expander("🔍 Click to see what data the model receives"):
                            debug_cols = ['Date', 'Month', 'DayOfWeek', 'IsWeekend', 'Stock', 'Sales_MA_7']
                            available_debug_cols = [col for col in debug_cols if col in future_df.columns]
                            st.write("**Sample of future data (first 7 days):**")
                            st.dataframe(future_df[available_debug_cols].head(7))
                            
                            st.write("**Predictions for each day:**")
                            debug_pred = future_df[['Date', 'Predicted_Sales']].head(10).copy()
                            debug_pred['Date'] = debug_pred['Date'].dt.strftime('%Y-%m-%d (%A)')
                            st.dataframe(debug_pred)
                        
                        # Check if predictions are too similar
                        pred_std = future_df['Predicted_Sales'].std()
                        pred_range = future_df['Predicted_Sales'].max() - future_df['Predicted_Sales'].min()
                        
                        if pred_std < 1:
                            st.warning(f"""
                            ⚠️ **Model Issue Detected:** 
                            - Predictions are too similar (standard deviation: {pred_std:.2f})
                            - Range between min/max: {pred_range:.1f}
                            - This suggests the model isn't using day-of-week or seasonal patterns properly
                            """)
                            
                            # Let's try to see why
                            st.write("**🔍 Potential causes:**")
                            st.write("1. All input features might be too similar")
                            st.write("2. Model might be defaulting to average value")
                            st.write("3. Feature importance might be dominated by constant values")
                            
                            # Show feature values for first few days
                            if len(future_df) >= 7:
                                sample_features = future_df[['DayOfWeek', 'Month', 'IsWeekend', 'Stock']].head(7)
                                st.write("**Feature variation (first 7 days):**")
                                st.dataframe(sample_features)
                        
                        else:
                            st.success(f"✅ Model predictions look varied (std: {pred_std:.2f}, range: {pred_range:.1f})")
                        
                    else:
                        # Statistical Backup Forecasting
                        st.markdown("### 📊 Statistical Forecast Results")
                        future_df = simple_forecast_backup(df, selected_product, forecast_days)
                        
                        if future_df is None:
                            st.error("❌ Could not generate forecast for selected product.")
                            st.stop()
                    
                    # Simple basic chart - like the original versions
                    st.markdown("---")
                    st.subheader(f"📈 Forecast Visualization for {selected_product}")
                    
                    # Historical data (last 30 days for clarity)
                    historical_data = product_data.tail(30)
                    
                    fig = go.Figure()
                    
                    # Historical data - simple blue line
                    fig.add_trace(go.Scatter(
                        x=historical_data['Date'],
                        y=historical_data['UnitsSold'],
                        mode='lines+markers',
                        name='Historical Sales',
                        line=dict(color='blue', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Forecast data - simple red line
                    fig.add_trace(go.Scatter(
                        x=future_df['Date'],
                        y=future_df['Predicted_Sales'],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=4)
                    ))
                    
                    # Basic layout
                    fig.update_layout(
                        title=f'Sales Forecast for {selected_product}',
                        xaxis_title='Date',
                        yaxis_title='Units Sold',
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Enhanced forecast summary
                    st.subheader("📊 Enhanced Forecast Summary")
                    
                    total_forecast = future_df['Predicted_Sales'].sum()
                    avg_daily_forecast = future_df['Predicted_Sales'].mean()
                    max_daily_forecast = future_df['Predicted_Sales'].max()
                    min_daily_forecast = future_df['Predicted_Sales'].min()
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("🎯 Total Forecast", f"{total_forecast:.0f}")
                    with col2:
                        st.metric("📊 Daily Average", f"{avg_daily_forecast:.1f}")
                    with col3:
                        st.metric("📈 Peak Day", f"{max_daily_forecast:.0f}")
                    with col4:
                        st.metric("📉 Low Day", f"{min_daily_forecast:.0f}")
                    with col5:
                        volatility = future_df['Predicted_Sales'].std()
                        st.metric("📊 Volatility", f"{volatility:.1f}")
                    
                    # Enhanced Business Intelligence
                    st.markdown("---")
                    st.subheader("💡 Enhanced Business Intelligence")
                    
                    current_stock = product_info['Stock']
                    safety_buffer = 1.2  # 20% safety buffer
                    recommended_stock = total_forecast * safety_buffer
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Stock recommendations
                        if current_stock < recommended_stock:
                            shortage = recommended_stock - current_stock
                            urgency = "🚨 CRITICAL" if shortage > total_forecast * 0.5 else "⚠️ Important"
                            
                            st.markdown(f"""
                            <div class="recommendation-box">
                                <h4>{urgency} Stock Alert</h4>
                                <p><strong>Current Stock:</strong> {current_stock:.0f} units</p>
                                <p><strong>Forecasted Demand:</strong> {total_forecast:.0f} units</p>
                                <p><strong>Recommended Stock:</strong> {recommended_stock:.0f} units</p>
                                <p><strong>🛒 Order Needed:</strong> {shortage:.0f} units</p>
                                <p><strong>📅 Days Until Stockout:</strong> {(current_stock / avg_daily_forecast):.1f} days</p>
                                <p><em>🎯 Priority: {"High" if shortage > total_forecast * 0.5 else "Medium"}</em></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            buffer_days = (current_stock - total_forecast) / avg_daily_forecast
                            st.markdown(f"""
                            <div class="forecast-highlight">
                                <h4>✅ Stock Status: Optimal</h4>
                                <p><strong>Current Stock:</strong> {current_stock:.0f} units</p>
                                <p><strong>Forecasted Demand:</strong> {total_forecast:.0f} units</p>
                                <p><strong>Safety Buffer:</strong> {current_stock - total_forecast:.0f} units</p>
                                <p><strong>📅 Buffer Duration:</strong> {buffer_days:.1f} days</p>
                                <p><em>🎯 Status: No immediate action needed</em></p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        # Peak analysis and scheduling
                        peak_days = future_df.nlargest(5, 'Predicted_Sales')[['Date', 'Predicted_Sales']]
                        peak_info = []
                        for _, row in peak_days.iterrows():
                            date_str = row['Date'].strftime('%m/%d (%a)')
                            sales = row['Predicted_Sales']
                            peak_info.append(f"<li><strong>{date_str}:</strong> {sales:.0f} units</li>")
                        
                        avg_sales = product_data['UnitsSold'].tail(30).mean()
                        growth_trend = "📈 Growing" if avg_daily_forecast > avg_sales else "📉 Declining" if avg_daily_forecast < avg_sales * 0.9 else "➡️ Stable"
                        
                        st.markdown(f"""
                        <div class="forecast-highlight">
                            <h4>📈 Peak Sales Analysis</h4>
                            <p><strong>Sales Trend:</strong> {growth_trend}</p>
                            <p><strong>Top 5 Peak Days:</strong></p>
                            <ul>{"".join(peak_info)}</ul>
                            <p><em>🎯 Action: Schedule extra staff & marketing</em></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Feature importance (only for ML models)
                    if use_ml_model and 'model' in locals():
                        st.markdown("---")
                        st.subheader("🎯 ML Model Feature Importance")
                        
                        feature_importance = pd.DataFrame({
                            'Feature': features,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        # Enhanced feature names
                        feature_names = {
                            'Month': '📅 Month', 'DayOfWeek': '📆 Day of Week', 'WeekOfYear': '🗓️ Week',
                            'Quarter': '📊 Quarter', 'DayOfMonth': '📋 Day of Month',
                            'IsWeekend': '🏖️ Weekend', 'IsMonthStart': '🗓️ Month Start', 'IsMonthEnd': '🗓️ Month End',
                            'Product_encoded': '🏷️ Product', 'Category_encoded': '📂 Category',
                            'Stock': '📦 Stock', 'PricePerUnit': '💰 Price', 'WeightPerUnit': '⚖️ Weight',
                            'Sales_MA_3': '📈 3-Day Avg', 'Sales_MA_7': '📈 7-Day Avg', 
                            'Sales_MA_14': '📈 14-Day Avg', 'Sales_MA_30': '📈 30-Day Avg',
                            'Sales_Trend_7': '📈 7-Day Trend', 'Stock_Sales_Ratio': '📊 Stock/Sales Ratio',
                            'Product_vs_Category_Performance': '🎯 Product Performance'
                        }
                        
                        feature_importance['Feature_Name'] = feature_importance['Feature'].map(
                            lambda x: feature_names.get(x, x)
                        )
                        
                        # Interactive chart
                        fig_importance = px.bar(
                            feature_importance.head(12),
                            x='Importance',
                            y='Feature_Name',
                            orientation='h',
                            title='🎯 Top 12 Most Important Factors',
                            color='Importance',
                            color_continuous_scale='Viridis'
                        )
                        fig_importance.update_layout(
                            yaxis={'categoryorder': 'total ascending'},
                            height=500
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                        
                        # Insights
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**🔍 Key Insights:**")
                            top_3 = feature_importance.head(3)
                            for i, row in top_3.iterrows():
                                st.write(f"• **{row['Feature_Name']}:** {row['Importance']:.3f}")
                        
                        with col2:
                            # Calculate category impacts
                            temporal_features = ['Month', 'DayOfWeek', 'Quarter', 'WeekOfYear']
                            temporal_importance = feature_importance[
                                feature_importance['Feature'].isin(temporal_features)
                            ]['Importance'].sum()
                            
                            historical_features = ['Sales_MA_3', 'Sales_MA_7', 'Sales_MA_14', 'Sales_MA_30']
                            historical_importance = feature_importance[
                                feature_importance['Feature'].isin(historical_features)
                            ]['Importance'].sum()
                            
                            st.write("**📊 Factor Categories:**")
                            st.write(f"• **Seasonal Impact:** {temporal_importance:.1%}")
                            st.write(f"• **Historical Patterns:** {historical_importance:.1%}")
                            st.write(f"• **Business Factors:** {(1-temporal_importance-historical_importance):.1%}")
                    
                    # Download options
                    with st.expander("📋 Download Forecast Data", expanded=False):
                        display_df = future_df[['Date', 'Predicted_Sales']].copy()
                        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                        display_df['Predicted_Sales'] = display_df['Predicted_Sales'].round(1)
                        display_df.columns = ['📅 Date', f'🔮 Predicted Sales ({method_text})']
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download {method_text} Forecast CSV",
                            data=csv,
                            file_name=f'{method_text.lower()}_forecast_{selected_product}_{datetime.now().strftime("%Y%m%d")}.csv',
                            mime='text/csv'
                        )
                
                except Exception as e:
                    st.error(f"❌ Error generating forecast: {str(e)}")
                    st.write("**Debug Info:**")
                    st.write(f"- Product: {selected_product}")
                    st.write(f"- Data points: {len(product_data)}")
                    st.write(f"- Date range: {product_data['Date'].min()} to {product_data['Date'].max()}")
    
    else:
        st.warning("📁 Please upload and clean your data in the HOME page first.")
        st.info("""
        **🔮 To use Enhanced ML Forecasting:**
        1. Go to the **HOME** page
        2. Upload your Excel file with sales data
        3. Ensure data contains: Product, Date, UnitsSold, Stock
        4. Return here for AI-powered forecasts with Random Forest ML
        """)

# ========== Sidebar ==========
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data Tools")

if st.session_state.df_clean is not None:
    if st.sidebar.button("📥 Export Data"):
        csv = st.session_state.df_clean.to_csv(index=False)
        st.sidebar.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name=f"ahva_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Data quality report
    if st.sidebar.button("📋 Data Quality Report"):
        with st.expander("📊 Data Quality Summary", expanded=True):
            df = st.session_state.df_clean
            
            st.write("**📈 Dataset Overview:**")
            st.write(f"- Total records: {len(df):,}")
            if 'Date' in df.columns and not df['Date'].isna().all():
                st.write(f"- Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
            if 'Product' in df.columns:
                st.write(f"- Unique products: {df['Product'].nunique()}")
            if 'Category' in df.columns:
                st.write(f"- Categories: {', '.join(df['Category'].unique())}")
            
            if 'UnitsSold' in df.columns:
                st.write("**💰 Sales Statistics:**")
                st.write(f"- Total units sold: {df['UnitsSold'].sum():,}")
                st.write(f"- Average daily sales: {df['UnitsSold'].mean():.1f}")
                st.write(f"- Peak day sales: {df['UnitsSold'].max():,}")
            
            st.write("**✅ Data Quality:**")
            missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
            if missing_pct.sum() == 0:
                st.write("🎯 Perfect! No missing data detected!")
            else:
                st.write("Missing data by column:")
                for col, pct in missing_pct[missing_pct > 0].items():
                    st.write(f"  - {col}: {pct}%")

st.sidebar.markdown("---")
st.sidebar.markdown("**📦 Ahva Dashboard v2.1**")
st.sidebar.markdown("*🤖 Enhanced ML Platform*")
st.sidebar.markdown("Built with ❤️ using Streamlit & scikit-learn")

# Debug mode
if st.sidebar.checkbox("🔧 Debug Mode"):
    st.sidebar.write("**🛠️ Technical Details:**")
    st.sidebar.write(f"Data loaded: {st.session_state.df_clean is not None}")
    
    if st.session_state.df_clean is not None:
        st.sidebar.write(f"Rows: {len(st.session_state.df_clean):,}")
        st.sidebar.write(f"Columns: {len(st.session_state.df_clean.columns)}")

# Success indicator
if st.session_state.df_clean is not None:
    st.sidebar.success("✅ Enhanced Dashboard Ready!")
    st.sidebar.info("🔮 ML Forecasting Active")
