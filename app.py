import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
    .highlight-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        color: #856404;
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
    
    return df_clean

def simple_forecast(df, product_name, days=30):
    """חיזוי פשוט בלי Machine Learning"""
    product_data = df[df['Product'] == product_name]
    
    if len(product_data) == 0:
        return None
    
    # חישוב ממוצע מכירות
    avg_sales = product_data['UnitsSold'].mean()
    
    # חישוב מגמה (אם יש מספיק נתונים)
    if len(product_data) > 7:
        recent_avg = product_data.tail(7)['UnitsSold'].mean()
        older_avg = product_data.head(7)['UnitsSold'].mean()
        trend = (recent_avg - older_avg) / len(product_data) if older_avg > 0 else 0
    else:
        trend = 0
    
    # יצירת חיזוי
    forecast_data = []
    last_date = df['Date'].max()
    
    # וידוא שלast_date הוא Timestamp
    if not isinstance(last_date, pd.Timestamp):
        last_date = pd.to_datetime(last_date)
    
    for i in range(1, days + 1):
        # שימוש ב-pd.Timedelta במקום timedelta רגיל
        future_date = last_date + pd.Timedelta(days=i)
        
        # חיזוי בסיסי עם מגמה
        predicted_sales = max(0, avg_sales + (trend * i))
        
        # התאמה לעונתיות (יום בשבוע)
        day_of_week = future_date.weekday()
        if day_of_week in [5, 6]:  # סוף שבוע
            predicted_sales *= 0.8
        
        forecast_data.append({
            'Date': future_date,
            'Predicted_Sales': predicted_sales
        })
    
    return pd.DataFrame(forecast_data)

# ========== Navigation ==========
st.sidebar.markdown("<h2 class='sidebar-title'>🧭 Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to:", ["🏠 HOME", "📊 Analysis", "📈 Seasonality", "🔮 Basic Forecast"])

# ========== Session State ==========
if "df" not in st.session_state:
    st.session_state.df = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

# ========== HOME PAGE ==========
if page == "🏠 HOME":
    st.markdown("""
    <h1 style='margin-bottom: 10px; text-align: center;'>📦 Ahva Inventory Dashboard</h1>
    <p style='text-align: center; font-size: 18px; color: #666;'>Sales Analytics & Basic Forecasting Platform</p>
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

        # Calculate efficiency metrics
        if total_stock > 0 and total_demand > 0:
            efficiency = (total_demand / total_stock) * 100
        else:
            efficiency = 0

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

        # Check required columns
        required_cols = ['Category', 'UnitsSold']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.info("Available columns: " + ", ".join(df.columns))
        else:
            # Sales by Category
            st.subheader("🏷️ Sales by Category")
            category_sales = df.groupby("Category")["UnitsSold"].agg(['sum', 'mean', 'count']).reset_index()
            category_sales.columns = ['Category', 'Total_Sales', 'Avg_Sales', 'Records']
            
            # Display as table and charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Category Performance:**")
                category_sales['Avg_Sales'] = category_sales['Avg_Sales'].round(1)
                st.dataframe(category_sales, use_container_width=True)
            
            with col2:
                st.write("**Sales Distribution:**")
                for idx, row in category_sales.iterrows():
                    percentage = (row['Total_Sales'] / category_sales['Total_Sales'].sum()) * 100
                    st.write(f"**{row['Category']}:** {row['Total_Sales']:,} units ({percentage:.1f}%)")

            # Time-based analysis (simple)
            if 'Date' in df.columns and not df['Date'].isna().all():
                st.markdown("---")
                st.subheader("📈 Sales Over Time")
                
                # Group by month
                df['YearMonth'] = df['Date'].dt.to_period('M')
                monthly_sales = df.groupby('YearMonth')['UnitsSold'].sum().reset_index()
                monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)
                
                st.write("**Monthly Sales Summary:**")
                for idx, row in monthly_sales.iterrows():
                    st.write(f"**{row['YearMonth']}:** {row['UnitsSold']:,} units")
                
                # Simple line chart using st.line_chart
                if len(monthly_sales) > 1:
                    chart_data = monthly_sales.set_index('YearMonth')
                    st.line_chart(chart_data['UnitsSold'])

            # Top performing products
            if 'Product' in df.columns:
                st.markdown("---")
                st.subheader("🏆 Top Products")
                
                product_performance = df.groupby('Product')['UnitsSold'].agg(['sum', 'mean', 'count']).reset_index()
                product_performance.columns = ['Product', 'Total_Sales', 'Avg_Sales', 'Records']
                product_performance = product_performance.sort_values('Total_Sales', ascending=False).head(10)
                product_performance['Avg_Sales'] = product_performance['Avg_Sales'].round(1)
                
                st.write("**Top 10 Products by Total Sales:**")
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
            selected_product = st.selectbox("🏷️ Select Product for Analysis:", products)
            
            # Filter data for selected product
            product_data = df[df['Product'] == selected_product].copy()
            
            if len(product_data) == 0:
                st.warning("⚠️ No data found for selected product.")
            else:
                st.subheader(f"📅 Seasonality for {selected_product}")
                
                # Monthly analysis
                product_data['Month'] = product_data['Date'].dt.month
                product_data['MonthName'] = product_data['Date'].dt.month_name()
                monthly_sales = product_data.groupby(['Month', 'MonthName'])['UnitsSold'].agg(['sum', 'mean', 'count']).reset_index()
                monthly_sales.columns = ['Month', 'MonthName', 'Total_Sales', 'Avg_Sales', 'Records']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Monthly Sales:**")
                    for idx, row in monthly_sales.iterrows():
                        st.write(f"**{row['MonthName']}:** {row['Total_Sales']:,} units")
                
                with col2:
                    # Statistics
                    st.write("**Statistics:**")
                    st.metric("Total Sales", f"{product_data['UnitsSold'].sum():,.0f}")
                    
                    if len(monthly_sales) > 0:
                        peak_month = monthly_sales.loc[monthly_sales['Total_Sales'].idxmax(), 'MonthName']
                        st.metric("Peak Month", peak_month)
                        
                        avg_monthly = monthly_sales['Total_Sales'].mean()
                        st.metric("Avg Monthly", f"{avg_monthly:.1f}")
                
                # Simple bar chart
                if len(monthly_sales) > 1:
                    chart_data = monthly_sales.set_index('MonthName')['Total_Sales']
                    st.bar_chart(chart_data)
                
                # Weekly pattern
                st.markdown("---")
                st.subheader("📊 Weekly Pattern")
                
                product_data['DayOfWeek'] = product_data['Date'].dt.day_name()
                weekly_sales = product_data.groupby('DayOfWeek')['UnitsSold'].sum().reset_index()
                
                # Order days
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekly_sales['DayOfWeek'] = pd.Categorical(weekly_sales['DayOfWeek'], categories=day_order, ordered=True)
                weekly_sales = weekly_sales.sort_values('DayOfWeek')
                
                st.write("**Sales by Day of Week:**")
                for idx, row in weekly_sales.iterrows():
                    st.write(f"**{row['DayOfWeek']}:** {row['UnitsSold']:,} units")
                
                # Chart
                chart_data = weekly_sales.set_index('DayOfWeek')['UnitsSold']
                st.bar_chart(chart_data)

    else:
        st.warning("📁 Please upload a file in the HOME page first.")

# ========== BASIC FORECAST PAGE ==========
elif page == "🔮 Basic Forecast":
    st.markdown("<h1>🔮 Basic Sales Forecasting</h1><hr>", unsafe_allow_html=True)
    
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean.copy()
        
        st.info("📊 This is a simplified forecasting tool using statistical methods. For advanced ML forecasting, install the required libraries.")
        
        # Check if we have enough data
        if len(df) < 5:
            st.error("❌ Insufficient data for forecasting. Need at least 5 records.")
        else:
            st.subheader("📊 Generate Simple Forecast")
            
            col1, col2 = st.columns(2)
            with col1:
                selected_product = st.selectbox("🏷️ Select Product:", df['Product'].unique())
            with col2:
                forecast_days = st.slider("📅 Forecast Period (days):", 7, 60, 30)
            
            if st.button("🔮 Generate Basic Forecast", type="primary"):
                try:
                    # Generate simple forecast
                    forecast_df = simple_forecast(df, selected_product, forecast_days)
                    
                    if forecast_df is not None:
                        st.success("✅ Forecast generated successfully!")
                        
                        # Display forecast summary
                        total_forecast = forecast_df['Predicted_Sales'].sum()
                        avg_daily_forecast = forecast_df['Predicted_Sales'].mean()
                        max_daily_forecast = forecast_df['Predicted_Sales'].max()
                        min_daily_forecast = forecast_df['Predicted_Sales'].min()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Forecast", f"{total_forecast:.0f}")
                        with col2:
                            st.metric("Daily Average", f"{avg_daily_forecast:.1f}")
                        with col3:
                            st.metric("Peak Day", f"{max_daily_forecast:.0f}")
                        with col4:
                            st.metric("Low Day", f"{min_daily_forecast:.0f}")
                        
                        # Simple line chart
                        st.subheader("📈 Forecast Visualization")
                        chart_data = forecast_df.set_index('Date')['Predicted_Sales']
                        st.line_chart(chart_data)
                        
                        # Business recommendations
                        st.markdown("---")
                        st.subheader("💡 Simple Recommendations")
                        
                        product_data = df[df['Product'] == selected_product].iloc[-1]
                        current_stock = product_data['Stock']
                        recommended_stock = total_forecast * 1.2  # 20% buffer
                        
                        if current_stock < recommended_stock:
                            st.markdown(f"""
                            <div class="warning-box">
                                <h4>⚠️ Stock Alert</h4>
                                <p><strong>Current Stock:</strong> {current_stock:.0f} units</p>
                                <p><strong>Forecasted Demand:</strong> {total_forecast:.0f} units</p>
                                <p><strong>Recommended Stock:</strong> {recommended_stock:.0f} units</p>
                                <p><strong>Additional Stock Needed:</strong> {recommended_stock - current_stock:.0f} units</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="highlight-box">
                                <h4>✅ Stock Status: Good</h4>
                                <p><strong>Current Stock:</strong> {current_stock:.0f} units</p>
                                <p><strong>Forecasted Demand:</strong> {total_forecast:.0f} units</p>
                                <p><strong>Buffer Available:</strong> {current_stock - total_forecast:.0f} units</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detailed forecast table
                        with st.expander("📋 Detailed Forecast Table", expanded=False):
                            display_df = forecast_df.copy()
                            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                            display_df['Predicted_Sales'] = display_df['Predicted_Sales'].round(1)
                            display_df.columns = ['Date', 'Predicted Sales']
                            
                            st.dataframe(display_df, use_container_width=True)
                            
                            # Download option
                            csv = display_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Forecast CSV",
                                data=csv,
                                file_name=f'simple_forecast_{selected_product}_{datetime.now().strftime("%Y%m%d")}.csv',
                                mime='text/csv'
                            )
                    
                    else:
                        st.error("❌ Could not generate forecast for selected product.")
                
                except Exception as e:
                    st.error(f"❌ Error generating forecast: {str(e)}")
            
            # Method explanation
            with st.expander("🔍 How This Forecast Works", expanded=False):
                st.write("""
                **This basic forecast uses simple statistical methods:**
                
                1. **Average Sales:** Calculates the historical average sales for the product
                2. **Trend Analysis:** Detects if sales are increasing or decreasing over time
                3. **Seasonality:** Applies basic weekend/weekday adjustments
                4. **Safety Buffer:** Recommends 20% extra stock as safety margin
                
                **Limitations:**
                - No complex seasonal patterns
                - No external factors (holidays, promotions, etc.)
                - Linear trend assumptions
                
                **For advanced forecasting with Machine Learning, install the full requirements:**
                - plotly, scikit-learn, seaborn, matplotlib
                """)
    
    else:
        st.warning("📁 Please upload and clean your data in the HOME page first.")

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
st.sidebar.markdown("**📦 Ahva Dashboard v1.5**")
st.sidebar.markdown("*Basic Analytics Platform*")
st.sidebar.markdown("Built with ❤️ using Streamlit")

# Instructions for full version
st.sidebar.markdown("---")
st.sidebar.info("""
**🚀 Want Advanced Features?**

Add this to requirements.txt:
```
plotly>=5.14.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
```

Then you'll get:
- Interactive charts
- Machine Learning forecasts
- Advanced visualizations
- Heatmaps & trends
""")

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
