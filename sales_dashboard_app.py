### ---------------------------------------------------------------------------------------- ###
### -------------------------------- PACKAGES AND FUNCTIONS -------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

import streamlit as st
import numpy as np
from individual_calendar import *
import refinitiv.data as rd
import seaborn as sns
rd.open_session('desktop.workspace')

def get_refinitiv_historical_data(tickers,flds,end_date_str):
    '''
    :param tickers: list
    :return df: pandas dataframe
    '''
    df = rd.get_history(
        universe=tickers,
        fields=flds,
        interval='1d',
        start='1999-12-31',
        end=end_date_str
    )
    df.index = df.index.values
    df.index = pd.to_datetime(df.index)
    df.columns = tickers
    return(df)

# Set page configuration and styling
st.set_page_config(
    page_title="Candor Research Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTab {
        font-size: 18px;
    }
    .stPlot {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Organize the app into tabs
(tab1, tab2, tab3, tab4,
 tab5, tab6, tab7, tab8,
 tab9, tab10, tab11, tab12,
 tab13, tab14
 ) = st.tabs([
    "Basic Info",
    "SBC Trend",
    "Stock Ownership",
    "Historical Volatility Analysis",
    "Historical Return Analysis",
    'Historical Volume Analysis',
    'Filing Deadlines',
    'Respective Peers Performances',
    'Profitability Metrics',
    'Liquidity Metrics',
    'Efficiency Metrics',
    'Leverage and Solvency Metrics',
    'Sell-to-Cover Model',
    'Probability Price Levels'
])


### ---------------------------------------------------------------------------------------- ###
### -------------------------------------- BASIC INFO -------------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

# Tab 1: Basic Information
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Enter Ticker Symbol", "AAPL")
        ticker_object = yf.Ticker(ticker)
        number_of_employees = ticker_object.info.get('fullTimeEmployees')

        # Get market cap and convert to billions
        market_cap = ticker_object.info.get('marketCap', 0) / 1e9
        st.metric("Number of Employees", str(number_of_employees))
        st.metric("Market Cap", f"${market_cap:.2f}B")

    with col2:
        # Get broker recommendations
        recommendations = ticker_object.recommendations
        recommendations.columns = ['Period','Strong Buy','Buy','Hold','Sell','Strong Sell']
        recommendations['Period'] = ['Currently','1 Month Ago','2 Months Ago','3 Months Ago']
        if recommendations is not None:
            current_brokers = recommendations.tail()
            st.subheader("Broker Recommendations")
            st.dataframe(current_brokers)

### ---------------------------------------------------------------------------------------- ###
### ------------------------------------- SBC PER YEAR ------------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

with tab2:
    st.subheader("Stock-Based Compensation Trends")
    ticker_object = yf.Ticker(ticker)
    financials = ticker_object.financials
    for suffix in ['.O','.K','a','']:
        try:
            sbc = get_refinitiv_historical_data([ticker+suffix],
                                                'TR.StockBasedCompActual',
                                                '2025-1-31')
            mktcap = get_refinitiv_historical_data([ticker+suffix],
                                                'TR.CompanyMarketCapitalization',
                                                '2025-1-31')
            rev = get_refinitiv_historical_data([ticker+suffix],
                                                'TR.F.TotRevenue',
                                                '2025-1-31')
            fcf = get_refinitiv_historical_data([ticker+suffix],
                                                'TR.FCFActValue',
                                                '2025-1-31')

            break
        except:
            continue
    sbc.columns = [ticker]
    mktcap.columns = [ticker]
    rev.columns = [ticker]
    fcf.columns = [ticker]
    yearly_sbc = pd.DataFrame({
        'Date': sbc.index.values,
        'SBC Expense ($)': sbc[ticker]
    })
    yearly_mtkcap = pd.DataFrame({
        'Date': mktcap.index.values,
        'Market Cap ($)': mktcap[ticker]
    })
    yearly_rev = pd.DataFrame({
        'Date': rev.index.values,
        'Revenue ($)': rev[ticker]
    })
    yearly_fcf = pd.DataFrame({
        'Date': fcf.index.values,
        'FCF ($)': fcf[ticker]
    })

    st.dataframe(yearly_sbc)
    st.line_chart(yearly_sbc.set_index('Date'), use_container_width=True)

### ---------------------------------------------------------------------------------------- ###
### --------------------------------- STOCK OWNERSHIP INFO --------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

with tab3:
    st.subheader("Stock Ownership Data")

    col1, col2 = st.columns(2)
    ticker_object = yf.Ticker(ticker)

    with col1:
        # Display Major Holders
        major_holders = ticker_object.major_holders
        major_holders.index = ['Insiders','Institutional','Institutional Float','# of Institutions']
        major_holders.loc['Insiders'] = str(round(major_holders.loc['Insiders'].values[0] * 100,2)) + '%'
        major_holders.loc['Institutional'] = str(round(major_holders.loc['Institutional'].values[0] * 100,2)) + '%'
        major_holders.loc['Institutional Float'] = str(round(major_holders.loc['Institutional Float'].values[0] * 100,2)) + '%'
        st.subheader("Major Holders")
        st.dataframe(major_holders)

    with col2:
        # Display Institutional Holders
        institutional_holders = ticker_object.institutional_holders
        institutional_holders['Date Reported'] = [x.date() for x in institutional_holders['Date Reported']]
        st.subheader("Top Institutional Holders")
        st.dataframe(institutional_holders)

### ---------------------------------------------------------------------------------------- ###
### ---------------------------- HISTORICAL VOLATILITY ANALYSIS ---------------------------- ###
### ---------------------------------------------------------------------------------------- ###

with tab4:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Historical Volatility")

        # Add date input fields with unique keys
        start_date = st.date_input("Start Date", value=pd.to_datetime('2023-01-01'), key="vol_start_date")
        end_date = st.date_input("End Date", value=pd.to_datetime('today'), key="vol_end_date")

        # Fetch stock data for selected date range
        ticker_object = yf.Ticker(ticker)
        data = ticker_object.history(start=start_date, end=end_date)

        # Calculate daily returns
        data['Daily_Return'] = np.log(data['Close'] / data['Close'].shift(1))

        # Calculate historical volatility (20-day rolling standard deviation)
        data['Volatility'] = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252)

        # Calculate moving averages
        data['30-Day MA'] = data['Volatility'].rolling(window=30).mean()
        data['90-Day MA'] = data['Volatility'].rolling(window=90).mean()

        # Create figure with multiple lines
        fig = plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Volatility'],
                 label='Historical Volatility', marker='')
        plt.plot(data.index, data['30-Day MA'],
                 label='30-Day Moving Average', linestyle='--')
        plt.plot(data.index, data['90-Day MA'],
                 label='90-Day Moving Average', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.title(f'{ticker} Historical Volatility with Moving Averages')
        plt.legend()
        plt.grid(True)

        st.pyplot(fig)

    with col2:
        # Display current metrics using the latest available values
        st.metric("30-Day Volatility", f"{data['30-Day MA'].iloc[-1] * 100:.2f}%")
        st.metric("90-Day Volatility", f"{data['90-Day MA'].iloc[-1] * 100:.2f}%")

### ---------------------------------------------------------------------------------------- ###
### ----------------------------- HISTORICAL RETURN PERFORMANCE ---------------------------- ###
### ---------------------------------------------------------------------------------------- ###

with tab5:
    st.subheader("Stock Price Performance")

    # Add date input fields
    col_dates = st.columns(2)
    with col_dates[0]:
        start_date = st.date_input("Start Date", value=pd.to_datetime('2023-01-01'), key="perf_start_date")
    with col_dates[1]:
        end_date = st.date_input("End Date", value=pd.to_datetime('today'), key="perf_end_date")

    # Fetch stock data for selected date range
    ticker_object = yf.Ticker(ticker)
    data = ticker_object.history(start=start_date, end=end_date)

    # Calculate returns
    data['Daily Returns (%)'] = ((data['Close'] - data['Close'].shift(1)) /
                                 data['Close'].shift(1) * 100)
    data['Cumulative Performance (%)'] = (((1 + data['Daily Returns (%)'] /
                                            100).cumprod() - 1) * 100)

    # Calculate monthly and annual performance
    monthly_performance = data['Daily Returns (%)'].resample('M').sum()
    annual_performance = data['Daily Returns (%)'].resample('Y').sum()

    # Create three columns for charts
    col1, col2, col3 = st.columns(3)

    data.index = [x.date() for x in data.index]
    monthly_performance.index = [x.date() for x in monthly_performance.index]
    annual_performance.index = [x.date() for x in annual_performance.index]

    with col1:
        # Cumulative Performance Line Chart
        fig1 = plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Cumulative Performance (%)'])
        plt.title(f'{ticker} Cumulative Performance')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.grid(True)
        st.pyplot(fig1)

    with col2:
        # Monthly Performance Bar Chart
        fig2 = plt.figure(figsize=(10, 6))
        monthly_performance.tail(12).plot(kind='bar')
        plt.title('Monthly Performance')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(fig2)

    with col3:
        # Annual Performance Bar Chart
        fig3 = plt.figure(figsize=(10, 6))
        annual_performance.tail(5).plot(kind='bar')
        plt.title('Annual Performance')
        plt.xlabel('Year')
        plt.ylabel('Return (%)')
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(fig3)

### ---------------------------------------------------------------------------------------- ###
### ------------------------------- HISTORICAL VOLUME AND ADV ------------------------------ ###
### ---------------------------------------------------------------------------------------- ###

with tab6:
    st.subheader("Trading Volume Analysis")

    # Add date input fields
    col_dates = st.columns(2)
    with col_dates[0]:
        start_date = st.date_input("Start Date", value=pd.to_datetime('2023-01-01'), key="volume_start_date")
    with col_dates[1]:
        end_date = st.date_input("End Date", value=pd.to_datetime('today'), key="volume_end_date")

    # Get historical volume data
    historical_data = ticker_object.history(start=start_date, end=end_date)
    volume_data = historical_data[['Volume']]

    # Calculate ADV and rolling ADV
    adv = volume_data['Volume'].mean()
    rolling_adv = volume_data['Volume'].rolling(window=30).mean()
    volume_data['5% ADV'] = adv * 0.05
    volume_data['10% ADV'] = adv * 0.10
    volume_data['20% ADV'] = adv * 0.20

    # Create smaller figure with adjusted size
    fig = plt.figure(figsize=(8, 4))  # Reduced from (12, 6) to (8, 4)
    plt.plot(volume_data.index, volume_data['Volume'],
             label='Daily Volume', color='blue')
    plt.plot(volume_data.index, rolling_adv,
             label='30-Day Rolling ADV', color='orange')

    plt.title(f'{ticker} Trading Volume with ADV Limits')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)

    # Correct formatter implementation
    from matplotlib.ticker import FuncFormatter

    def share_formatter(x, p):
        return f'{int(x):,} shares'

    plt.gca().yaxis.set_major_formatter(FuncFormatter(share_formatter))

    # Use columns to constrain the width
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.pyplot(fig, use_container_width=False)  # Set use_container_width to False

    # Display ADV metrics in full number format
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("5% ADV", f"{int(adv * 0.05):,} shares")
    with col2:
        st.metric("10% ADV", f"{int(adv * 0.10):,} shares")
    with col3:
        st.metric("20% ADV", f"{int(adv * 0.20):,} shares")


### ---------------------------------------------------------------------------------------- ###
### ----------------------------------- FILING DEADLINES ----------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

with tab7:
    st.subheader("Deadlines")
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    stock_performance = np.random.uniform(-5, 5, len(dates))
    data = pd.DataFrame({
        'Date': dates,
        'Stock Performance (%)': stock_performance
    })
    st.pyplot(individual_calendar(ticker))

### ---------------------------------------------------------------------------------------- ###
### ------------------------------- RESPECTIVE PEERS ANALYSIS ------------------------------ ###
### ---------------------------------------------------------------------------------------- ###

with tab8:
    st.subheader("Peer Analysis")

    # Input for peer tickers
    peer_input = st.text_input("Enter peer tickers (comma-separated)", "POST,K,GIS")
    peer_tickers = [ticker.strip() for ticker in peer_input.split(',')]

    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime('2023-01-01'), key="peer_start_date")
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime('today'), key="peer_end_date")

    # Fetch and process data
    if st.button("Analyze Peers"):
        try:
            # Create empty DataFrames
            prices_df = pd.DataFrame()
            metrics_df = pd.DataFrame()

            # Fetch S&P 500 data first
            spx = yf.Ticker("^GSPC")
            spx_hist = spx.history(start=start_date, end=end_date)
            spx_returns = spx_hist['Close'].pct_change()

            # Fetch data for each ticker
            for tick in peer_tickers:
                stock = yf.Ticker(tick)
                hist = stock.history(start=start_date, end=end_date)

                # Calculate daily returns
                daily_returns = hist['Close'].pct_change()

                # Calculate metrics
                annualized_return = (daily_returns.mean() * 252) * 100
                annualized_vol = (daily_returns.std() * np.sqrt(252)) * 100
                sharpe_ratio = annualized_return / annualized_vol
                max_drawdown = ((hist['Close'] / hist['Close'].cummax() - 1).min()) * 100

                # Calculate beta against S&P 500
                covariance = daily_returns.cov(spx_returns)
                market_variance = spx_returns.var()
                beta = covariance / market_variance

                # Store metrics
                metrics_df = pd.concat([metrics_df, pd.DataFrame({
                    'Ticker': [tick],
                    'Annualized Return (%)': [round(annualized_return, 2)],
                    'Annualized Volatility (%)': [round(annualized_vol, 2)],
                    'Sharpe Ratio': [round(sharpe_ratio, 2)],
                    'Maximum Drawdown (%)': [round(max_drawdown, 2)],
                    'Beta': [round(beta, 2)]
                })])

                # Store prices for performance chart
                prices_df[tick] = hist['Close'] / hist['Close'].iloc[0] * 100

            # Display metrics table
            st.subheader("Peer Comparison Metrics")
            st.dataframe(metrics_df.set_index('Ticker'))

            # Display performance chart
            st.subheader("Relative Performance")
            fig = plt.figure(figsize=(12, 6))
            for column in prices_df.columns:
                plt.plot(prices_df.index, prices_df[column], label=column)

            plt.title("Relative Performance (Base 100)")
            plt.xlabel("Date")
            plt.ylabel("Relative Performance")
            plt.legend()
            plt.grid(True)
            st.pyplot(fig)

            # Correlation matrix
            st.subheader("Correlation Matrix")
            returns_df = prices_df.pct_change()
            correlation_matrix = returns_df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error occurred: {str(e)}")

### ---------------------------------------------------------------------------------------- ###
### --------------------------------- PROFITABILITY METRICS -------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

with tab9:
    st.subheader("Profitability Metrics")

    # Download historical financial data for Apple
    data = yf.Ticker(ticker)

    # Calculate metrics
    gross_profit_margin = (data.financials.loc['Gross Profit'] /
                           data.financials.loc['Total Revenue']).to_frame(name="Gross Profit Margin").dropna()

    net_profit_margin = (data.financials.loc['Net Income'] /
                         data.financials.loc['Total Revenue']).to_frame(name="Net Profit Margin").dropna()

    roe = (data.financials.loc['Net Income'] /
           data.balance_sheet.loc['Stockholders Equity']).to_frame(name="Return on Equity").dropna()

    # Combine metrics
    metrics = pd.concat([
        gross_profit_margin,
        net_profit_margin,
        roe
    ], axis=1)

    # Display DataFrame
    st.dataframe(metrics)

    # Create and display plots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    # Gross Profit Margin Plot
    gross_profit_margin.plot(ax=ax1, marker='o')
    ax1.set_title('Gross Profit Margin')
    ax1.grid(True)

    # Net Profit Margin Plot
    net_profit_margin.plot(ax=ax2, marker='o', color='green')
    ax2.set_title('Net Profit Margin')
    ax2.grid(True)

    # ROE Plot
    roe.plot(ax=ax3, marker='o', color='red')
    ax3.set_title('Return on Equity')
    ax3.grid(True)

    plt.tight_layout()
    st.pyplot(fig)


### ---------------------------------------------------------------------------------------- ###
### ---------------------------------- LIQUIDITY METRICS ----------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

with tab10:
    st.subheader("Liquidity Metrics")
    data = yf.Ticker(ticker)

    try:
        # Try different possible balance sheet labels
        current_assets = data.balance_sheet.loc['Current Assets']
        current_liabilities = data.balance_sheet.loc['Current Liabilities']

        # Calculate current ratio
        current_ratio = (current_assets / current_liabilities).to_frame(name="Current Ratio").dropna()

        # Calculate working capital
        working_capital = (current_assets - current_liabilities).to_frame(name="Working Capital").dropna()

        # Try to calculate quick ratio if inventory data exists
        try:
            inventory_labels = ['Inventory', 'Total Inventory', 'Net Inventory']
            inventory_data = None
            for label in inventory_labels:
                try:
                    inventory_data = data.balance_sheet.loc[label]
                    break
                except KeyError:
                    continue

            if inventory_data is not None:
                quick_ratio = ((current_assets - inventory_data) /
                               current_liabilities).to_frame(name="Quick Ratio").dropna()
            else:
                quick_ratio = current_ratio.copy()
                quick_ratio.columns = ["Quick Ratio"]
                st.warning("Inventory data not available - Quick Ratio equals Current Ratio")

        except Exception as e:
            quick_ratio = current_ratio.copy()
            quick_ratio.columns = ["Quick Ratio"]
            st.warning("Could not calculate Quick Ratio - using Current Ratio instead")

        # Combine and display metrics
        liquidity_metrics = pd.concat([current_ratio, quick_ratio, working_capital], axis=1)
        st.dataframe(liquidity_metrics)

        # Create plots side by side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        current_ratio.plot(ax=ax1, marker='o')
        ax1.set_title('Current Ratio')
        ax1.grid(True)

        quick_ratio.plot(ax=ax2, marker='o', color='green')
        ax2.set_title('Quick Ratio')
        ax2.grid(True)

        working_capital.plot(ax=ax3, marker='o', color='red')
        ax3.set_title('Working Capital')
        ax3.grid(True)

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error calculating liquidity metrics: {str(e)}")



### ---------------------------------------------------------------------------------------- ###
### ---------------------------------- EFFICIENCY METRICS ---------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

with tab11:
    st.subheader("Efficiency Metrics")
    data = yf.Ticker(ticker)

    try:
        metrics_available = False

        # Try to calculate inventory turnover
        try:
            inventory_labels = ['Inventory', 'Total Inventory', 'Net Inventory']
            inventory_data = None
            for label in inventory_labels:
                try:
                    inventory_data = data.balance_sheet.loc[label]
                    break
                except KeyError:
                    continue

            if inventory_data is not None:
                inventory_turnover = (data.financials.loc['Cost Of Revenue'] /
                                      inventory_data.rolling(window=2).mean()).to_frame(
                    name="Inventory Turnover").dropna()
                metrics_available = True
            else:
                st.warning("Inventory data not available")
                inventory_turnover = pd.DataFrame()
        except Exception as e:
            st.warning(f"Could not calculate Inventory Turnover: {str(e)}")
            inventory_turnover = pd.DataFrame()

        # Try to calculate AR turnover
        try:
            receivables_labels = ['Net Receivables', 'Accounts Receivable']
            receivables_data = None
            for label in receivables_labels:
                try:
                    receivables_data = data.balance_sheet.loc[label]
                    break
                except KeyError:
                    continue

            if receivables_data is not None:
                accounts_receivable_turnover = (data.financials.loc['Total Revenue'] /
                                                receivables_data.rolling(window=2).mean()).to_frame(
                    name="AR Turnover").dropna()
                metrics_available = True
            else:
                st.warning("Accounts Receivable data not available")
                accounts_receivable_turnover = pd.DataFrame()
        except Exception as e:
            st.warning(f"Could not calculate AR Turnover: {str(e)}")
            accounts_receivable_turnover = pd.DataFrame()

        if metrics_available:
            efficiency_metrics = pd.concat([inventory_turnover, accounts_receivable_turnover], axis=1)
            st.dataframe(efficiency_metrics)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            if not inventory_turnover.empty:
                inventory_turnover.plot(ax=ax1, marker='o')
                ax1.set_title('Inventory Turnover Over Time')
                ax1.grid(True)

            if not accounts_receivable_turnover.empty:
                accounts_receivable_turnover.plot(ax=ax2, marker='o', color='green')
                ax2.set_title('Accounts Receivable Turnover Over Time')
                ax2.grid(True)

            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error calculating efficiency metrics: {str(e)}")

### ---------------------------------------------------------------------------------------- ###
### ---------------------------- LEVERAGE AND SOLVENCY METRICS ----------------------------- ###
### ---------------------------------------------------------------------------------------- ###

with tab12:
    st.subheader("Leverage and Solvency Metrics")
    data = yf.Ticker(ticker)

    try:
        # Calculate debt to equity
        try:
            total_liabilities = data.balance_sheet.loc['Current Liabilities']
            stockholders_equity = data.balance_sheet.loc['Stockholders Equity']
            debt_to_equity = (total_liabilities / stockholders_equity).to_frame(name="Debt-to-Equity Ratio").dropna()
        except Exception as e:
            st.warning(f"Could not calculate Debt-to-Equity ratio: {str(e)}")
            debt_to_equity = pd.DataFrame()

        # Calculate EBITDA
        try:
            operating_income = data.financials.loc['Operating Income']
            depreciation = data.cashflow.loc['Depreciation And Amortization']
            ebitda = (operating_income + depreciation).to_frame(name="EBITDA").dropna()
        except Exception as e:
            st.warning(f"Could not calculate EBITDA: {str(e)}")
            ebitda = pd.DataFrame()

        # Display metrics if available
        if not debt_to_equity.empty or not ebitda.empty:
            solvency_metrics = pd.concat([debt_to_equity, ebitda], axis=1)
            st.dataframe(solvency_metrics)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            if not debt_to_equity.empty:
                debt_to_equity.plot(ax=ax1, marker='o')
                ax1.set_title('Debt-to-Equity Ratio Over Time')
                ax1.grid(True)

            if not ebitda.empty:
                ebitda.plot(ax=ax2, marker='o', color='green')
                ax2.set_title('EBITDA Over Time')
                ax2.grid(True)

            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error calculating leverage and solvency metrics: {str(e)}")

### ---------------------------------------------------------------------------------------- ###
### ------------------------------ SELL TO COVER MODEL ANALYSIS ---------------------------- ###
### ---------------------------------------------------------------------------------------- ###










### ---------------------------------------------------------------------------------------- ###
### ---------------------------- PROBABILITY PRICE LEVEL ANALYSIS -------------------------- ###
### ---------------------------------------------------------------------------------------- ###







