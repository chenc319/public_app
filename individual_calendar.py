### ---------------------------------------------------------------------------------------- ###
### -------------------------------- PACKAGES AND FUNCTIONS -------------------------------- ###
### ---------------------------------------------------------------------------------------- ###

### IMPORT PACKAGES ###
import pandas as pd
import yfinance as yf
import functools as ft
import requests
from datetime import datetime
import matplotlib.pyplot as plt
from datetime import timedelta

def individual_calendar(ticker):
    ### FUNCTIONS ###
    def merge_dfs(array_of_dfs):
        new_df = ft.reduce(lambda left, right: pd.merge(left,
                                                        right,
                                                        left_index=True,
                                                        right_index=True,
                                                        how='outer'), array_of_dfs)
        return (new_df)


    def get_future_earnings(company):
        # API setup
        api_key = "32b3f2feec9f4a36a97dbe3066f60c71"
        url = f'https://api.twelvedata.com/earnings?symbol={company}&apikey={api_key}'
        try:
            # Fetch data
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            # Filter future dates
            future_dates = [
                entry['date']
                for entry in data['earnings']
                if datetime.strptime(entry['date'], '%Y-%m-%d') > datetime.now()
            ]
            return future_dates
        except Exception as e:
            return f"Error: {str(e)}"


    def get_fiscal_year_end(company):
        api_key = "32b3f2feec9f4a36a97dbe3066f60c71"
        url = f"https://api.twelvedata.com/statistics?symbol={company}&apikey={api_key}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                fiscal_year_end = data['statistics']['financials']['fiscal_year_ends']
                return fiscal_year_end
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"


    def is_too_close(date, previous_dates, threshold_days=30):
        return any(abs(date - prev_date).days < threshold_days for prev_date in previous_dates)

    ### GET FUTURE EARNINGS, FISCAL YEAR END, FILER TYPE ###
    future_earnings = get_future_earnings(ticker)
    fiscal_year_end = (datetime.strptime(get_fiscal_year_end(ticker), "%Y-%m-%d").date() + timedelta(366)).strftime(
        "%m-%d-%Y")
    ticker_obj = yf.Ticker(ticker)
    floatshares = ticker_obj.get_shares_full()[-1]
    current_price = yf.Ticker(ticker).fast_info['last_price']
    if floatshares and current_price:
        floatvalue = floatshares * current_price
    if floatvalue > 700000000:
        filer_type = 1
    elif 700000000 >= floatvalue >= 75000000:
        filer_type = 2
    else:
        filer_type = 3

    ### ---------------------------------------------------------------------------------------- ###
    ### --------------------------------------- ANALYSIS --------------------------------------- ###
    ### ---------------------------------------------------------------------------------------- ###

    date_obj = datetime.strptime(fiscal_year_end, "%m-%d-%Y").date()
    q1_end = (date_obj + timedelta(days=90)).strftime("%m-%d-%Y")
    q2_end = (date_obj + timedelta(days=181)).strftime("%m-%d-%Y")
    q3_end = (date_obj + timedelta(days=273)).strftime("%m-%d-%Y")
    q4_end = (date_obj + timedelta(days=365)).strftime("%m-%d-%Y")
    company_quarter_ends = [q1_end, q2_end, q3_end, q4_end]

    ### ----------------------------------- 10-K DEADLINES ----------------------------------- ###
    date_obj = datetime.strptime(fiscal_year_end, "%m-%d-%Y").date()
    if filer_type == 1:
        k10 = (date_obj + timedelta(days=60)).strftime("%m-%d-%Y")
    elif filer_type == 2:
        k10 = (date_obj + timedelta(days=75)).strftime("%m-%d-%Y")
    elif filer_type == 3:
        k10 = (date_obj + timedelta(days=90)).strftime("%m-%d-%Y")

    ### ----------------------------------- 10-Q DEADLINES ----------------------------------- ###
    if filer_type == 3:
        day_value = 45
    else:
        day_value = 40
    q1 = (datetime.strptime(company_quarter_ends[0], "%m-%d-%Y").date() + timedelta(days=day_value)).strftime("%m-%d-%Y")
    q2 = (datetime.strptime(company_quarter_ends[1], "%m-%d-%Y").date() + timedelta(days=day_value)).strftime("%m-%d-%Y")
    q3 = (datetime.strptime(company_quarter_ends[2], "%m-%d-%Y").date() + timedelta(days=day_value)).strftime("%m-%d-%Y")
    q4 = (datetime.strptime(company_quarter_ends[3], "%m-%d-%Y").date() + timedelta(days=day_value)).strftime("%m-%d-%Y")
    q_10_list = [q1, q2, q3, q4]

    ### ------------------------------- PROXY STATEMENT DEF 14A ------------------------------- ###
    date_obj = datetime.strptime(fiscal_year_end, "%m-%d-%Y").date()
    new_date = date_obj + timedelta(days=121)
    proxy = new_date.strftime("%m-%d-%Y")

    ### --------------------------- FORM 20F FOREIGN PRIVATE ISSUERS --------------------------- ###
    date_obj = datetime.strptime(fiscal_year_end, "%m-%d-%Y").date()
    new_date = date_obj + timedelta(days=123)
    f20 = new_date.strftime("%m-%d-%Y")

    ### --------------------------- FORM 11K EMPLOYEE BENEFITS PLAN ---------------------------- ###
    date_obj = datetime.strptime(fiscal_year_end, "%m-%d-%Y").date()
    new_date = date_obj + timedelta(days=181)
    k11 = new_date.strftime("%m-%d-%Y")

    ### --------------------------------------- FORM 13F --------------------------------------- ###
    q1 = (datetime.strptime(company_quarter_ends[0], "%m-%d-%Y").date() + timedelta(days=46)).strftime("%m-%d-%Y")
    q2 = (datetime.strptime(company_quarter_ends[1], "%m-%d-%Y").date() + timedelta(days=46)).strftime("%m-%d-%Y")
    q3 = (datetime.strptime(company_quarter_ends[2], "%m-%d-%Y").date() + timedelta(days=46)).strftime("%m-%d-%Y")
    q4 = (datetime.strptime(company_quarter_ends[3], "%m-%d-%Y").date() + timedelta(days=46)).strftime("%m-%d-%Y")
    f_13_list = [q1, q2, q3, q4]

    ### ---------------------------------------- FORM 5 --------------------------------------- ###
    date_obj = datetime.strptime(fiscal_year_end, "%m-%d-%Y").date()
    new_date = date_obj + timedelta(days=46)
    f5 = new_date.strftime("%m-%d-%Y")

    ### ---------------------------------------------------------------------------------------- ###
    ### --------------------------------------- CALENDAR --------------------------------------- ###
    ### ---------------------------------------------------------------------------------------- ###

    schedule = pd.DataFrame(columns=['deadlines', 'dates'])
    schedule.loc['10K'] = ['10K', k10]
    schedule.loc['10Q Q1'] = ['10Q Q1', q_10_list[0]]
    schedule.loc['10Q Q2'] = ['10Q Q2', q_10_list[1]]
    schedule.loc['10Q Q3'] = ['10Q Q3', q_10_list[2]]
    schedule.loc['10Q Q4'] = ['10Q Q4', q_10_list[3]]
    schedule.loc['Proxy (14A)'] = ['Proxy (14A)', proxy]
    schedule.loc['20F'] = ['20F', f20]
    schedule.loc['11K'] = ['11K', k11]
    schedule.loc['13F Q1'] = ['13F Q1', f_13_list[0]]
    schedule.loc['13F Q2'] = ['13F Q2', f_13_list[1]]
    schedule.loc['13F Q3'] = ['13F Q3', f_13_list[2]]
    schedule.loc['13F Q4'] = ['13F Q4', f_13_list[3]]
    schedule.loc['5F'] = ['5F', f5]

    # Create the dataframe with your data
    df = schedule.copy()
    df['dates'] = pd.to_datetime(df['dates'])
    df = df.sort_values('dates')

    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Filter dataframe for each subplot
    df_10k_10q = df[df['deadlines'].str.contains('10')]
    df_13f = df[df['deadlines'].str.contains('13F')]
    df_others = df[~df['deadlines'].str.contains('10|13F')]

    # First subplot (10-K and 10-Q)
    ax1.scatter(df_10k_10q['dates'], [1] * len(df_10k_10q), s=100, color='blue', zorder=3)
    processed_dates_1 = []
    for deadline, date in zip(df_10k_10q['deadlines'], df_10k_10q['dates']):
        label = f"{deadline}\n{date.strftime('%Y-%m-%d')}"

        if is_too_close(date, processed_dates_1):
            y_pos = 1
            xytext = (0, -25)
        else:
            y_pos = 1
            xytext = (0, 10)

        ax1.annotate(label, (date, y_pos),
                     xytext=xytext,
                     textcoords='offset points',
                     ha='center',
                     bbox=dict(boxstyle='round,pad=0.3',
                               fc='white',
                               ec='gray',
                               alpha=0.8),
                     fontsize=9)
        processed_dates_1.append(date)

    # Second subplot (13F filings)
    ax2.scatter(df_13f['dates'], [1] * len(df_13f), s=100, color='blue', zorder=3)
    processed_dates_2 = []
    for deadline, date in zip(df_13f['deadlines'], df_13f['dates']):
        label = f"{deadline}\n{date.strftime('%Y-%m-%d')}"

        if is_too_close(date, processed_dates_2):
            y_pos = 1
            xytext = (0, -25)
        else:
            y_pos = 1
            xytext = (0, 10)

        ax2.annotate(label, (date, y_pos),
                     xytext=xytext,
                     textcoords='offset points',
                     ha='center',
                     bbox=dict(boxstyle='round,pad=0.3',
                               fc='white',
                               ec='gray',
                               alpha=0.8),
                     fontsize=9)
        processed_dates_2.append(date)

    # Third subplot (other filings)
    ax3.scatter(df_others['dates'], [1] * len(df_others), s=100, color='blue', zorder=3)
    processed_dates_3 = []
    for deadline, date in zip(df_others['deadlines'], df_others['dates']):
        label = f"{deadline}\n{date.strftime('%Y-%m-%d')}"

        if is_too_close(date, processed_dates_3):
            y_pos = 1
            xytext = (0, -25)
        else:
            y_pos = 1
            xytext = (0, 10)

        ax3.annotate(label, (date, y_pos),
                     xytext=xytext,
                     textcoords='offset points',
                     ha='center',
                     bbox=dict(boxstyle='round,pad=0.3',
                               fc='white',
                               ec='gray',
                               alpha=0.8),
                     fontsize=9)
        processed_dates_3.append(date)

    # Get the overall date range
    all_dates = df['dates']
    date_min = all_dates.min()
    date_max = all_dates.max()

    # Customize all subplots
    for ax in [ax1, ax2, ax3]:
        ax.set_yticks([])
        ax.grid(True, axis='x', linestyle='--', alpha=0.7, zorder=0)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
        ax.set_facecolor('white')
        ax.set_xlim(date_min - pd.Timedelta(days=30), date_max + pd.Timedelta(days=30))
        plt.setp(ax.get_xticklabels(), rotation=45)

    # Set titles with increased spacing
    ax1.set_title('10-K and 10-Q Filing Deadlines', pad=20, fontsize=12, fontweight='bold')
    ax2.set_title('13F Filing Deadlines', pad=20, fontsize=12, fontweight='bold')
    ax3.set_title('Other Filing Deadlines', pad=20, fontsize=12, fontweight='bold')

    # Adjust main title with increased spacing
    fig.suptitle(ticker + ' Filing Deadlines Timeline', fontsize=14, fontweight='bold', y=0.98)

    # Adjust layout with more space at the top
    plt.subplots_adjust(top=0.92)
    plt.margins(x=0.1)
    plt.tight_layout()
    return fig