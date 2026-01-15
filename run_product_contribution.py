import streamlit.components as stc

# Utils
import base64 
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
import pandas as pd 
import streamlit as st
import numpy as np
import pickle
import pandas as pd
from io import BytesIO

import plotly.tools as tls
import plotly.graph_objs as go
from dateutil import parser
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import plotly.io as pio
from analysis import move_file
pio.renderers.default = 'browser'
from datetime import timedelta, datetime
import pandas as pd
from plotly.subplots import make_subplots
from analysis import  move_file
from tqdm import tqdm
from general_layout import generate_download_button
import warnings
warnings.filterwarnings('ignore')

# from pyxlsb import open_workbook as open_xlsb
# pip install pyxlsb
# pip install xlsxwriter

# 這邊就用80/20
# 問題：如何將某檔案上傳，取95%利潤；product取會員？



#SECTION - 附屬function
def load_and_prepare_data(data_path:str):
    
    """該函數將處理資料載入和初始預處理。

    Parameters:
        data_path (str): 要放入的交易資料csv檔名稱

    Returns:
        dataFrame: 預處理完成的資料
    """
    
    data = pd.read_csv(data_path)
    data['利潤'] = data['單價'] - data['成本']

    # 將訂單時間轉換成datetime形式
    data['訂單時間'] = data['訂單時間'].str.replace('T', ' ')
    data['訂單時間'] = pd.to_datetime(data['訂單時間'])
    
    return data

def selected_time_data(data: pd.DataFrame,
                       time: str,
                       year_start: str,
                       year_end: str):
    """此函數將回傳選定時間區間內的資料。

    Parameters:
        data (dataFrame): 要放入的交易資料
        time (str): 訂單時間欄位名稱
        year_start (str): 起始日期形式，舉例：'2019-1-1'.
        year_end (str): 終止日期形式，舉例：'2019-12-1'

    Returns:
        dataFrame: 選定時間區間內的資料
    """

    return data[(data[time] > parser.parse(year_start)) & (data[time] < parser.parse(year_end))]
    # return data[(data[time] > year_start) & (data[time] < year_end)]


def calculate_product_profit(data, product, profit):
    
    """此函數將計算每個產品的利潤貢獻。

    Parameters:
        data (dataFrame): 要放入的交易資料
        product (str): data裡面的「產品」欄位名稱
        profit (str): data裡面的「利潤」欄位名稱

    Returns:
        dataFrame: 每個產品的利潤貢獻資料
    """

    # 產品/貢獻比例：計算每一個產品的利潤總和
    product_profit = data.groupby(product, as_index=False)[profit].sum()
    product_profit = product_profit.sort_values(profit, ascending=False)

    # 產品的貢獻比
    product_profit['利潤佔比'] = product_profit[profit] / product_profit[profit].sum()
    product_profit['累計利潤佔比'] = product_profit['利潤佔比'].cumsum()

    # 產品比
    product_profit['累計系列次數'] = range(1, len(product_profit) + 1)
    product_profit['累計系列佔比'] = product_profit['累計系列次數'] / len(product_profit)

    # 四捨五入
    product_profit['累計系列佔比'],product_profit['累計利潤佔比'],product_profit['利潤佔比'] = round(product_profit['累計系列佔比'], 2), round(product_profit['累計利潤佔比'], 2), round(product_profit['利潤佔比'], 2)

    # 【st修改】
    # 輸出篩選產品貢獻度（利潤）資料
    # st.dataframe(product_profit)
    # generate_download_button(df=product_profit, filename="0_產品貢獻度（利潤）表")
    # product_profit.to_csv('0_產品貢獻度（利潤）表.csv', encoding='utf-8-sig')

    return product_profit



def save_and_plot(product_profit, product, profit, profit_percent, plot_yn = True):
    """此函數將繪製 產品/貢獻度比例圖，並回傳篩選後的產品資料

    Parameters:
        product_profit (dataFrame): 產品貢獻度（利潤）資料
        product (str): data裡面的「產品」欄位名稱
        profit (str): data裡面的「利潤」欄位名稱
        profit_percent (float): 篩選貢獻多少「%」利潤優先分析的產品. The default is 0.8.

    Returns:
        dataFrame: 篩選後的產品資料
    """

    if plot_yn:
        # 產品/貢獻度比例圖
        fig = px.bar(product_profit, x=product, y='利潤佔比',
                    hover_data=['累計利潤佔比', '累計系列佔比'],
                    color=profit, text='累計利潤佔比', height=400,
                    title='產品/貢獻度比例圖')
        fig.update_traces(textposition='outside')
        fig1 = fig
        
        # 【st修改】
        # plot(fig, filename='0_產品貢獻度比例圖.html', auto_open=False)

        # 篩選貢獻80%利潤的產品
        product_profit_selected = product_profit[product_profit['累計利潤佔比'] <= profit_percent]
        fig = px.bar(product_profit_selected, x=product, y='利潤佔比',
                    hover_data=['累計利潤佔比', '累計系列佔比'], color=profit,
                    text = '累計利潤佔比',
                    height= 600,
                    title='貢獻' + str(profit_percent*100) + '%的' + '產品貢獻度比例圖'
                    )
        fig.update_traces( textposition='outside')
        fig2 = fig
        
        
        # 【st修改】
        # plot(fig, filename= '1_' + '貢獻' + str(profit_percent*100) + '%的' + '產品貢獻度比例圖'+'.html',
        #     auto_open=False)

        # 【st修改】
        # 建議優先分析的產品
        # product_profit_selected.to_csv('1_貢獻' + str(profit_percent * 100) + '%的產品貢獻度比例表.csv', encoding='utf-8-sig')
        
    else:
        product_profit_selected = product_profit[product_profit['累計利潤佔比'] <= profit_percent]

    return product_profit_selected, fig1, fig2

#!SECTION - 附屬function


#SECTION - 【主要function】80/20法則（產品貢獻度）分析

def product_contribution(data, year_start, year_end, product, profit, profit_percent, time, which_series = ''):
    """透過 80/20 法則（產品貢獻度），篩選出建議優先分析的產品

    Parameters:
        data (str): 要放入的交易資料data
        year_start (str): 起始日期形式，舉例：'2019-1-1'.
        year_end (str): 終止日期形式，舉例：'2019-12-1'
        product (str): data裡面的「產品」欄位名稱
        profit (str): data裡面的「利潤」欄位名稱
        profit_percent (float): 篩選貢獻多少「%」利潤優先分析的產品. The default is 0.8.
        time (str): 訂單時間欄位名稱

    Returns:
        dataFrame: 建議優先分析的產品
    """
    
    # if which_series != '':
    #     data = data[data['系列'] ==which_series]
    # else:
    #     which_series = '全部'

    # 選定時間區間內的資料：回傳選定時間區間內的資料function
    sales_data = selected_time_data(data, time, year_start, year_end)

    # 計算每個產品的利潤貢獻function
    product_profit = calculate_product_profit(sales_data, product, profit)

    # 繪製 產品/貢獻度比例圖，並回傳篩選後的產品資料function
    product_profit_selected, fig1, fig2 = save_and_plot(product_profit, product, profit, profit_percent, plot_yn = True)

    # 【st修改】歸納資料
    # move_file(dectect_name='產品貢獻', folder_name='00_【金牛'+ product +'】貢獻度分析_'+which_series)
    # 建議優先分析的產品
    
    
    # 【st修改】將所有input資料回傳
    return sales_data, product_profit, product_profit_selected, fig1, fig2

#!SECTION - 【主要function】80/20法則（產品貢獻度）分析


#SECTION - user input區

def user_input_features():
    
    # para: year 
    year = st.sidebar.selectbox('Year', range(2018, 2020))
    month = st.sidebar.selectbox('Month', range(1, 13))
    date = st.sidebar.selectbox('Date', range(1, 31))
    
    # para: product 
    product = st.sidebar.selectbox('查看標的',('系列','去識別化會員編碼'))
    
    # para: profit 
    profit = st.sidebar.selectbox('計算標的',('利潤','單價'))
    
    # profit_percent
    profit_percent  = st.sidebar.slider('需要貢獻多少比例的利潤', 0.1,1.0,0.8)
    
    data = {'year': str(year)+ '-'+ str(month) +'-'+str(date),
            'product': product,
            'profit': profit,
            'profit_percent': profit_percent,
            }
    # features = pd.DataFrame(data, index=[0])
    return data

#!SECTION - user input區



#SECTION - 顯示區
# --------------- Display -------------------
def run_product_contribution():
    st.write("""
    # 零售案例 80/20法則 - 【金牛系列】貢獻度分析 App
    """)
    
    st.sidebar.header('User Input Features')
    
    uploaded_file = st.sidebar.file_uploader("請上傳您的CSV檔案", type=["csv"])
    
    if uploaded_file is not None:
        # sales_data = pd.read_csv(uploaded_file)
        sales_data = load_and_prepare_data(uploaded_file) 
    else:
        sales_data = load_and_prepare_data('sales_data_sample2.csv')
        # sales_data = load_and_prepare_data(sales_data) 
        # input_df = user_input_features()
    
    # Displays the user input features
    st.subheader('使用者輸入的資料表')
    
    if uploaded_file is not None:
        st.write(sales_data.iloc[0:100])
    else:
        st.write('等待使用者的資料上傳，目前所使用的是範例資料')
        st.write(sales_data)
    
    
    # Descriptive分析
    with st.expander("Data Types Summary"):
    	st.dataframe(sales_data.dtypes)
    
    with st.expander("Descriptive Summary"):
    	st.dataframe(sales_data.describe())
    
    with st.expander("廣告代號all Distribution"):
    	st.dataframe(sales_data['廣告代號all'].value_counts())
    
    with st.expander("系列 Distribution"):
    	st.dataframe(sales_data['系列'].value_counts())
    
    # 設定參數
    
    # 設定日期，例如：2019-1-1
    year_start = st.sidebar.date_input('起始日期', datetime(2018, 1, 1))
    year_end = st.sidebar.date_input('結束日期', datetime(2019, 12, 1)) 
    
    # 設定產品
    product = st.sidebar.selectbox('查看標的',('系列','產品'))
    
    # 設定 需要貢獻多少比例的利潤
    profit_percent = st.sidebar.slider('需要貢獻多少比例的利潤', 0.1,1.0,0.8)
    
    # 設定 要選擇的系列
    
    if product != '系列':
        seri = sales_data['系列'].unique().tolist()
        which_series = st.sidebar.selectbox('請問您要分析哪一個系列下的產品', (seri))
    else:
        which_series = ''
        
    # 顯示選擇的系列
    if which_series != '':
        sales_data = sales_data[sales_data['系列'] ==which_series]
    else:
        which_series = '全部'
    
    
    # st.write('您選擇的起始日期:', type(year_start))
    # st.write('您選擇的起始日期:', type(year_start))
    
    
    st.write('---')
    
    if st.button('開始分析'):
        
        # 分析
        sales_data, product_profit, product_profit_selected, fig1, fig2 = product_contribution(
                data = sales_data,
                year_start= str(year_start),
                year_end = str(year_end),
                product = product,
                profit = '利潤',
                profit_percent = profit_percent,
                time = '訂單時間',
                which_series=which_series
            )
        
        # Display
        st.subheader('產品貢獻度（利潤）分析')
        
        st.markdown('#### 產品/貢獻度比例表')
        st.dataframe(product_profit)
        generate_download_button(df=product_profit, filename="0_產品貢獻度（利潤）表")
        
        st.markdown('#### 產品/貢獻度比例圖')
        st.plotly_chart(fig1)
        
        
        st.write('---')
        
        st.subheader('貢獻' + str(profit_percent*100) + '%的' + '產品貢獻度比例分析')
        st.markdown('#### '+'貢獻' + str(profit_percent*100) + '%的' + '產品貢獻度比例分析表')
        st.dataframe(product_profit_selected)
        generate_download_button(df=product_profit_selected, filename='貢獻' + str(profit_percent*100) + '%的' + '產品貢獻度比例表')
        
        st.markdown('#### '+'貢獻' + str(profit_percent*100) + '%的' + '產品貢獻度比例分析圖')
        st.plotly_chart(fig2)
