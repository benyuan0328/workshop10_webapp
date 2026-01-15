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
from stqdm import stqdm
from general_layout import generate_download_button
import warnings
warnings.filterwarnings('ignore')

# from pyxlsb import open_workbook as open_xlsb
# pip install pyxlsb
# pip install xlsxwriter

# 這邊就用80/20
# 問題：如何將某檔案上傳，取95%利潤；product取會員？



#SECTION - 附屬function


def csv_downloader(df,filename):
	csvfile = df.to_csv()
	b64 = base64.b64encode(csvfile.encode("UTF-8-sig")).decode()
	new_filename = filename+"_{}_.csv".format(timestr)
	st.markdown("##### 下載上表 #####")
	href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">請點此下載</a>'
	st.markdown(href,unsafe_allow_html=True)
 
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
    
    
    # 將所有input資料回傳
    return sales_data, product_profit, product_profit_selected, fig1, fig2

def filter_data(series_data_month, column_name, series, compare_year1, compare_year2):
    '''
    根據給定的條件篩選數據。

    Parameters:
    - series_data_month: 每個系列在每一個月的利潤(pd.DataFrame)。
    - column_name: 進行篩選的欄位名稱。
    - series: 要分析的系列。
    - compare_year1: 初始年份，在此為2018(int)。
    - compare_year2: 最新的比較年份，在此2019(int)。

    Returns:
    - series_data_month_growth: 符合篩選條件的數據DataFrame。
    - test: 表示是否有比較年份的字符串。
    '''

    # 篩選出不同「系列」
    series_data_month_growth = series_data_month[series_data_month[column_name] == series] # 系列3
    
    # 做出「月份」欄位
    series_data_month_growth['訂單時間_年份'] = series_data_month_growth['訂單時間'].dt.year
    series_data_month_growth['訂單時間_月份'] = series_data_month_growth['訂單時間'].dt.to_period('M')

    # 根據每一「訂單時間_年份」與「訂單時間_月份」來計算「利潤」的「總和」
    series_data_month_growth = series_data_month_growth.groupby([column_name, '訂單時間_年份','訂單時間_月份'], as_index = False)['利潤'].sum()


    # 篩選出訂單時間_年份大於等於2018年，我們僅比較2018與2019年的成長率
    series_data_month_growth = series_data_month_growth[series_data_month_growth['訂單時間_年份'] >= compare_year1 ]
    # series_data_month_growth = series_data_month_growth[series_data_month_growth['訂單時間_年份'] >= int(compare_year1.strip('年'))]

    # 測試看看有無初始比較年份
    # 做出「年份」欄位
    test = series_data_month_growth[series_data_month_growth['訂單時間_年份']==compare_year1]
    test2 = series_data_month_growth[series_data_month_growth['訂單時間_年份']==compare_year2]

    if (len(test)>=1) and (len(test2)>=1):
        # print('有初始年份可以比較!')
        test = '有資料'
    else:
        # print('沒有初始年份可以比較')
        test = '跳過'
        series_data_month_growth = '沒有'
        return series_data_month_growth, test

    series_data_month_growth['月份'] = series_data_month_growth['訂單時間_月份'].dt.strftime('%m')

    # 將「訂單時間_年份」轉換
    series_data_month_growth = series_data_month_growth.pivot(index=['月份',column_name], columns='訂單時間_年份', values='利潤')

    series_data_month_growth = series_data_month_growth.reset_index()

    # 將欄位名稱2018與2019改變名稱
    series_data_month_growth = series_data_month_growth.rename(columns={compare_year1:str(compare_year1) +'年',
                                                                        compare_year2:str(compare_year2) +'年' })

    # 四捨五入小數點
    series_data_month_growth = round(series_data_month_growth)

    return series_data_month_growth, test

def calculate_monthly_growth(series_data_month_growth, compare_year1_str, compare_year2_str, month_end):
    '''
    計算每月的成長率。

    Parameters:
    - series_data_month_growth: 要進行計算的DataFrame。
    - compare_year1_str: 第一個比較年份的字符串。
    - compare_year2_str: 第二個比較年份的字符串。
    - month_end: 要計算成長率的最後一個月份。

    Returns:
    - series_data_month_growth: 包含成長率的DataFrame。
    '''
    # 計算每個月的年增長
    series_data_month_growth['年增成長'] = round( (series_data_month_growth[compare_year2_str] - series_data_month_growth[compare_year1_str] )/ series_data_month_growth[compare_year1_str], 2)

    # 12 月的年增長還沒有出來，所以我們只取到11月，我們將NaN值刪除
    series_data_month_growth = series_data_month_growth.dropna()

    # 將月份轉換成datetime時間格式
    series_data_month_growth['月份'] = series_data_month_growth['月份'].astype(int)

    # 11月還沒出來
    series_data_month_growth['月份'] = series_data_month_growth['月份'].astype(int)
    series_data_month_growth = series_data_month_growth[series_data_month_growth['月份'] <= month_end]

    # 計算「總年增成長率」：了解今年比較去年來說，增長了多少
    # domain knowledge(業內知識)：去年11月開始便沒有資料，所以只取到10月
    series_data_month_growth['總年增成長率'] = (series_data_month_growth[compare_year2_str].sum() - series_data_month_growth[compare_year1_str].sum())/ series_data_month_growth[compare_year1_str].sum()

    # series_data_month_growth.to_csv( 'Top' +  str(rank) +'_高獲利系列_「' + series + '」_「潛在」成長比較分析表.csv', encoding='utf-8-sig')

    return series_data_month_growth


def plot_monthly_analysis(series_data_month_growth, compare_year1_str, compare_year2_str, series, rank):
    '''
    ########## plot 1 ##########
    計算每月的成長率。

    Parameters:
    - series_data_month_growth: 要進行計算的DataFrame。
    - compare_year1_str: 第一個比較年份的字符串。
    - compare_year2_str: 第二個比較年份的字符串。
    - month_end: 要計算成長率的最後一個月份。

    Returns:
    - series_data_month_growth: 包含成長率的DataFrame。
    '''
    # 繪製系列每個月的「潛在」成長比較分析圖
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # bar chart for 2018 and 2019的收益
    fig.add_bar(x=series_data_month_growth['月份'], 
                y=series_data_month_growth[compare_year1_str], 
                name=compare_year1_str,
                secondary_y=False,)

    fig.add_bar(x=series_data_month_growth['月份'], 
                y=series_data_month_growth[compare_year2_str], 
                name=compare_year2_str,
                secondary_y=False,)

    # line chart for 年增成長
    fig.add_scatter(x=series_data_month_growth['月份'],
                    y=series_data_month_growth['年增成長'],
                    text=series_data_month_growth['年增成長'],
                    textposition='top center',
                    mode="lines+text+markers",
                    name="年增成長",
                    customdata=series_data_month_growth[['年增成長']],
                    hovertemplate="<br>".join([
                        "年增成長 = %{customdata[0]:.2f}",
                    ]
                    ), secondary_y=True,)

    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="grey",
                secondary_y=True)

    fig.update_layout(
        hovermode="x unified",
        dragmode='pan',
        # title="Financial analysis of manual budget allocation",
        xaxis_title="月份",
        yaxis_title="收益",
        xaxis=dict(
            tickmode='array',
            tickvals=series_data_month_growth['月份'],  # Set ticks at every x-value
            ticktext=[str(x) +'月' for x in series_data_month_growth['月份']]  # Optionally format tick labels
        )
    )
    plot(fig, filename='00_Top' +  str(rank) +'_高獲利系列_「' + series + '」_每月「潛在」成長比較分析圖.html',
        auto_open=False)
    
    return '00_Top' +  str(rank) +'_高獲利系列_「' + series + '」_每月「潛在」成長比較分析圖.html'


def calculate_annual_growth(series_data_month_growth, column_name, compare_year1_str, compare_year2_str):
    '''
    計算每年的成長率。

    Parameters:
    - series_data_month_growth: 要進行計算的DataFrame。
    - column_name: 分組的列名。
    - compare_year1_str: 第一個比較年份的字符串。
    - compare_year2_str: 第二個比較年份的字符串。

    Returns:
    - series_data_month_growth_all_year: 包含年增長率的DataFrame。
    '''
    # 請問如何給我系列3(不同系列)全年的「潛在」總成長比較分析圖
    series_data_month_growth_all_year = series_data_month_growth.groupby(column_name, as_index = False)[[compare_year1_str,compare_year2_str]].sum()

    # 計算「年增成長」
    series_data_month_growth_all_year['年增成長'] = round( (series_data_month_growth_all_year[compare_year2_str] - series_data_month_growth_all_year[compare_year1_str] )/ series_data_month_growth_all_year[compare_year1_str], 2)

    return series_data_month_growth_all_year

def plot_annual_analysis(series_data_month_growth_all_year, column_name, compare_year1_str, compare_year2_str, series, rank):
    '''
    繪製單一系列全年的潛在成長比較分析圖。

    Parameters:
    - series_data_month_growth_all_year: 包含年增長率的DataFrame。
    - column_name: 要顯示的列名。
    - compare_year1_str: 第一個比較年份的字符串。
    - compare_year2_str: 第二個比較年份的字符串。
    - series: 系列名稱。
    - rank: 系列的排名。

    Returns:
    - fig: Plotly 圖形物件。
    '''
    # plotly繪圖「總年增成長率」bar line chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_bar(x=series_data_month_growth_all_year[column_name],
                y=series_data_month_growth_all_year[compare_year2_str],
                name=compare_year2_str,
                secondary_y=False,)

    fig.add_bar(x=series_data_month_growth_all_year[column_name],
                y=series_data_month_growth_all_year[compare_year1_str],
                name=compare_year1_str,
                secondary_y=False,)

    # line chart for 年增成長
    fig.add_scatter(x=series_data_month_growth_all_year[column_name],
                    y=series_data_month_growth_all_year['年增成長'],
                    text=series_data_month_growth_all_year['年增成長'],
                    textposition='top center',
                    mode="lines+text+markers",
                    name="年增成長",
                    customdata=series_data_month_growth_all_year[['年增成長']],
                    hovertemplate="<br>".join([
                        "年增成長 = %{customdata[0]:.2f}",
                    ]
                    ), secondary_y=True,)

    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="grey",
                secondary_y=True)

    fig.update_layout(
        hovermode="x unified",
        dragmode='pan',
        # title="Financial analysis of manual budget allocation",
        xaxis_title=column_name,
        yaxis_title="收益",
        xaxis=dict(
            tickmode='array',
            tickvals=series_data_month_growth_all_year[column_name],  # Set ticks at every x-value
            ticktext=[str(x) +'月' for x in series_data_month_growth_all_year[column_name]]  # Optionally format tick labels
        )
    )
    plot(fig, filename='01_Top' +  str(rank) +'_高獲利系列_「' + series + '」_全年「潛在」成長比較分析圖.html',
        auto_open=False)

    return 'Top' +  str(rank) +'_高獲利系列_「' + series + '」_全年「潛在」成長比較分析圖.html'

def potiential_growth(data, series, column_name, rank, compare_year1, compare_year2, month_end, plot_yn=False):
    '''
    計算指定系列的潛在增長率分析。

    Parameters:
    - data_path: 分析的DataFrame數據。
    - series: 要分析的系列。
    - column_name: 列名。
    - rank: 系列的排名。
    - compare_year1: 第一年的比較年份。
    - compare_year2: 第二年的比較年份。
    - month_end: 分析的最後月份。
    - plot_yn: 是否繪製圖表，默認為False。

    Returns:
    - series_data_month_growth: 包含每月潛在增長率的DataFrame。
    - series_data_month_growth_all_year: 包含每年潛在增長率的DataFrame。
    - test: 有資料或跳過的標記。
    '''
        


    # 產出：回傳選定時間區間內的資料
    
    sales_data = selected_time_data(data = data, 
                                    time = '訂單時間', 
                                    year_start=compare_year1,
                                    year_end=compare_year2)
    
    # 將compare_year1的2018切出來，並且轉換成int
    compare_year1 = int(compare_year1.split('-')[0])
    compare_year2 = int(compare_year2.split('-')[0])
    
    series_data_month_growth, test = filter_data(sales_data, column_name, series, compare_year1, compare_year2)

    if test == '跳過':
        # test = '跳過'
        # series_data_month_growth = '沒有'
        return series_data_month_growth, '沒有', test
    else:
        compare_year1_str = str(compare_year1) +'年'
        compare_year2_str = str(compare_year2) +'年'

        # series_data_month_growth = round(series_data_month_growth)

        series_data_month_growth = calculate_monthly_growth(series_data_month_growth, compare_year1_str, compare_year2_str,month_end)
        series_data_month_growth = series_data_month_growth[series_data_month_growth['月份'] <= month_end]

        if plot_yn:
            plot_monthly_analysis(series_data_month_growth, compare_year1_str, compare_year2_str, series, rank)
        # else:
        #     print('不繪圖')

        series_data_month_growth_all_year = calculate_annual_growth(series_data_month_growth, column_name, compare_year1_str, compare_year2_str)

        if plot_yn:
            plot_annual_analysis(series_data_month_growth_all_year, column_name, compare_year1_str, compare_year2_str, series, rank)
        # else:
        #     print('不繪圖')

        return series_data_month_growth, series_data_month_growth_all_year, '有資料'
    
    
def potiential_growth_all(data, column_name, rank, compare_year1, compare_year2, month_end, 
                          product_profit_selected, plot_yn=False,
                          size = 5):
    '''
    計算總系列「潛在」成長的明星商品比較分析。

    Parameters:
    - data: 分析的DataFrame數據。
    - column_name: 列名。
    - rank: 系列的排名。
    - compare_year1: 第一年的比較年份。
    - compare_year2: 第二年的比較年份。
    - month_end: 分析的最後月份。
    - plot_yn: 是否繪製圖表，默認為False。

    Returns:
    - series_data_month_growth_all_year_df: 包含每年潛在增長率的DataFrame。
    - series_data_month_growth_all_year_df_top: 包含每年潛在增長率的前80%的DataFrame。
    - series_data_month_growth_df: 包含每月潛在增長率的DataFrame。
    '''
    
    # 存每一系列的「全年「潛在」成長比較分析」
    series_data_month_growth_all_year_list = []

    # 存每一系列的「每月「潛在」成長比較分析表」
    series_data_month_growth_list = []

    # 載入tqdm進度條
    all_series = data[column_name].unique()

    for series in stqdm(all_series):
        series_data_month_growth, series_data_month_growth_all_year, test = potiential_growth(
                                                    data = data, 
                                                    series = series, 
                                                    column_name = column_name,
                                                    rank = rank, 
                                                    compare_year1 = compare_year1, 
                                                    compare_year2 = compare_year2,
                                                    month_end = month_end,
                                                    plot_yn = plot_yn)

        if test == '有資料':
            # 存每一系列的「每月「潛在」成長比較分析表」
            series_data_month_growth_list.append(series_data_month_growth)

            # 存每一系列的「全年「潛在」成長比較分析」
            series_data_month_growth_all_year_list.append(series_data_month_growth_all_year)
        
        # elif test == '跳過':
        #     print(series, '跳過')
            
        
    # # 合併總體「潛在」全年成長比較分析
    series_data_month_growth_all_year_df = pd.concat(series_data_month_growth_all_year_list, axis=0)

    series_data_month_growth_all_year_df

    # 排序年增成長，由高到低
    series_data_month_growth_all_year_df = series_data_month_growth_all_year_df.sort_values('年增成長', ascending=False)

    series_data_month_growth_all_year_df


    series_data_month_growth_all_year_df['潛力開發排序'] = range(1, len(series_data_month_growth_all_year_df) + 1)

    # 【st修正】
    series_data_month_growth_all_year_df#.to_csv('0_總體「潛在」成長比較分析.csv', encoding= 'utf-8-sig')
    

    # # 合併全部系列「潛在」每月成長比較分析表

    series_data_month_growth_df = pd.concat(series_data_month_growth_list, axis=0)

    # 計算每一個系列的在月份成長率的序列長度（eg. 若size==5， 即有5個月的成長率比較基礎的資料）
    series_length_mon = series_data_month_growth_df.groupby([column_name], as_index=False).size()

    # 使用merge合併
    series_data_month_growth_all_year_df = series_data_month_growth_all_year_df.merge(series_length_mon, on=column_name, how='inner')

    # 請抓出 size > 5的系列(即至少有6個月的成長率比較基礎的資料)
    series_data_month_growth_all_year_df_top = series_data_month_growth_all_year_df[series_data_month_growth_all_year_df['size']>=size]


    # 請抓出年增長率>0 的系列
    series_data_month_growth_all_year_df_top = series_data_month_growth_all_year_df_top[series_data_month_growth_all_year_df_top['年增成長']>0]

    # 導出「貢獻80.0%的高利潤產品系列」，並將其從「全部潛在系列」中刪除
    # profit_data80 = pd.read_csv('0_貢獻80.0%的高利潤產品系列表.csv')
    # profit_data80 = save_and_plot(product_profit = product_profit,
    #                                 product = column_name,
    #                                 profit = '利潤',
    #                                 profit_percent = profit_percent,
    #                                 plot_yn=False)

    # product_profit_selected = product_profit[product_profit['累計利潤佔比']<=profit_percent]
    # here
    series_data_month_growth_all_year_df_top = series_data_month_growth_all_year_df_top[~series_data_month_growth_all_year_df_top[column_name].isin(product_profit_selected[column_name])]

    # 【st修正】
    series_data_month_growth_all_year_df_top#.to_csv('1_全部系列「潛在」全年成長比較分析表v2.csv', encoding='utf-8-sig')

    # 【st修正】
    series_data_month_growth_df#.to_csv('2_全部系列「潛在」每月成長比較分析表.csv', encoding='utf-8-sig')

    # '0_總體「潛在」成長比較分析.csv'
    # '1_全部系列「潛在」全年成長比較分析表v2.csv'
    # '2_全部系列「潛在」每月成長比較分析表.csv'
    return series_data_month_growth_all_year_df,series_data_month_growth_all_year_df_top, series_data_month_growth_df

def visualize_and_handle_files(series_data_month_growth_all_year_df_top: pd.DataFrame,
                               column_name: str,
                               compare_year1: int,
                               compare_year2: int,
                               which_series: str = ''):
    """
    將資料視覺化並處理文件。

    Parameters:
        series_data_month_growth_all_year_df_top (pd.DataFrame): 含有系列潛在成長分析結果的數據框(DataFrame)。
        column_name (str): 系列欄位的名稱。
        compare_year1 (int): 第一個比較的年份。
        compare_year2 (int): 第二個比較的年份。
    """

    # 代理年份，從 2018 --> 2018年
    compare_year1_str = str(compare_year1) +'年'
    compare_year2_str = str(compare_year2) +'年'
    series_data_month_growth_all_year_df_top = series_data_month_growth_all_year_df_top.sort_values(by=compare_year2_str, ascending=False)

    # 繪製總體「潛在」全年成長比較分析圖
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=series_data_month_growth_all_year_df_top[column_name], y=series_data_month_growth_all_year_df_top[compare_year1_str], name=compare_year1_str, secondary_y=False,)
    fig.add_bar(x=series_data_month_growth_all_year_df_top[column_name], y=series_data_month_growth_all_year_df_top[compare_year2_str], name=compare_year2_str, secondary_y=False,)
    fig.add_scatter(x=series_data_month_growth_all_year_df_top[column_name], y=series_data_month_growth_all_year_df_top['年增成長'], text=series_data_month_growth_all_year_df_top['年增成長'], textposition='top center', mode="lines+text+markers", name="年增成長", customdata=series_data_month_growth_all_year_df_top[['年增成長']], hovertemplate="<br>".join(["年增成長 = %{customdata[0]:.2f}",]), secondary_y=True,)
    fig.add_hline(y=0, line_dash="dash", line_color="grey", secondary_y=True)
    fig.update_layout(
        hovermode="x unified",
        dragmode='pan',
        xaxis_title="系列",
        yaxis_title="收益",
        xaxis=dict(
            tickmode='array',
            tickvals=series_data_month_growth_all_year_df_top[column_name],
        )
    )
    fig.update_layout(title="總體「潛在」全年成長比較分析圖")
    
    
    # plot(fig, filename='3_全部系列「潛在」全年成長比較分析圖.html', auto_open=False)
    
    # 【st修正】
    # move_file(dectect_name = '潛在', folder_name = '01_「潛在」' +column_name +'成長比較分析_' + which_series)
    
    return fig


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

def plot_data_process(sales_data: pd.DataFrame,
                      year_start: str = '2019-1-1',
                      year_end: str = '2019-12-1',
                      time: str = '訂單時間',
                      series: str = '系列1',
                      column_name: str = '系列'
                      ):
    '''
    Parameters
    -------------------
    sales_data：Dataframe
        整理繪圖所需資料
    series：str
        選擇系列
        
    Returns first_plot, second_plot_data
    -------------------
    '''
    
    # sales_data = pd.read_csv(data_path)
    
    # 選擇系列
    sales_data = sales_data[sales_data[column_name] == series]
    
    # 轉換時間
    # sales_data['訂單時間'] = pd.to_datetime(sales_data['訂單時間'])
    sales_data = selected_time_data(sales_data, time, year_start, year_end)

    
    plot_data = sales_data[['單價', '成本', '系列', '產品', '訂單時間', '廣告代號all']]
    plot_data['利潤'] = plot_data['單價'] - plot_data['成本']
    plot_data['購買人數'] = 1
    # 總購買、總利潤、平均利潤
    total_buy = plot_data.groupby("廣告代號all")['購買人數'].sum()
    total_profit = plot_data.groupby("廣告代號all")['利潤'].sum()
    mean_profit = plot_data.groupby("廣告代號all")['利潤'].mean()

    # 整理第一&二個圖資料：總購買人數、利潤占比 & 總利潤、平均利潤
    first_plot = pd.concat([total_buy, total_profit, mean_profit], axis=1).reset_index()
    first_plot.columns = ['廣告代號', '總購買人數', '總利潤', "平均利潤"]
    first_plot['利潤占比'] = first_plot['總利潤'] / first_plot['總利潤'].sum()
    first_plot_data = first_plot.sort_values('總購買人數', ascending=False)

    # 整理三個圖資料：廣告時間序列圖
    second_plot = plot_data[['單價', '成本', '系列', '產品', '訂單時間', '廣告代號all', '利潤']]
    top3_ad = total_profit.reset_index().sort_values("利潤", ascending=False)['廣告代號all'].tolist()[:10]
    # 找出前n名(最多前三)(根據total profit)的資料
    second_plot = second_plot[second_plot['廣告代號all'].str.contains("|".join(top3_ad))]
    second_plot['月'] = second_plot['訂單時間'].dt.month
    second_plot_data = second_plot.groupby(['廣告代號all', "月"])['利潤'].sum().reset_index()

    return plot_data, first_plot_data, second_plot_data


def ad_plot(plot_data, first_plot_data, second_plot_data):
    '''
    Parameters
    -------------------
    first_plot：Dataframe
        繪製第 一 & 二 張圖所需資料
    second_plot_data：Dataframe
        繪製第 三 張圖所需資料
    '''

    # 繪製第0張圖：廣告效益分析 - 系列、產品與廣告效益朝陽圖分析
    fig = px.sunburst(plot_data, 
                    path=['系列', '產品','廣告代號all'], values='利潤', color='利潤')

    fig0 = fig
    
    # 【st修改】
    # plot(fig, filename='01_廣告分析_系列、產品與廣告朝陽圖.html', auto_open=False)

    # 繪製第1張圖：總購買人數、利潤占比
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=first_plot_data['廣告代號'],
                         y=first_plot_data['總購買人數'],
                         name='總購買人數'),
                  secondary_y=False)

    fig.add_trace(go.Scatter(x=first_plot_data['廣告代號'],
                             y=first_plot_data['利潤占比'],
                             mode='lines+markers',
                             name="利潤占比"),
                  secondary_y=True)

    fig.update_layout(title = "廣告效益分析 - 利潤占比&購買人數",)
    fig.update_yaxes(title_text="<b>利潤占比</b>", secondary_y=True)
    fig.update_yaxes(title_text="<b>總購買人數</b>", secondary_y=False)
    
    fig1 = fig
    
    # 【st修改】
    # plot(fig, filename='02_廣告分析_利潤占比&購買人數.html', auto_open=False)

    # 繪製第2張圖：廣告時間序列圖
    fig2 = go.Figure()
    ad_list = list(set(second_plot_data['廣告代號all'].tolist()))
    for ad in ad_list:
        temp_second_data = second_plot_data[second_plot_data["廣告代號all"] == ad]
        fig2.add_trace(go.Scatter(x=temp_second_data['月'],
                                  y=temp_second_data['利潤'],
                                  mode='markers+lines',
                                  name=ad))
    fig2.update_yaxes(title_text="<b>利潤</b>")
    fig2.update_layout(title = "廣告效益分析 - 廣告時間序列圖",
                       xaxis_range=[1,12])

    plot(fig2, filename='03_廣告分析_廣告時間序列圖.html', auto_open=False)

    # 系列、產品與廣告朝陽圖, 利潤占比&購買人數圖, 廣告時間序列圖
    return fig0, fig1, fig2

#!SECTION - 附屬function


#SECTION - 【主要function】潛在明星商品分析

def run_ad_plot(
    sales_data: pd.DataFrame,
    series: str = '系列1',
    column_name: str = '系列',
    year_start: str = '2019-1-1',
    year_end: str = '2019-12-1',
    time: str = '訂單時間'
):
    '''
    Parameters
    -------------------
    None
    '''
    
    plot_data, first_plot_data, second_plot_data = plot_data_process( sales_data=sales_data,
                                                            series=series,
                                                            column_name=column_name,
                                                            year_start=year_start,
                                                            year_end=year_end,
                                                            time=time
                                                            )
    
    fig0, fig1, fig2 = ad_plot(plot_data = plot_data,
                                first_plot_data=first_plot_data,
                                second_plot_data = second_plot_data)
    # 
    return fig0, fig1, fig2 ,first_plot_data, second_plot_data

#!SECTION - 【主要function】80/20法則（產品貢獻度）分析


#SECTION - 顯示區
def run_ad_effect_analysis():
        
    st.write("""
            # 廣告效益分析
            """)
    
    st.sidebar.header('User Input Features')
    
    uploaded_file = st.sidebar.file_uploader("請上傳您的CSV檔案", type=["csv"])
    
    if uploaded_file is not None:
        sales_data = load_and_prepare_data(data_path = uploaded_file)
    else:
        sales_data = load_and_prepare_data('sales_data_sample2.csv') 
    
    # Displays the user input features
    st.subheader('使用者輸入的資料表')
    
    if uploaded_file is not None:
        st.write(sales_data.iloc[0:100])
    else:
        st.write('等待使用者的資料上傳，目前所使用的是範例資料')
        st.write(sales_data)
    
    
    # 設定參數
    
    # 設定日期，例如：2019-1-1
    year_start = st.sidebar.date_input('起始日期', datetime(2018, 1, 1))
    year_end = st.sidebar.date_input('結束日期', datetime(2019, 12, 1)) 
    
    # 設定產品
    column_name = st.sidebar.selectbox('查看標的',('系列','產品'))
    
    # 設定 要選擇的系列
    
    if column_name == '系列':
        seri = sales_data['系列'].unique().tolist()
        which_series = st.sidebar.selectbox('請問您要分析哪一個系列', (seri))
    elif column_name == '產品':
        # sales_data = sales_data[sales_data['系列'] ==which_series]
        seri = sales_data['產品'].unique().tolist()
        which_series = st.sidebar.selectbox('請問您要分析哪一個產品', (seri))
    
        
    
    # st.write('您選擇的起始日期:', type(year_start))
    # st.write('您選擇的起始日期:', type(year_start))
    
    
    st.write('---')
    
    # 設定按鈕：當按鈕被按下時，將會執行分析，並且將按鈕diabled
    if 'run_button' in st.session_state and st.session_state.run_button == True:
        st.session_state.running = True
    else:
        st.session_state.running = False
    
    # st.write(st.session_state.running)
    # if st.button('開始分析', disabled=st.session_state.running, key='run_button'):
    if st.button('開始分析', disabled=st.session_state.running, key='run_button'):
        # st.write(st.session_state.running)
        # 分析
        
        # 系列、產品與廣告朝陽圖, 利潤占比&購買人數圖, 廣告時間序列圖
        # 問題：如何形塑 run_ad_plot 的參數
                
        fig0, fig1, fig2,first_plot_data, second_plot_data = run_ad_plot(
            sales_data = sales_data,
            series = which_series,
            column_name = column_name,
            year_start = str(year_start),
            year_end = str(year_end),
            time = '訂單時間'
        )
        
        # Display
        # 問題：呈現圖表
        # 第一個 subheader 顯示「系列、產品與廣告朝陽圖」
        st.subheader('系列、產品與廣告朝陽圖')
        st.plotly_chart(fig0)
        st.write('---') # 分隔線
            
        # 第二個 subheader 顯示「利潤占比&購買人數圖」
        st.subheader('利潤占比&購買人數圖')
        st.plotly_chart(fig1)
        st.dataframe(first_plot_data)
        csv_downloader(df=first_plot_data, filename="01_利潤占比&購買人數圖")
        st.write('---') # 分隔線
        
        # 第三個 subheader 顯示「廣告時間序列圖」
        st.subheader('廣告時間序列圖')
        st.plotly_chart(fig2)
        st.dataframe(second_plot_data)
        csv_downloader(df=second_plot_data, filename="02_廣告時間序列圖")
        
        st.write('---') # 分隔線
        
        # hint：使用 st.plotly_chart(fig)
        # st.subheader('系列、產品與廣告朝陽圖')
        # st.plotly_chart(fig0)
        # st.write('---')
        
        if st.button('重新來過'):
            # st.session_state.run_button = True
            st.experimental_rerun()
            
# 學員分析系列3、1、8