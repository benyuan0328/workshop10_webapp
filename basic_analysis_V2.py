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
import warnings
warnings.filterwarnings('ignore')

# %%
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

    # 輸出篩選產品貢獻度（利潤）資料
    product_profit.to_csv('0_產品貢獻度（利潤）表.csv', encoding='utf-8-sig')

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
        plot(fig, filename='0_產品貢獻度比例圖.html', auto_open=False)

        # 篩選貢獻80%利潤的產品
        product_profit_selected = product_profit[product_profit['累計利潤佔比'] <= profit_percent]
        fig = px.bar(product_profit_selected, x=product, y='利潤佔比',
                    hover_data=['累計利潤佔比', '累計系列佔比'], color=profit,
                    text = '累計利潤佔比',
                    height= 600,
                    title='貢獻' + str(profit_percent*100) + '%的' + '產品貢獻度比例圖'
                    )
        fig.update_traces( textposition='outside')
        plot(fig, filename= '1_' + '貢獻' + str(profit_percent*100) + '%的' + '產品貢獻度比例圖'+'.html',
            auto_open=False)

        # 建議優先分析的產品
        product_profit_selected.to_csv('1_貢獻' + str(profit_percent * 100) + '%的產品貢獻度比例表.csv', encoding='utf-8-sig')
    else:
        product_profit_selected = product_profit[product_profit['累計利潤佔比'] <= profit_percent]

    return product_profit_selected


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

    # 選定時間區間內的資料：回傳選定時間區間內的資料function
    sales_data = selected_time_data(data, time, year_start, year_end)

    # 計算每個產品的利潤貢獻function
    product_profit = calculate_product_profit(sales_data, product, profit)

    # 繪製 產品/貢獻度比例圖，並回傳篩選後的產品資料function
    product_profit_selected = save_and_plot(product_profit, product, profit, profit_percent, plot_yn = True)

    # 【st修改】歸納資料
    move_file(dectect_name='產品貢獻', folder_name='00_【金牛'+ product +'】貢獻度分析_'+which_series)
    # 建議優先分析的產品
    return product_profit_selected


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
                          product_profit_selected, plot_yn=False):
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
    from tqdm import tqdm
    all_series = data[column_name].unique()

    for series in tqdm(all_series):
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

    series_data_month_growth_all_year_df.to_csv('0_總體「潛在」成長比較分析.csv', encoding= 'utf-8-sig')
    

    # # 合併全部系列「潛在」每月成長比較分析表

    series_data_month_growth_df = pd.concat(series_data_month_growth_list, axis=0)

    # 計算每一個系列的在月份成長率的序列長度（eg. 若size==5， 即有5個月的成長率比較基礎的資料）
    series_length_mon = series_data_month_growth_df.groupby([column_name], as_index=False).size()

    # 使用merge合併
    series_data_month_growth_all_year_df = series_data_month_growth_all_year_df.merge(series_length_mon, on=column_name, how='inner')

    # 請抓出 size > 5的系列(即至少有6個月的成長率比較基礎的資料)
    series_data_month_growth_all_year_df_top = series_data_month_growth_all_year_df[series_data_month_growth_all_year_df['size']>5]


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

    series_data_month_growth_all_year_df_top.to_csv('1_全部系列「潛在」全年成長比較分析表v2.csv', encoding='utf-8-sig')

    series_data_month_growth_df.to_csv('2_全部系列「潛在」每月成長比較分析表.csv', encoding='utf-8-sig')


    return series_data_month_growth_all_year_df,series_data_month_growth_all_year_df_top

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
    plot(fig, filename='3_全部系列「潛在」全年成長比較分析圖.html', auto_open=False)

    move_file(dectect_name = '潛在', folder_name = '01_「潛在」' +column_name +'成長比較分析_' + which_series)
    
    
def analyze_potential_growth_data(data: str,
                                  column_name: str, 
                                  compare_year1_str: str, 
                                  compare_year2_str: str, 
                                  month_end: int,
                                  product_profit_selected:pd.DataFrame,
                                  which_series: str = ''
                                  ) -> pd.DataFrame:
    """
    處理潛在成長的主要函數。

    Parameters:
        all_data (str): 包含原始數據的文件路徑。
        column_name (str): 系列名稱的列名。
        compare_year1 (int): 第一個比較的年份。
        compare_year2 (int): 第二個比較的年份。
        month_end (int): 比較的月份。

    Returns:
        pd.DataFrame: 包含潛在成長分析結果的數據框(DataFrame)。
    """
    # 資料處理
    # all_data = load_and_prepare_data(data_path = data_path)
    
    # 分析
    series_data_month_growth_all_year_df, series_data_month_growth_all_year_df_top = potiential_growth_all(
            data = data,
            column_name = column_name,
            rank = 0,
            compare_year1 = compare_year1_str,
            compare_year2 = compare_year2_str,
            month_end = month_end,
            product_profit_selected = product_profit_selected)
    
    # 繪圖
    compare_year1 = int(compare_year1_str.split('-')[0])
    compare_year2 = int(compare_year2_str.split('-')[0])
    
    visualize_and_handle_files(series_data_month_growth_all_year_df_top, 
                               column_name, 
                               compare_year1, compare_year2,
                               which_series)

    return series_data_month_growth_all_year_df_top



