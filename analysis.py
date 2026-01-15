# import progressbar
import plotly.express as px
from dateutil import parser
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

import pandas as pd
import re
import numpy as np
# from progressbar import progressbar 
from apyori import apriori


def pricing_apriori_fun(
                    series = '系列1', 
                    segment_name = 'mid_mid',
                    client_seg_list='',
                    month_result_list_first='',
                    month=12,
                    sales_data='',
                    ):


    client_seg_list_month = client_seg_list[client_seg_list['訂單時間'].dt.month == month]

    client_seg_list_tmp = client_seg_list_month[client_seg_list_month['segmentation']==segment_name]
    client_seg_list_tmp = client_seg_list_tmp.drop_duplicates('訂單編號')
    client_seg_list_tmp = client_seg_list_tmp[['訂單編號']]
    client_seg_list_new_group = sales_data.merge(client_seg_list_tmp, on='訂單編號', how='right')

    # 選定特殊月份
    # 2. 計算所有含有該產品的「訂單編號」
    data_tmp = month_result_list_first[month_result_list_first['month']==month]
    
    data_tmp = data_tmp.fillna('沒有產品')

    orderid = client_seg_list_new_group[client_seg_list_new_group['產品'] == data_tmp['購買的產品'].iloc[0] ][['訂單編號']]
    orderid
    client_seg_list_tmp = client_seg_list_new_group.merge(orderid, on='訂單編號', how='right')

    
    # -------抓出一起搭配購買的產品訂單編號-----------
    # 這樣因爲會有不同種的產品搭配所以要用for loop來逐一記錄
    with_prod_col = [i for i in data_tmp.columns if '產品組合' in i and '產品組合購買機率' not in i and '購買的產品' not in i and '產品組合總利潤' not in i and '產品組合佔綜合總利潤百分比' not in i   ]

    orderid_tmp_list = []
    prod_tmp_list = []
    for col in with_prod_col:
        if data_tmp[col].iloc[0] != '沒有產品':
            # print(col)
            prod_tmp_list.append(data_tmp[col].iloc[0])
            orderid_tmp = client_seg_list_tmp[client_seg_list_tmp['產品'].str.contains(data_tmp[col].iloc[0]+'$')][['訂單編號']]
            orderid_tmp_list.append(orderid_tmp)

    orderid_tmp = pd.concat(orderid_tmp_list)
    orderid_tmp = orderid_tmp.drop_duplicates(['訂單編號'])
    # prod_tmp_list = '|'.join(prod_tmp_list)

    # 這邊要確認多種產品搭配是否都在一個訂單裡面
    client_seg_list_tmp_tt_list = []
    for num in orderid_tmp['訂單編號']:
        client_seg_list_tmp_tt = client_seg_list_tmp[client_seg_list_tmp['訂單編號']==num]

        good = True
        for i in prod_tmp_list:
            if len(client_seg_list_tmp_tt[client_seg_list_tmp_tt['產品']==i])==0:
                good = False
            # else:
                

        if good:
            client_seg_list_tmp_tt_list.append(client_seg_list_tmp_tt )
                    
    client_seg_list_tmp_tt_list = pd.concat(client_seg_list_tmp_tt_list)


    # 抓取主要購買產品與一起搭配購買的產品訂單編號
    # 舉例：買1-11也有一起買1-12的產品訂單編號
    # 目的：計算兩個產品一起搭配購買的利潤
    client_seg_list_tmp_buy_with = client_seg_list_tmp.merge(orderid_tmp, on='訂單編號', how='right')

    with_prod_col = [i for i in data_tmp.columns if '產品組合' in i and '產品組合購買機率' not in i and '產品組合總利潤' not in i and '產品組合佔綜合總利潤百分比' not in i   ]
    with_prod_col = with_prod_col +['購買的產品']
    matech_l = []
    for col in with_prod_col:
        if data_tmp[col].iloc[0] != '沒有產品':
            # print(col)
            matech = client_seg_list_tmp_tt_list[client_seg_list_tmp_tt_list['產品'].str.contains(data_tmp[col].iloc[0] + '$')]
            matech_l.append(matech)

    matech_l = pd.concat(matech_l)

    # 將訂單groupby
    matech_l = matech_l.groupby(['訂單編號'], as_index = False)[['單價', '利潤']].mean()
    
    matech_l
    matech_l['分段'] = pd.cut(matech_l["單價"], np.arange(matech_l["單價"].min()-10, matech_l["單價"].max()+ 30, 30) )
    a= matech_l


    matech_l2 = matech_l.groupby( pd.cut(matech_l["單價"], np.arange(matech_l["單價"].min()-10, matech_l["單價"].max()+ 30, 30)))['利潤'].sum()
    matech_l2 = matech_l2.reset_index()
    matech_l2['單價區間'] = matech_l2["單價"].astype(str)
    ex = matech_l2['單價區間'].str.split(',', expand= True)
    matech_l2['單價區間'] = ('$' +ex[0].str.replace('(','') +'~' + ex[1].str.replace(']','')).str.replace('.0$','').str.replace('.0~','~') +'元新臺幣'
    matech_l2 = matech_l2.sort_values('利潤', ascending= False)
    matech_l2['建議採納售價區間排名'] = range(1,len(matech_l2)+1)

    matech_l2["all"] = '購買【'+data_tmp['購買的產品'].iloc[0] +'】並搭配【'  + data_tmp['產品組合1'].iloc[0] +'】' + '；建議採納售價 = ' + matech_l2['單價區間'].iloc[0]

    fig = px.treemap(matech_l2, path=['all', '單價區間'], 
            values='利潤', color='單價區間',
            title=matech_l2["all"].iloc[0]
            )

    plot(fig, filename='05_'+series + "_區隔_" + segment_name+ "_商品搭配【最佳定價】"+ '_' + str(month) +'月_'+ matech_l2["all"].iloc[0] +'.html',
                auto_open= False)

    del matech_l2['單價']
    matech_l2.to_csv('05_'+series + "_區隔_" + segment_name+ "_商品搭配【最佳定價】"+ '_' + str(month) +'月_'+ matech_l2["all"].iloc[0] +'.csv', encoding = 'utf-8-sig')
    # fig = px.histogram( matech_l, x="單價", y="利潤", 
    #                     marginal="violin",
    #                     nbins=14
    #                         )


    return matech_l2




def apriori_fun_for_month(series = '系列1', segment_name = 'mid_mid',min_support=0.001, min_lift=1.000000001,
        client_seg_list='',sales_data=''  ):

    client_seg_list_tmp = client_seg_list[client_seg_list['segmentation']==segment_name]

    client_seg_list_tmp = client_seg_list_tmp.drop_duplicates('訂單編號')

 
    client_seg_list_tmp = client_seg_list_tmp[['訂單編號']]
 
    client_seg_list_tmp = sales_data.merge(client_seg_list_tmp, on='訂單編號', how='right')

 
    record=[]
    # series_data = sales_data[sales_data["系列"] == '系列1']
    order_number = np.unique(client_seg_list_tmp["訂單編號"])
    # try:
    #     os.mkdir(series + "_區隔_" + segment_name+ "_商品搭配分析") # 創建各系列資料夾
    # except:
    #     print(series + "_區隔_" + segment_name+ "_商品搭配分析")
    
    print('資料處理中...')

    for i in order_number:
        cart = sales_data[sales_data["訂單編號"]==i]["產品"].values
        record.append(cart)
        # print( '訂單編號： ', sales_data['訂單編號'].iloc[0])
        # print(cart)

 
    record
    print('機器學習處理中...')
    # 分析
    association_rules = apriori(record, min_support=min_support, min_lift=min_lift) # 建立分析規則
    association_results = pd.DataFrame(association_rules) # 結果資料表

 
    association_results
 
    # 建立各商品組合及 support 的分表
    thebig = association_results["items"].str.len().max()
    separate = pd.DataFrame(association_results["items"].values.tolist(),columns=[ "p"+ str(x) for x in range(thebig)])
    separate["support"] = association_results["support"].values

    # 將 confidence 與 lift 從 association_results["ordered_statistics"] 中取出
    separate["confidence"] = association_results["ordered_statistics"].str[0].str[2]
    separate["lift"]=association_results["ordered_statistics"].str[0].str[3]
    print('完成！')
    return separate, client_seg_list_tmp

 
def integrate_apriori_fun(series = '系列1', 
                        segment_name = 'mid_mid',
                        min_support=0.005, 
                        min_lift=1.000000001,
                        client_seg_list = '',
                        sales_data = '' ,
                        month = 8
                        ):
    
    # 選定特殊月份
    client_seg_list_month = client_seg_list[client_seg_list['訂單時間'].dt.month == month]

    # 購物籃分析的結果
    # 月與人篩選上可能有問題
    association_results, client_seg_list_new_group = apriori_fun_for_month(
            series = series, 
            segment_name = segment_name,
            min_support= min_support, 
            min_lift = min_lift,
            client_seg_list = client_seg_list_month,
            sales_data = sales_data  )
    
    # finance
    financial_association_results = financial_apriori_fun(association_results,
                                                client_seg_list_new_group) 


    # '產品組合佔綜合總利潤百分比' > '不搭配產品（單品）佔綜合總利潤百分比'
    financial_association_results_selected = financial_association_results[financial_association_results['產品組合佔綜合總利潤百分比']>financial_association_results[ '不搭配產品（單品）佔綜合總利潤百分比']]

    # 利潤排序
    financial_association_results_selected = financial_association_results_selected.sort_values('產品組合總利潤', ascending = False)

    # 設定排名
    financial_association_results_selected['利潤排名'] = range(1, len(financial_association_results_selected)+1)

    # 輸出csv
    financial_association_results_selected.to_csv( '01_'+series + "_區隔_" + segment_name+ "_商品搭配分析推薦表"+ '_' + str(month) +'月' +".csv", encoding = 'utf-8-sig')

    # 輸出每一個系列
    prod_unq = financial_association_results_selected['購買的產品'].unique()

    for i in prod_unq:
        financial_association_results_selected_tmp = financial_association_results_selected[financial_association_results_selected['購買的產品']==i]
        financial_association_results_selected_tmp = financial_association_results_selected_tmp.sort_values('產品組合總利潤', ascending = False)
        financial_association_results_selected_tmp['利潤排名'] = range(1, len(financial_association_results_selected_tmp)+1)

        # 輸出csv
        financial_association_results_selected_tmp.to_csv('02_'+series+ '_區隔_'+ segment_name +'_' + i +'_商品搭配分析推薦表'+ '_' + str(month) +'月' +'.csv', encoding = 'utf-8-sig')


    financial_association_results_selected= financial_association_results_selected.fillna('沒有產品')
    indices = [i for i, s in enumerate(financial_association_results_selected) if '支持度' in s]
    path = financial_association_results_selected.columns[0:indices[0]]

    fig = px.sunburst(financial_association_results_selected, 
                    path= path, values="產品組合總利潤",
                    color="產品組合總利潤", #hover_data=['iso_alpha'],
                    color_continuous_scale='RdBu')

    fig.update_layout(title = '03_'+series + "_區隔_" + segment_name+ "_商品搭配分析"+ '_' + str(month) +'月' )
    # iplot(fig)
    plot(fig, filename='03_'+series + "_區隔_" + segment_name+ "_商品搭配分析"+ '_' + str(month) +'月' +'.html',
                auto_open= False)


    move_file(series, series + "_區隔_" + segment_name+ "_商品搭配分析"+ '_' + str(month) +'月' )

    financial_association_results_selected['month'] = month

    return financial_association_results_selected



def financial_apriori_fun(association_results,client_seg_list_new_group):

    data_tmp_all_list = []
    for i in np.unique(association_results["p0"]):
        print(i )
        
        # 1. 篩選相同產品(p0)與其他產品搭配購買機率
        data = association_results[association_results["p0"]==i]

        # 2. 計算所有含有該產品的「訂單編號」
        orderid = client_seg_list_new_group[client_seg_list_new_group['產品'] == i][['訂單編號']]
        orderid
        client_seg_list_tmp = client_seg_list_new_group.merge(orderid, on='訂單編號', how='right')

        data_tmp_great_list= []
        for product in progressbar.progressbar(range(0, len(data))):
            
            # 抓出每一row的產品搭配組合
            data_tmp = data.iloc[product:product+1]

            # -------抓出一起搭配購買的產品訂單編號-----------
            # 這樣因爲會有不同種的產品搭配所以要用forloop來逐一記錄
            with_prod_col = [i for i in data_tmp.columns if 'p' in i and 'support' not in i and 'p0' not in i   ]

            orderid_tmp_list = []
            prod_tmp_list = []
            for col in with_prod_col:
                if data_tmp[col].iloc[0] is not None:
                    # print(col)
                    prod_tmp_list.append(data_tmp[col].iloc[0])
                    orderid_tmp = client_seg_list_tmp[client_seg_list_tmp['產品'].str.contains(data_tmp[col].iloc[0] + '$')][['訂單編號']]
                    orderid_tmp_list.append(orderid_tmp)

            orderid_tmp = pd.concat(orderid_tmp_list)
            orderid_tmp = orderid_tmp.drop_duplicates(['訂單編號'])
            # prod_tmp_list = '|'.join(prod_tmp_list)

            # 這邊要確認多種產品搭配是否都在一個訂單裡面
            client_seg_list_tmp_tt_list = []
            for num in orderid_tmp['訂單編號']:
                client_seg_list_tmp_tt = client_seg_list_tmp[client_seg_list_tmp['訂單編號']==num]

                good = True
                for i in prod_tmp_list:
                    if len(client_seg_list_tmp_tt[client_seg_list_tmp_tt['產品']==i])==0:
                        good = False
                    # else:
                        

                if good:
                    client_seg_list_tmp_tt_list.append(client_seg_list_tmp_tt )
                            
            client_seg_list_tmp_tt_list = pd.concat(client_seg_list_tmp_tt_list)


            # 抓取主要購買產品與一起搭配購買的產品訂單編號
            # 舉例：買1-11也有一起買1-12的產品訂單編號
            # 目的：計算兩個產品一起搭配購買的利潤
            client_seg_list_tmp_buy_with = client_seg_list_tmp.merge(orderid_tmp, on='訂單編號', how='right')
            client_seg_list_tmp_buy_with_sum = client_seg_list_tmp_tt_list.groupby('產品', as_index = False)['利潤'].sum()
            p_main = client_seg_list_tmp_buy_with_sum[client_seg_list_tmp_buy_with_sum['產品']==data_tmp['p0'].iloc[0]]

            # # 抓出搭配產品的利潤
            p_with_list = []
            for col in with_prod_col:
                if data_tmp[col].iloc[0] is not None:
                    p_with = client_seg_list_tmp_buy_with_sum[client_seg_list_tmp_buy_with_sum['產品']==data_tmp[col].iloc[0]]
                    p_with_list.append(p_with)
            p_with_list= pd.concat(p_with_list)

            p_all_list= pd.concat([p_main, p_with_list])
            


            # -------抓出利潤-----------
            
            # un = client_seg_list_tmp_tt_list['訂單編號'].unique()
            
            # -------抓出單品利潤-----------
            single_prd = pd.DataFrame(client_seg_list_tmp['訂單編號'].value_counts())
            single_prd = single_prd[single_prd['訂單編號'] ==1].reset_index()
            
            if len(single_prd)!= 0 :
                client_seg_list_tmp_single_tmp_list= []
                client_seg_list_tmp_single = client_seg_list_tmp.copy()
                for num in single_prd['index']:
                    client_seg_list_tmp_single_tmp = client_seg_list_tmp_single[client_seg_list_tmp_single['訂單編號']==num]
                    client_seg_list_tmp_single_tmp_list.append(client_seg_list_tmp_single_tmp)

                client_seg_list_tmp_single = pd.concat(client_seg_list_tmp_single_tmp_list)

                client_seg_list_tmp_buy_only = client_seg_list_tmp_single.groupby('產品', as_index = False)['利潤'].sum()
                
                p_main_single = client_seg_list_tmp_buy_only[client_seg_list_tmp_buy_only['產品']==data_tmp['p0'].iloc[0]]
            else:
                p_main_single = pd.DataFrame({'利潤': [0] })

                

            #----------- 比較 --------------

            # 將購物籃分析中篩選出來的產品與其所有搭配產品的總利潤加總，代表本選定產品所貢獻的總利潤
            data_tmp['本產品綜合總利潤'] = client_seg_list_tmp['利潤'].sum()

            # 計算選定的產品與建議產品搭配組合之利潤，代表選定的產品與其搭配的產品利潤，組合利潤越高，則越有價值
            data_tmp['產品組合總利潤'] = p_all_list['利潤'].sum()
            
            # data_tmp['外溢利潤'] = client_seg_list_tmp_buy_with_sum['利潤'].sum()
            # data_tmp['外溢利潤與產品搭配利潤比'] = data_tmp['外溢利潤']/data_tmp['產品搭配總利潤']

            # 「產品組合總利潤」佔「本產品綜合總利潤」百分比，若越多，則該組合越有價值
            data_tmp['產品組合佔綜合總利潤百分比'] = p_all_list['利潤'].sum() / client_seg_list_tmp['利潤'].sum()
            
            # 「不搭配產品（單品）」佔「本產品綜合總利潤」百分比，代表若本百分比高於「產品組合佔綜合總利潤百分比」，則建議不用打該產品組合
            data_tmp['不搭配產品（單品）佔綜合總利潤百分比'] = p_main_single['利潤'].sum() / data_tmp['本產品綜合總利潤'] 
            
            data_tmp_great_list.append(data_tmp)

        data_tmp_great_list = pd.concat(data_tmp_great_list)
        
        data_tmp_all_list.append(data_tmp_great_list)

    result = pd.concat(data_tmp_all_list)

    with_prod_col = [i for i in result.columns if 'p' in i and 'support' not in i and 'p0' not in i   ]


    changecol = [i.replace('p', '產品組合') for i in with_prod_col ]

    for org, ch in zip(with_prod_col, changecol):
        result = result.rename( columns = {org:ch})
    
    result = round(result, 2)

    result = result.rename( columns = {'p0':'購買的產品', 
                            'support':'支持度',
                            'confidence':'產品組合購買機率',
                            'lift':"提升度"})
    
    return result

def apriori_fun(series = '系列1', segment_name = 'mid_mid',min_support=0.001, min_lift=1.000000001,
        client_seg_list='',sales_data=''  ):

    client_seg_list_tmp = client_seg_list[client_seg_list['segmentation']==segment_name]

    client_seg_list_tmp = client_seg_list_tmp.drop_duplicates('去識別化會員編碼')

 
    client_seg_list_tmp = client_seg_list_tmp[['去識別化會員編碼']]
    client_seg_list_tmp

 
    client_seg_list_tmp = sales_data.merge(client_seg_list_tmp, on='去識別化會員編碼', how='right')

 
    record=[]
    # series_data = sales_data[sales_data["系列"] == '系列1']
    order_number = np.unique(client_seg_list_tmp["訂單編號"])
    # try:
    #     os.mkdir(series + "_區隔_" + segment_name+ "_商品搭配分析") # 創建各系列資料夾
    # except:
    #     print(series + "_區隔_" + segment_name+ "_商品搭配分析")
    
    print('資料處理中...')

    for i in order_number:
        cart = sales_data[sales_data["訂單編號"]==i]["產品"].values
        record.append(cart)
        # print( '訂單編號： ', sales_data['訂單編號'].iloc[0])
        # print(cart)

 
    record
    print('機器學習處理中...')
    # 分析
    association_rules = apriori(record, min_support=min_support, min_lift=min_lift) # 建立分析規則
    association_results = pd.DataFrame(association_rules) # 結果資料表

 
    association_results
 
    # 建立各商品組合及 support 的分表
    thebig = association_results["items"].str.len().max()
    separate = pd.DataFrame(association_results["items"].values.tolist(),columns=[ "p"+ str(x) for x in range(thebig)])
    separate["support"] = association_results["support"].values

    # 將 confidence 與 lift 從 association_results["ordered_statistics"] 中取出
    separate["confidence"] = association_results["ordered_statistics"].str[0].str[2]
    separate["lift"]=association_results["ordered_statistics"].str[0].str[3]
    print('完成！')
    return separate, client_seg_list_tmp

 
def move_file(dectect_name, folder_name):
    '''
    dectect_name:
        
    folder_name:
        
    '''    
    # 抓出為【正常模型】的所有檔案名稱
    import os 
    save = []
    for i in os.listdir():
        if dectect_name in i:
            save.append(i)
    
    # save=[i for i in os.listdir() if plot_name2 in i]
    
    # make folder
    ff = [i for i in save if not '.' in i ]
    ff = [i for i in ff if  '（' in i ]
    
    
            
    try:
        os.makedirs(folder_name, exist_ok=True)
        folder_namenew= folder_name
    
    except:
        
        try:
            os.makedirs(folder_name + '（' +str(0)+'）', exist_ok=True)
            folder_namenew= folder_name + '（' +str(0)+'）'
        except: 
            
            for i in range(0, 10):
                iinn = [j for j in ff if folder_name + '（' +str(i)+'）'  in j]
                if len(iinn) == 0:
                    os.makedirs(folder_name + '（' +str(i)+'）', exist_ok=True)
                    folder_namenew =folder_name + '（' +str(i)+'）'
                    break
                
                # break
        
    
    
    # # move files to that created folder
    # import shutil
    # save = [i for i in save if '.' in i ]
    # for m in save:
    #     shutil.move(m, folder_namenew)
    
    import shutil

    # move files to that created folder
    save = [i for i in save if '.' in i]
    for m in save:
        dst_path = os.path.join(folder_namenew, os.path.basename(m))
        try:
            shutil.move(m, dst_path)
        except shutil.Error as e:
            # 如果目標路徑已存在文件，強制覆蓋目標文件
            if "already exists" in str(e):
                shutil.move(m, dst_path, copy_function=shutil.copy2)
            else:
                print(f"Error moving file {m}: {e}")




# def move_file(dectect_name, folder_name):
#     '''
#     dectect_name:
        
#     folder_name:
        
#     '''    
#     # 抓出為【正常模型】的所有檔案名稱
#     import os 
#     save = []
#     for i in os.listdir():
#         if dectect_name in i:
#             save.append(i)
    
#     # save=[i for i in os.listdir() if plot_name2 in i]
    
#     # make folder
#     ff = [i for i in save if not '.' in i ]
#     ff = [i for i in ff if  '（' in i ]
    
    
            
#     try:
#         os.makedirs(folder_name)
#         folder_namenew= folder_name
    
#     except:
        
#         try:
#             os.makedirs(folder_name + '（' +str(0)+'）')
#             folder_namenew= folder_name + '（' +str(0)+'）'
#         except: 
            
#             for i in range(0, 10):
#                 iinn = [j for j in ff if folder_name + '（' +str(i)+'）'  in j]
#                 if len(iinn) == 0:
#                     os.makedirs(folder_name + '（' +str(i)+'）')
#                     folder_namenew =folder_name + '（' +str(i)+'）'
#                     break
                
#                 # break
        
    
    
#     # move files to that created folder
#     import shutil
#     save = [i for i in save if '.' in i ]
#     for m in save:
#         shutil.move(m, folder_namenew)


def list_fun(label, sales_data_tmp,  seg, ip,code, member):
    '''
    label : mid_mid
    '''
        
    seg = seg[seg['segmentation']==label]
    g = seg[[member]].merge(sales_data_tmp, on =member)
    # sales_data_tmp['年']= sales_data_tmp['訂單時間'].apply(lambda x: x.year)
    g = seg.merge(sales_data_tmp[[member, '年']], on =member, how = 'left').drop_duplicates()
    mem = pd.DataFrame(g[member].value_counts()).reset_index()
    mem.columns = [member, '不同年的回購次數']
    g = g.merge(mem,on =member, how = 'left')
    g = g.sort_values(['年','不同年的回購次數'], ascending = False)
    
    # ip= ip.split('_')[1]
    
    seg = g[[member]].merge(sales_data_tmp, on =member)
    
    g = g.rename(columns ={'count': '消費次數（忠誠度）', 
                                 '單價':'消費金額（貢獻度）',
                                 '年':'最新購買年份',
                                 'segmentation': '區隔' })
    g['建議邀請序'] = range(1, (len(g)+1))
    
    g = g[[member, '消費次數（忠誠度）',  '消費金額（貢獻度）',	'不同年的回購次數', '最新購買年份','建議邀請序' ,'區隔']]
    
    # 不同年的回購次數
    g= g.sort_values(['最新購買年份','不同年的回購次數', '消費金額（貢獻度）'], ascending = False)
    g =g.drop_duplicates(member)
    g['建議邀請序'] = range(1, (len(g)+1))
    g['屬性'] = '高回購'
    
    g.to_csv(code + '會員編號清單_高回購_【'+label+'】_' + ip+'.csv', encoding = 'UTF-8-sig')
    
    
    # 消費金額（貢獻度）
    g= g.sort_values(['消費金額（貢獻度）'], ascending = False)
    g['建議邀請序'] = range(1, (len(g)+1))
    g['屬性'] = '高消費'
    g.to_csv(code + '會員編號清單_消費金額_【'+label+'】_' + ip+'.csv', encoding = 'UTF-8-sig')
    
    seg.to_csv(code + '詳細會員原始資料清單_【' +label+'】_'+ ip+'.csv', encoding = 'UTF-8-sig')
    return g,seg




def high_potential(product_2019_yes_over,product_profit_years,product = 'product',profit = '利潤'):
        
    potential_prod_df = pd.DataFrame()
    for i in product_2019_yes_over[product]:
        tmp =  product_profit_years[product_profit_years[product] == i]
        tmp_base = tmp.iloc[0:len(tmp)-1].reset_index()
        tmp_base['與上一年差異'] = tmp[profit].diff().dropna().reset_index()[profit]
        tmp_base['成長率'] = tmp_base['與上一年差異'] /tmp_base[profit]
        
        prod = pd.DataFrame({
    
            product :[i],
    
            '平均成長率':[ tmp_base['成長率'].mean() ]
        })
        
        potential_prod_df = pd.concat([prod, potential_prod_df], axis = 0)
    
    potential_prod_df = potential_prod_df.sort_values('平均成長率', ascending= False )
    return potential_prod_df
from sklearn.cluster import AgglomerativeClustering

def loy_con_bind(contribution,loyalty, select_product,member ):
    
    seg = loyalty.merge(contribution, on = member)
    seg['segmentation'] = seg['loyalty_precent_rank_cluster'] + '_' + seg['contribution_precent_rank_cluster']
    print(seg['segmentation'].value_counts())
    
    # 使用交叉分析製作顧客區隔
    seg_matrix = pd.crosstab(seg['contribution_precent_rank_cluster'], 
                             seg['loyalty_precent_rank_cluster'] )
    
    seg_matrix = seg_matrix[['low', 'mid','high']]
    seg_matrix  = seg_matrix.reset_index()
    
    
    seg_matrix = pd.concat([
        seg_matrix[seg_matrix['contribution_precent_rank_cluster']=='high'],
        seg_matrix[seg_matrix['contribution_precent_rank_cluster']=='mid'],
        seg_matrix[seg_matrix['contribution_precent_rank_cluster']=='low'],
    ], axis = 0)
    
    print(seg_matrix)
    # seg_matrix.to_csv(select_product+'_顧客區隔分析.csv', encoding = 'utf-8-sig')
    seg_matrix.to_csv('01會員資料區隔_'+ select_product+'.csv', encoding = 'UTF-8-sig')

    return seg, seg_matrix
# 建立function
def customer_loyalty(data, member = '會員' ,
                     product = 'product',
                     select_product = '產品1',year= None):
    '''
    Parameters
    ----------
    data : dataFrame
        要放入的交易資料.
        
    member : 字串, optional
        data裡面的「會員」欄位名稱.
        
    select_product : TYPE, optional
        要選擇做顧客分群的產品. The default is '產品1'.

    Returns
    -------
    loyalty : TYPE
        忠誠度（消費次數）.

    '''
    
    # 我們要根據product來準備進行分羣
    sales_data_tmp = data[data[product] ==select_product ]
    
    if year:
        sales_data_tmp = sales_data_tmp[sales_data_tmp['年']==year]
    
    
    # ----忠誠度：消費次數----
    
    # 忠誠度：消費次數計算
    '''
    因爲一筆資料就是消費一次，所以先創建一個過度參數
    來用groupby計算每一個消費者總體消費次數
    '''
    sales_data_tmp['count'] = 1 
    loyalty = sales_data_tmp.groupby([member])[['count']].sum()
    
    # 分羣演算：階層分群(hierarchical clustering)
    from sklearn.cluster import AgglomerativeClustering
    
    # 以階層式分羣法分成3層
    
    # 定義模型
    model = AgglomerativeClustering(n_clusters=3, 
                                    affinity='euclidean', 
                                    linkage='ward')
    
    # 訓練分群模型
    model.fit(loyalty[['count']])
    
    # 抓出loyalty的標籤
    labels = model.labels_
    loyalty['loyalty_cluster'] =labels 
    
    # 將會員從index變成欄位
    loyalty = loyalty.reset_index()
    
    
    
    # cluster標籤高、中、低忠誠度
    
    # 找出每一個cluster標籤的消費次數程度
    cluster_level = loyalty.groupby('loyalty_cluster', as_index = False)['count'].mean()
    cluster_level = cluster_level.sort_values('count', ascending = False)
    
    # 找出最小與最大的cluster
    max_cluster = cluster_level['loyalty_cluster'].iloc[0]
    min_cluster = cluster_level['loyalty_cluster'].iloc[2]
    loyalty['loyalty_precent_rank_cluster'] = np.where(loyalty['loyalty_cluster']==max_cluster, 'high',
                                                np.where(loyalty['loyalty_cluster']==min_cluster, 'low','mid'))
    
    # 查看cluster標籤高、中、低忠誠度次數
    print(loyalty['loyalty_precent_rank_cluster'].value_counts())
    
    return loyalty



# 建立function
def customer_contribution(data, member = '會員' , 
                          price = '單價',
                          product ='product',
                          select_product = '產品1',year= None):
    '''
    Parameters
    ----------
    data : dataFrame
        要放入的交易資料.
        
    member : 字串, optional
        data裡面的「會員」欄位名稱.
        
    price : 字串, optional
        data裡面的「單價」欄位名稱.
        
    select_product : TYPE, optional
        要選擇做顧客分群的產品. The default is '產品1'.

    Returns
    -------
    contribution : TYPE
        忠誠度（消費次數）.

    '''
        
    # 我們要根據product來準備進行分羣
    sales_data_tmp = data[data[product] ==select_product ]
    
    # 貢獻度：消費金額計算
    contribution = sales_data_tmp.groupby([member])[[price]].sum()
    
    # 分羣演算：階層分群(hierarchical clustering)
    
    # 以階層式分羣法分成3層
        
    # 定義模型
    model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', 
                                    linkage='ward')
    
    # 訓練分群模型
    model.fit(contribution[ [price] ])
    
    # 抓出contribution的標籤
    labels = model.labels_    
    contribution['contribution_cluster'] =labels 
    
    # 將會員從index變成欄位
    contribution = contribution.reset_index()
    
    
    
    # cluster標籤高、中、低忠誠度
    
    # 找出每一個cluster標籤的消費次數程度
    cluster_level = contribution.groupby('contribution_cluster', as_index = False)[price].mean()
    cluster_level = cluster_level.sort_values(price, ascending = False)
    
    # 找出最小與最大的cluster
    max_cluster = cluster_level['contribution_cluster'].iloc[0]
    min_cluster = cluster_level['contribution_cluster'].iloc[2]
    contribution['contribution_precent_rank_cluster'] = np.where(contribution['contribution_cluster']==max_cluster, 'high',
                                                np.where(contribution['contribution_cluster']==min_cluster, 'low','mid'))
    
    # 查看cluster標籤高、中、低貢獻度次數
    print(contribution['contribution_precent_rank_cluster'].value_counts())

    return contribution

def product_contribution(data, year ,product, profit,profit_percent, time):
    '''

    Parameters
    ----------
    data : dataFrame
        要放入的交易資料.
        
    year : 日期形式
        舉例：'2019-1-1'.
        
    product : 字串
        data裡面的「產品」欄位名稱.
        
    profit : 字串
        data裡面的「利潤」欄位名稱.
        
    profit_percent : int, optional
        篩選貢獻多少「%」利潤優先分析的產品. The default is 0.8.

    Returns
    -------
    建議優先分析的產品.

    '''
        
    sales_data_2019 = data[ (data[time] > parser.parse(year)) ]

    # 產品/貢獻比例：計算每一個產品的利潤總和
    product_profit =  sales_data_2019.groupby(product, as_index = False )[profit].sum()
    product_profit = product_profit.sort_values(profit, ascending = False  )
    
    # 產品的貢獻比
    product_profit['利潤佔比'] = product_profit[profit] / product_profit[profit].sum()
    product_profit['累計利潤佔比'] = product_profit['利潤佔比'].cumsum()
    
    # 產品比
    product_profit['累計產品次數'] = range(1,len(product_profit)+1)
    product_profit['累計產品佔比'] = product_profit['累計產品次數'] / len(product_profit)
    
    # 四捨五入
    product_profit['累計產品佔比'], product_profit['累計利潤佔比'],product_profit['利潤佔比'] = round(product_profit['累計產品佔比'], 2), round(product_profit['累計利潤佔比'], 2), round(product_profit['利潤佔比'], 2)
    
    # 輸出篩選產品貢獻度（利潤）資料
    product_profit.to_csv('0_產品貢獻度（利潤）表.csv', encoding = 'utf-8-sig')
    
    # 產品/貢獻度比例圖
    import plotly.express as px
    
    fig = px.bar(product_profit, x=product, y='利潤佔比',
                 hover_data=['累計利潤佔比', '累計產品佔比'], color=profit,
                 text = '累計利潤佔比',
                 height= 400,
                 title='產品/貢獻度比例圖'
                 )
    fig.update_traces( textposition='outside')
    plot(fig, filename= '0_產品貢獻度比例圖.html')
    
    
    # 篩選貢獻80%利潤的產品
    
    product_profit_selected = product_profit[product_profit['累計利潤佔比']<=profit_percent]
    
    import plotly.express as px
    
    fig = px.bar(product_profit_selected, x=product, y='利潤佔比',
                 hover_data=['累計利潤佔比', '累計產品佔比'], color=profit,
                 text = '累計利潤佔比',
                 height= 600,
                 title='貢獻' + str(profit_percent*100) + '%的' + '產品貢獻度比例圖'
                 )
    fig.update_traces( textposition='outside')
    plot(fig, filename= '1_' + '貢獻' + str(profit_percent*100) + '%的' + '產品貢獻度比例圖'+'.html')
    
        
    # 建議優先分析的產品
    product_profit_selected.to_csv( '1_' + '貢獻' + str(profit_percent*100) + '%的' + '產品貢獻度比例表'+'.csv', encoding = 'utf-8-sig')
    
    # 歸納資料
    from analysis import  move_file
    move_file(dectect_name = '產品貢獻', folder_name = '0_產品貢獻度')

    
    # 建議優先分析的產品
    return product_profit_selected


def product_potential(data, product ='product', profit='利潤',year=2019 ,top=10):
    '''

    Parameters
    ----------
    data : dataFrame
        要放入的交易資料.
        
    year : 日期形式
        舉例：'2019-1-1'.
        
    product : 字串
        data裡面的「產品」欄位名稱.
        
    top : TYPE, optional
        選擇top多少的潛力商品. The default is 10.

    Returns
    -------
    top的潛力商品.

    '''
        
    # 找出每一個產品每一年的利潤
    data['年'] = data['訂單時間'].dt.year
    product_profit_years = data.groupby(['年', product], as_index = False )[profit].sum()
    
    # 篩選2019年有的產品
    product_2019_yes = product_profit_years[product_profit_years['年'] ==year]
    
    # 篩選所有product次數
    product_profit_years[product].value_counts()
    each_product = pd.DataFrame(product_profit_years[product].value_counts())
    each_product = each_product.reset_index()
    each_product = each_product.rename(columns ={product:'次數' ,
                                                'index':product})
    
    # 篩選2019年有的產品且已經存活超過1年以上
    product_2019_yes_over = each_product.merge(product_2019_yes, on = product, how= 'right')
    product_2019_yes_over = product_2019_yes_over[product_2019_yes_over['次數'] > 1 ]
    
    # 找出連續年度高潛力產品
    # from analysis import high_potential
    potential_prod_df = high_potential(product_2019_yes_over,product_profit_years,product, profit)

    
    # 輸出篩選產品貢獻度（利潤）資料
    potential_prod_df.to_csv('0_連續年度高潛力產品.csv', encoding = 'utf-8-sig')
    
    # 連續年度高潛力產品 - 平均成長率圖
    import plotly.express as px
    
    fig = px.bar(potential_prod_df, x=product, y='平均成長率',
                 hover_data=[product, '平均成長率'], 
                 color='平均成長率',
                 height= 400,
                 title='連續年度高潛力產品 - 平均成長率圖'
                 )
    plot(fig, filename= '0_連續年度高潛力產品 - 平均成長率圖.html')
    
    
    # 連續年度高潛力產品top10 - 平均成長率圖 
    
    import plotly.express as px
    
    fig = px.bar(potential_prod_df[0:top], x=product, y='平均成長率',
                 hover_data=[product, '平均成長率'], 
                 color='平均成長率',
                 height= 600,
                 title='1_連續年度高潛力產品' + str(top) + ' - 平均成長率圖'
                 )
    fig.update_traces( textposition='outside')
    plot(fig, filename= '1_連續年度高潛力產品' + str(top) + ' - 平均成長率圖.html')
    
    potential_prod_df[0:top].to_csv('1_連續年度高潛力產品' + str(top) + ' - 平均成長率表.csv', encoding = 'utf-8-sig')
    
    # 歸納資料
    from analysis import  move_file
    move_file(dectect_name = '連續年度', folder_name = '1_連續年度高潛力產品')

    
    return potential_prod_df[0:top]



def product_client_list(data,
                        seg,
                        product = 'product', select_product ='產品1',
                        member= '會員',
                        price = '單價'
                        ):
        
    # seg_mid = seg[seg['segmentation']=='mid_mid'] 
    
    sales_data_tmp = data[data[product] ==select_product ]
    
    # 使用merge製作完整顧客清單
    client_seg_list = seg[[member,'segmentation']].merge(sales_data_tmp, on =member)
    
    
    # seg, seg_matrix =  seg_matrix_fun(g) #seg, sematrix
    seg_deep_summary = seg.groupby('segmentation', as_index = False)['count', price].mean()
    seg_deep_summary = round(seg_deep_summary, 1 )
    seg_deep_summary = seg_deep_summary.rename(columns ={'count': '消費次數（忠誠度）', 
                                    price:'消費金額（貢獻度）'})
    
    
    seg_deep_summary.to_csv('02會員區隔指標總覽_'+ select_product+'.csv', encoding = 'UTF-8-sig')
    # seg_matrix.to_csv('01會員資料區隔_'+ select_product+'.csv', encoding = 'UTF-8-sig')
    
    # 高消費與高回購
    from analysis import list_fun
    high_high_list, high_high_seg = list_fun(label = 'high_high', sales_data_tmp = sales_data_tmp,  seg = seg, ip=select_product,code= '03',member=member)
    mid_mid_list, mid_mid_seg = list_fun(label= 'mid_mid', sales_data_tmp = sales_data_tmp,  seg = seg, ip=select_product,code= '04',member=member)
    low_low_list, low_low_seg = list_fun(label= 'low_low', sales_data_tmp = sales_data_tmp,  seg = seg, ip=select_product,code= '05',member=member)
    
    move_file(dectect_name = select_product, folder_name = '2_會員顧客清單')
    return client_seg_list


def ad_pattern(client_seg_list, select_product ='產品1', 
               select_segment = 'high_high',
               top = 10
               ):
    
    client_seg_list_pattern = client_seg_list[client_seg_list['segmentation']==select_segment]
    client_seg_list_pattern['count'] = 1 
    
    # --------廣告---------
    ad_profit = client_seg_list_pattern.groupby('廣告代號all', as_index = False)[[profit, 'count']].sum()
    
    # 每次廣告利潤
    ad_profit['每次廣告利潤'] = round(ad_profit[profit] /ad_profit['count'],2 )
    
    # 佔總利潤比例
    ad_profit['佔總利潤比例'] = ad_profit[profit] /ad_profit[profit].sum()
    
    # 廣告總利潤比較圖
    ad_profit = ad_profit.sort_values(profit, ascending = False)
    
    # ad_profit
    ad_profit.to_csv('廣告利潤表_'+ select_product+ '_'+ select_segment+'.csv', encoding = 'utf-8-sig')
    
    # 加註
    ad_profit['product'] = select_product
    ad_profit['區隔'] = select_segment
    
    fig = px.bar(ad_profit, x='廣告代號all', y=profit,
                 color=profit,
                 title='廣告總利潤比較圖'
                 )
    plot(fig, filename= '0_廣告總利潤比較圖_'+ select_product+ '_'+ select_segment+'.html')
    
    
    
    # 廣告總利潤比較圖top10
    ad_profit = ad_profit.sort_values(profit, ascending = False)
    fig = px.bar(ad_profit[0:top], x='廣告代號all', y=profit,
                 color=profit,
                 title='1_廣告總利潤比較圖_top_'+str(top) +'_'+ select_product+ '_'+ select_segment
                 )
    plot(fig, filename= '1_廣告總利潤比較圖_top_'+str(top) +'_'+ select_product+ '_'+ select_segment+'.html')
    
    
    # 每次廣告利潤比較圖
    ad_profit = ad_profit.sort_values('每次廣告利潤', ascending = False)
    
    # 將次數爲 < PR25 的去除
    ad_profit1 = ad_profit[ad_profit['count'] >= ad_profit['count'].describe()['25%'] ]
    ad_profit1 = ad_profit1.sort_values('每次廣告利潤', ascending = False)
    
    fig = px.bar(ad_profit1, x='廣告代號all', y='每次廣告利潤',
                 color='每次廣告利潤',
                 hover_data=['count'],
                 title='每次廣告利潤比較圖_'+ select_product+ '_'+ select_segment,
                 text = '每次廣告利潤'
                 )
    fig.update_traces(texttemplate='%{text:.2s}',  textposition='outside')
    plot(fig, filename= '2_每次廣告利潤比較圖_'+ select_product+ '_'+ select_segment+'.html')
    
    
    # 最適廣告次數與利潤圖
    # 累計除法 --》 廣告打幾次、畫出曲線
    for i in ad_profit[0:top]['廣告代號all']:
            
        curve = client_seg_list_pattern[client_seg_list_pattern['廣告代號all'] ==i]
        curve = curve.sort_values('訂單時間')
        curve = curve[[profit,'count']].cumsum()
        curve['每次廣告利潤'] = round(curve[profit] /curve['count'],2 )
        
        curve_max = curve[curve['每次廣告利潤'] == curve['每次廣告利潤'].max()]
        fig = px.line(curve, x="count", y="每次廣告利潤")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=curve['count'], y=curve["每次廣告利潤"],
                        mode='lines',
                        line=dict(color='royalblue')
                        ))
        
        fig.add_trace(go.Scatter(x=curve_max['count'], y=curve_max["每次廣告利潤"],
                        mode="markers+text",
                        text =  str(curve_max['count'].iloc[0])+'次'+'；'+ '每次廣告利潤 : '+ str(curve_max["每次廣告利潤"].iloc[0]),
                        line=dict(color='red'),
                        textposition="top center"
                        ))
        fig.update_layout(title_text='3_最適廣告次數與利潤圖_'+ i + '_'+ select_product+ '_'+ select_segment)
        
        plot(fig, filename= '3_最適廣告次數與利潤圖_'+ i + '_'+ select_product+ '_'+ select_segment+'.html',auto_open=False)


    # 歸納資料
    move_file(dectect_name =select_product, folder_name = select_product+'_'+select_segment+ '_廣告利潤綜整')



def normal_pattern(client_seg_list , 
                   select_product ='產品1', 
               select_segment = 'mid_mid',
               pattern = '尺寸',
               top = 10):
        
    client_seg_list_pattern = client_seg_list[client_seg_list['segmentation']==select_segment]
    client_seg_list_pattern['count'] = 1 
    
    size_profit = client_seg_list_pattern.groupby(pattern, as_index = False)[[profit, 'count']].sum()
    
    
    # 廣告總利潤比較圖
    size_profit = size_profit.sort_values(profit, ascending = False)
    
    size_profit.to_csv(pattern+'利潤表_'+ select_product+ '_'+ select_segment+'.csv', encoding = 'utf-8-sig')
    
    
    # 加註
    size_profit['product'] = select_product
    size_profit['區隔'] = select_segment
    
    fig = px.bar(size_profit, x=pattern, y=profit,
                 color=profit,
                 title=pattern+'總利潤比較圖'
                 )
    plot(fig, filename= '0_'+pattern+'總利潤比較圖_'+ select_product+ '_'+ select_segment+'.html')
    
    
    
    # 廣告總利潤比較圖top10
    size_profit = size_profit.sort_values(profit, ascending = False)
    top =10
    fig = px.bar(size_profit[0:top], x=pattern, y=profit,
                 color=profit,
                 title= '1_'+pattern+'總利潤比較圖_top_'+str(top) +'_'+ select_product+ '_'+ select_segment
                 )
    plot(fig, filename= '1_'+pattern+'總利潤比較圖_top_'+str(top) +'_'+ select_product+ '_'+ select_segment+'.html')
    
    # 歸納資料
    move_file(dectect_name =pattern, 
              folder_name = select_product+'_'+select_segment+ '_'+  pattern+'利潤綜整')

