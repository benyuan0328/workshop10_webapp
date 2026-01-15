import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from basic_analysis_V2 import product_contribution, analyze_potential_growth_data
import os
from run_product_contribution import run_product_contribution
from run_potential_growth import run_potential_growth
from run_ad_campaign import run_ad_effect_analysis
import streamlit.components.v1 as stc 

html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">商品搭售與廣告效益分析系統</h1>
		</div>
		"""
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
def main():
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
	# st.title("ML Web App with Streamlit")
    stc.html(html_temp)
    st.sidebar.image('docs/logo.png', width=250)
    st.sidebar.write('') # Line break
    st.sidebar.header('Navigation')

    menu = ['【金牛系列】貢獻度分析',"「潛在」系列成長比較分析","廣告效益分析"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == '【金牛系列】貢獻度分析':
        run_product_contribution()
        
    elif choice =='「潛在」系列成長比較分析':
        run_potential_growth()
        
    elif choice == "廣告效益分析":
        run_ad_effect_analysis()
    

if __name__ == '__main__':
    main()
