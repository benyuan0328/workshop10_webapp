import pandas as pd 
import streamlit as st


def generate_download_button(df, filename):
    csv_data = df.to_csv().encode('utf-8')
    st.download_button(label="下載csv檔",
                           data=csv_data,
                           file_name=f"{filename}.csv")
