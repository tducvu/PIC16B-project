import streamlit as st

def local_scss(file_name):
    with open(file_name) as f:
         st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


local_scss("style.scss")


