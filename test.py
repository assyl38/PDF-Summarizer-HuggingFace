import streamlit as st
import os #os module for environment variable management
from utils import *

def main():
    #set page configuration
    st.set_page_config(page_title="PDF Summarizer")

    st.title("PDF Summarizer APP")
    st.write("Summarize your pdf file in secs ...")
    st.divider() #inserting a divider for a better layout 

    #create a file uploader widget 
    pdf = st.file_uploader('Upload your PDF file', type='pdf')

    #create a button for users to submit their PDF for summarization
    submit = st.button("Generate Summary")

    if submit:
         
            response = summarizer(pdf)
            st.subheader('Summary of the PDF file') 
            st.write(response)

#Python execution here
if __name__ == '__main__':
    main() #call main fuction to start the Streamlit app