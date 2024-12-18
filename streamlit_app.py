import streamlit as st
from streamlit.logger import get_logger


LOGGER = get_logger(__name__)

def run():
    st.set_page_config(
        page_title="ML Model Generator and Implementation",
        page_icon="🤖"
    )
    # Main page content
    st.write("# ML Model Generator 🤖🤖")
    st.write("by Caritos_Dimanarig_Torallo")


    st.sidebar.success("Navigate above!")
    


if __name__ == "__main__":
    run()
