import streamlit as st
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import json

def contact():
    col1, col2 = st.columns(2)

    # col1.markdown("## Experience")
    with col1:
        st.markdown("""
                    <style>
                    .centered {
                        display: flex;
                        align-items: center;
                        height: 100%;
                        margin-top: 200px; /* Adjust this value as needed */
                    }
                    </style>
                    <div class="centered">
                        <h2>📨 Contact Form </h2>
                    </div>
                """, unsafe_allow_html=True)
    path = "Animation_contact.json"
    with open(path, "r") as file:
        url = json.load(file)
    with col2:
        st_lottie(url,
                  reverse=True,
                  height=400,
                  width=400,
                  speed=1,
                  loop=True,
                  quality='high',
                  )

    components.html(
        f"""
         <iframe src="https://docs.google.com/forms/d/e/1FAIpQLScLaMWyScjbqoo6I5w5MtoQwfSU-Izghn1y_jsTP-yuf5zZOA/viewform?embedded=true" width="640" height="741" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>
        """,
        height=1800,
    )


