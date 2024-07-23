import streamlit as st
from streamlit_lottie import st_lottie
import json
def feedbackRating():

    col1,col2 =st.columns(2)

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
                <h2>Testimonials </h2>
            </div>
        """, unsafe_allow_html=True)
    path = "Animation_rating.json"
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
    with st.container():
        st.markdown("### Ranked top 10% of all Upwork talent")
        st.write("#### Achieved a Top Rated badge with a 100% Job Success Rate on Upwork.")
        st.write("To verify the following information related to Upwork, please use this [link](https://www.upwork.com/freelancers/~010f3758a004ea64dd?viewMod).")
        st.image('up_10R.png', width=700)

        with st.container():
            st.write("### Testimonials From Clients")
            image_paths = ["up1.png", "up2.png", "up3.png", "up4.png", "up.png"]
            for i in range(5):
                if i%2 ==0:
                    col1,col2,col3 = st.columns([1,1,4])
                    col3.image(image_paths[i],width=700)
                else:
                    col1, col2, col3 = st.columns([4,1,1])
                    col1.image(image_paths[i], width=700)

            # st.markdown("""
            #     <style>
            #     .right-align {
            #         display: flex;
            #         flex-direction: column;
            #         align-items: flex-end;
            #     }
            #     .right-align img {
            #         margin-bottom: 20px; /* Adds space between images */
            #     }
            #     </style>
            #     <div class="right-align">
            #         <img src="up1.png" width="700">
            #         <img src="up2.png" width="700">
            #         <img src="up3.png" width="700">
            #         <img src="up4.png" width="700">
            #         <img src="up.png" width="700">
            #     </div>
            # """, unsafe_allow_html=True)

    # Ensure images are in the correct path
    # image_paths = ["up1.png", "up2.png", "up3.png", "up4.png", "up.png"]
    #
    # for path in image_paths:
    #     st.image(path, width=700)

    # with st.container():
    #     st.write("### Testimonials From Clients")
    #     st.markdown("""
    #                <style>
    #                .right-align img {
    #                    display: block;
    #                    margin-left: auto;
    #                    margin-right: 0;
    #                }
    #                </style>
    #                <div class="right-align">
    #                    <img src="up1.png" width="700">
    #                    <img src="up2.png" width="700">
    #                    <img src="up3.png" width="700">
    #                    <img src="up4.png" width="700">
    #                    <img src="up.png" width="700">
    #                </div>
    #            """, unsafe_allow_html=True)
    #
    #
    #
    #     st.image('up1.png',width=700)
        # st.image('up2.png', width=700)
        # st.image('up3.png', width=700)
        # st.image('up4.png', width=700)
        # st.image('up.png', width=700)






