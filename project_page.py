import streamlit as st
import json
from streamlit_lottie import st_lottie

def projects():
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
                    <h2>Projects </h2>
                </div>
            """, unsafe_allow_html=True)
    path = "Animation_girl.json"
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
        col1,col2 = st.columns(2)
        with col1:
            with st.container(border=True):


                # Displaying the title of the project
                st.title("Insightful Data Explorer")

                # Displaying the description
                st.markdown("""
                **Description:**
                The Insightful Data Explorer is a Streamlit-based application designed for comprehensive data analysis and machine learning tasks. Key features include:
        
                - **Data Handling:** Upload, edit, and preprocess CSV or Excel files.
                - **Chat with Data:** Interactive data exploration using Google Gemini-1.5-Flash-Latest.
                - **Visualization:** Custom and automated chart generation.
                - **Feature Engineering:** Transform and create new features, handle missing values and outliers.
                - **AutoML:** Automated model selection, training, and optimization for various machine learning tasks with PyCaret.
                - **Data Profiling:** Detailed data profiling using YData Profiling.
                """)

                # Displaying the tools used
                st.markdown("""
                **Tools Used:**
                
                 **Python** ,
                 **Pandas**,
                **Streamlit** ,
                **Google Gemini**, 
                **PyCaret**, 
                **PygWalker**, 
                **AutoViz**, 
                **YData Profiling**, 
                """)
                st.markdown(""" """)


                c1,c2,c3,c4 = st.columns(4)
                c1.markdown("""**[Link to app](https://insightful-data-explorer-001.streamlit.app)**  """)
                c2.markdown("""**[GitHub](https://github.com/archanags001/Insightful-Data-Explorer)**""")
                c3.markdown("""**[LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7220172770226102272/)** """)
                c4.markdown("""**[X](https://x.com/streamlit/status/1814406829075542029)**""")
                rc1,rc2 = st.columns(2)
                rc1.markdown("""**[Streamlit community](https://buff.ly/3WqhYiB)**""")
                rc2.markdown("""**[YouTube](https://www.youtube.com/watch?v=dwlE4p2uF6k)**""")


        with col2:
            with st.container(border=True):
                st.markdown(""" """)
                
                # Displaying the title of the project
                st.title("InsightBot")
                st.markdown(""" """)
                st.markdown(""" """)


                # Displaying the description
                st.markdown("""
                **Description:**
                InsightBot is a tool that allows users to chat with their data using Google Gemini-1.5-Flash-Latest. It offers features such as:

                - **Data Interaction:** Chat with your data for interactive analysis.
                - **Sample Datasets:** Provides sample datasets for users to explore if they don't have their own data.
                """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)


                # Displaying the tools used
                st.markdown("""
                **Tools Used:**

                **Python**, 
                **Streamlit**,  **Google Gemini**, **Pandas**
                
                """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)


                c1, c2 = st.columns(2)
                c1.markdown("""**[Link to app](https://chat-with-data-gemini.streamlit.app)**  """)
                c2.markdown("""**[GitHub](https://github.com/archanags001/InsightBot)**""")
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                
                
                


        with col1:
            with st.container(border=True):
                # Displaying the title of the project
                st.title("California Housing Price Prediction")

                # Displaying the description
                st.markdown("""
                **Description:**
                The California Housing Price Prediction project aims to predict housing prices in California using machine learning. It includes:

                - **Data Preprocessing:** Cleaning and preparing the dataset.
                - **Exploratory Data Analysis (EDA):** Visualizing data patterns.
                - **Feature Engineering:** Creating new features to enhance model performance.
                - **Model Training and Evaluation:** Implementing and assessing various regression algorithms, including Linear Regression, Decision Tree Regression, and Random Forest Regression.
                - **Hyperparameter Tuning:** Optimizing model parameters for accuracy.
                """)
                st.markdown(""" """)



                # Displaying the tools used
                st.markdown("""
                **Tools Used:**

                **Python**, **Pandas**,  **NumPy**, **Matplotlib**,  **Seaborn**, **Scikit-learn**
                """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                

                # Adding the GitHub link
                st.markdown("""**[GitHub](https://github.com/archanags001/ml_projects/blob/main/California_Housing_Price_Prediction.ipynb)**""")

        with col2:
            with st.container(border=True):
                # Displaying the title of the project
                st.title("Tensor Creation and Neural Network Performance")

                # Displaying the description
                st.markdown("""
                **Description:**
                This project involves four parts:

                - **Tensor Creation:** Develop a Python class to construct a tensor from given data and shape inputs using nested lists, managing excess data or padding as necessary.
                - **Neural Network:** Build a fully-connected neural network to classify MNIST digits.
                - **Dense Layer Class:** Implement a dense layer class with a forward() method to process input, weight, and bias tensors for forward propagation.
                - **Performance Comparison:** Compare the performance of the neural network from Part 2 with the dense layer implementation from Part 3.
                """)
                st.markdown(""" """)


                # Displaying the tools used
                st.markdown("""
                **Tools Used:**

                **Python** , **NumPy**, **TensorFlow**, **Keras**,  **Matplotlib**
                """)
                st.markdown(""" """)


                # Adding the GitHub link
                st.markdown("""**[GitHub](https://github.com/archanags001/coding_challenge/blob/main/Coding_challenge_.ipynb)**""")

        with col1:
            with st.container(border=True):
                st.markdown(""" """)

                # Displaying the title of the project
                st.title("Multiple LSTMs")
                st.markdown(""" """)
                



                # Displaying the description
                st.markdown("""
                **Description:**
                The "Multiple LSTMs" project focuses on building and comparing multiple Long Short-Term Memory (LSTM) models for time series forecasting. The project involves:

                - **Data Preparation:** Loading and preprocessing time series data.
                - **LSTM Model Implementation:** Creating and training multiple LSTM models.
                - **Model Evaluation:** Comparing the performance of the LSTM models.
                - **Visualization:** Plotting results to visualize model performance.
                """)
                st.markdown(""" """)
                st.markdown(""" """)
                


                # Displaying the tools used
                st.markdown("""
                **Tools Used:**

                **Python**,
                **TensorFlow**, **Keras**, 
                **NumPy**, 
                **Matplotlib**,
                """)
                
                st.markdown(""" """)
                st.markdown(""" """)




                # Adding the GitHub link
                st.markdown("""**[GitHub](https://github.com/archanags001/coding_challenge/blob/main/Multiple_LSTMs.ipynb)**""")
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)



        with col2:
            with st.container(border=True):

                # Displaying the title of the project
                st.title("TensorFlow Projects")

                # Displaying the description
                st.markdown("""
                **Description:**
                The git repository contains various TensorFlow projects and notebooks, each addressing different machine learning tasks. Highlights include:


                - **Callbacks_TensorFlow_MNIST:** Demonstrates using callbacks to improve MNIST digit classification.
                - **Convolution_NN_mnist:** Implements a convolutional neural network for MNIST classification.
                - **Happy_or_sad:** A model to classify images as happy or sad.
                - **Improve_MNIST_with_Convolutions:** Enhances MNIST classification using convolutional layers.
                - **Sign_Language_MNIST:** Classifies sign language digits using a neural network.
                - **Training_Validation_with_ImageDataGenerator:** Explores data augmentation techniques.
                - **Multiclass_Classifier:** Implements a multiclass classification model.
                """)

                # Displaying the tools used
                st.markdown("""
                **Tools Used:**

                **Python**, 
                **TensorFlow**, **Keras**, 
                **NumPy**, 
                **Matplotlib**,
                """)
                st.markdown(""" """)


                # Adding the GitHub link
                st.markdown("""**[GitHub](https://github.com/archanags001/tensorflow)**""")

        with col1:
            with st.container(border=True):
                # Displaying the title of the project
                st.title("Recommendation System Using Pearson Correlation and Cosine Similarity")

                # Displaying the description
                st.markdown("""
                **Description:**
                This project implements a recommendation system using two different similarity metrics: Pearson Correlation and Cosine Similarity. The key tasks include:

                - **Data Preparation:** Loading and preprocessing the dataset.
                - **Pearson Correlation:** Calculating similarity between users/items using Pearson correlation.
                - **Cosine Similarity:** Calculating similarity between users/items using cosine similarity.
                - **Recommendation Generation:** Generating recommendations based on the computed similarities.
                """)

                # Displaying the tools used
                st.markdown("""
                **Tools Used:**

                **Python**,
                **Pandas**,  **NumPy**, 
                **Matplotlib**,
                """)
                st.markdown(""" """)


                # Adding the GitHub link
                st.markdown("""**[GitHub](https://github.com/archanags001/coding_challenge/blob/main/recommendation_Pearson_correlation_and_Cosine_similarity__.ipynb)**""")
                st.markdown(""" """)
        with col2:
            with st.container(border=True):
                # Displaying the title of the project
                st.title("Portfolio Explorer ")

                # Displaying the description
                st.markdown("""
                **Description:**
                The Portfolio Explorer is a Streamlit-based application designed to present a comprehensive and interactive personal portfolio. Key features include:

                - **Intro Page:** A dynamic introduction offering a professional overview.
                - **Resume Page:** A viewable and downloadable resume for quick access to detailed professional information.
                - **Experience Page:** An organized display of work experience, skills, and accomplishments.
                - **Projects Page:** A showcase of notable projects, including descriptions, technologies used, and links to repositories.
                - **Testimonial Page:** A collection of feedback and testimonials from clients and colleagues, highlighting accomplishments and collaborations.
                - **Contact Page:** An integrated contact form for easy communication and inquiries.
                """)

                # Displaying the tools used
                st.markdown("""
                **Tools Used:**

                **Python** , **Streamlit**
                """)
                st.markdown(""" """)

                c1, c2, c3, c4, c5 = st.columns(5)


                # Adding the GitHub link
                c1.markdown("""**[Link to app](https://portfolio-archana.streamlit.app)**  """)
                c2.markdown("""**[GitHub](https://github.com/archanags001/Portfolio/tree/main)**""")

        with col1:
            with st.container(border=True):
                st.markdown(""" """)
                st.markdown(""" """)
                # Displaying the title of the project
                st.title("Object Detection with YOLO")
                st.markdown(""" """)
                st.markdown(""" """)

                # Displaying the description
                st.markdown("""
                **Description:**
                The Object Detection project utilizes YOLO (You Only Look Once) to identify and classify objects within images efficiently. Key features include:

                - **Data Preparation:** Methods for preparing and preprocessing datasets tailored for YOLO object detection.
                - **Model Development:** Implementation of YOLO-based object detection models for real-time performance.
                - **Evaluation:** Techniques for assessing model accuracy and effectiveness, including visualizations of detection results.
                - **Application:** Demonstrations of applying the trained YOLO model to various images for accurate object detection.
                """)
                st.markdown(""" """)
                st.markdown(""" """)

                # Displaying the tools used
                st.markdown("""
                **Tools Used:**

                **Python**, **YOLO (You Only Look Once)** , **OpenCV**  , **Matplotlib**
                """)
                st.markdown(""" """)
                st.markdown(""" """)

                # Adding the GitHub link
                st.markdown("""**[GitHub](https://github.com/archanags001/ml_projects/blob/main/object_detection.pdf)**""")
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)


        with col2:
            with st.container(border=True):
                # Displaying the title of the project
                st.title("Multimodal Biometric and Multi-Attack Protection Using Image Features")

                st.markdown("""
                **Description:** Multimodal biometrics is an integration of two or more biometric systems. It overcomes the limitations of other biometrics system like unimodal biometric system. Multimodal biometric for fake identity detection using image features uses three biometric patterns and they are iris, face, and fingerprint. In this system user chooses two biometric patterns as input, which will be fused. Gaussian filter is used to smooth this fused image. Smoothed version of input image and input image is compared using image quality assessment to extract image features. In this system different image quality measures are used for feature extraction. Extracted image features are used by artificial neural network to classify an image as real or fake. Depending on whether image is real or fake appropriate action is taken. Actions could be showing user identification on screen if image is classified as real or raising an alert if image is classified as fake. This system can be used in locker, ATM and other areas where personal identification is required.""")

                # Displaying the published paper link
                st.markdown("""
                **Published Paper:** [Multimodal Biometric and Multi-Attack Protection Using Image Features](http://pnrsolution.org/Datacenter/Vol3/Issue2/140.pdf)
                """)

