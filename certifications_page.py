import streamlit as st
import json
from streamlit_lottie import st_lottie

def certifications():
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
                    <h1> Certifications </h1>
                </div>
            """, unsafe_allow_html=True)
    path = "Animation_edu.json"
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
    st.markdown("---")

    certification_list = [
        "**AGENTIC AI ENGINEER through the Ready Tensor Agentic AI Developer Certification Program**: [https://github.com/archanags001/Portfolio/blob/main/agentic_ai_engineer.pdf](https://github.com/archanags001/Portfolio/blob/main/agentic_ai_engineer.pdf)",
        "**AGENTIC AI EXPERT through the Ready Tensor Agentic AI Developer Certification Program**: [https://github.com/archanags001/Portfolio/blob/main/agentic_ai_expert.pdf](https://github.com/archanags001/Portfolio/blob/main/agentic_ai_expert.pdf)",
        "**AGENTIC AI BUILDER through the Ready Tensor Agentic AI Developer Certification Program**: [https://github.com/archanags001/Portfolio/blob/main/agentic_ai_builder_certificate.pdf](https://github.com/archanags001/Portfolio/blob/main/agentic_ai_builder_certificate.pdf)",
        "**RAG SYSTEMS EXPERT through the Ready Tensor Agentic AI Developer Certification Program**: [https://github.com/archanags001/Portfolio/blob/main/rag_certificate.pdf](https://github.com/archanags001/Portfolio/blob/main/rag_certificate.pdf)",
        "**Agentic AI and AI Agents: A Primer for Leaders**: [https://coursera.org/share/9cc5d0d094b8e9fa07613b2f55ca423b](https://coursera.org/share/9cc5d0d094b8e9fa07613b2f55ca423b)",
        "**Prompt Engineering for ChatGPT**: [https://coursera.org/share/91eae6b8a43656ce0ef49fa94b05b41e](https://coursera.org/share/91eae6b8a43656ce0ef49fa94b05b41e)",
        "**OpenAI GPTs: Creating Your Own Custom AI Assistants**: [https://coursera.org/share/e32eb55384bb328d3b99773947605b6f](https://coursera.org/share/e32eb55384bb328d3b99773947605b6f)",
        "**AI Agents and Agentic AI Architecture in Python**: [https://coursera.org/share/ca81f0079bcd40d4bb3273c13254b78a](https://coursera.org/share/ca81f0079bcd40d4bb3273c13254b78a)",
        "**AI Agents and Agentic AI with Python & Generative AI**: [https://coursera.org/share/6e4c87e9f555c7856df96b844c173cfa](https://coursera.org/share/6e4c87e9f555c7856df96b844c173cfa)",
        "**Develop Generative AI Applications: Get Started**: [https://coursera.org/share/697af1de860ebf20acd746431367b7e5](https://coursera.org/share/697af1de860ebf20acd746431367b7e5)",
        "**Google Data Analytics Capstone: Complete a Case Study**: [https://coursera.org/share/e7bef07ce1a439d1d15735f3db74bbce](https://coursera.org/share/e7bef07ce1a439d1d15735f3db74bbce)",
        "**Data Analysis with R Programming**: [https://coursera.org/share/cbb4ed0b34396cf96ee2d9acdbf8a132](https://coursera.org/share/cbb4ed0b34396cf96ee2d9acdbf8a132)",
        "**Share Data Through the Art of Visualization**: [https://coursera.org/share/2ed3dc68de24aaede3d3e5c0d85ac654](https://coursera.org/share/2ed3dc68de24aaede3d3e5c0d85ac654)",
        "**Analyze Data to Answer Questions**: [https://coursera.org/share/2345367a8689c0c6280abb28494e2340](https://coursera.org/share/2345367a8689c0c6280abb28494e2340)",
        "**Process Data from Dirty to Clean**: [https://coursera.org/share/7c1d1289479ea0e3def67035f0dd0df8](https://coursera.org/share/7c1d1289479ea0e3def67035f0dd0df8)",
        "**Prepare Data for Exploration**: [https://coursera.org/share/d9ecc23fad5ed1c62be81ab76c6c0ff8](https://coursera.org/share/d9ecc23fad5ed1c62be81ab76c6c0ff8)",
    ]

    for i, certificate in enumerate(certification_list, start=1):
        st.markdown(f"{i}. {certificate}")

