import streamlit as st
import requests
import pandas as pd
from typing import List, Dict

# Configure the page
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("SHL Assessment Recommendation System")
st.markdown("""
This system helps you find the most relevant SHL assessments based on your job requirements.
Simply enter your job description or requirements below.
""")

# Input section
query = st.text_area(
    "Enter your job description or requirements:",
    height=150,
    placeholder="Example: I am hiring for Java developers who can also collaborate effectively with my business teams..."
)

# Duration filter
max_duration = st.slider(
    "Maximum assessment duration (minutes):",
    min_value=0,
    max_value=120,
    value=60,
    step=5
)

# Submit button
if st.button("Get Recommendations", type="primary"):
    if query:
        try:
            # Make API request
            response = requests.post(
                "http://localhost:8000/recommend",
                json={"text": query, "max_duration": max_duration}
            )
            
            if response.status_code == 200:
                data = response.json()
                recommendations = data.get('recommendations', [])
                
                if recommendations:
                    # Convert to DataFrame for better display
                    df = pd.DataFrame(recommendations)
                    
                    # Rename columns for better display
                    df.columns = ['Assessment Name', 'URL', 'Remote Testing',
                                 'Adaptive Support', 'Duration', 'Test Type']
                    
                    # Display results
                    st.subheader("Recommended Assessments")
                    st.dataframe(
                        df,
                        column_config={
                            "URL": st.column_config.LinkColumn(),
                            "Remote Testing": st.column_config.CheckboxColumn(),
                            "Adaptive Support": st.column_config.CheckboxColumn()
                        },
                        hide_index=True
                    )
                else:
                    st.warning("No matching assessments found. Try adjusting your query or duration filter.")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Failed to get recommendations: {str(e)}")
    else:
        st.warning("Please enter a job description or requirements.")

# Add information about evaluation metrics
with st.expander("About the Recommendation System"):
    st.markdown("""
    ### Evaluation Metrics
    Our recommendation system is evaluated using the following metrics:
    
    1. **Mean Recall@K**: Measures how many relevant assessments are retrieved in the top K recommendations.
    2. **Mean Average Precision (MAP@K)**: Evaluates both relevance and ranking order of recommendations.
    
    The system aims to provide highly relevant assessment recommendations while considering your specified duration constraints.
    """)