import streamlit as st
import pandas as pd
from groq import Groq
# import requests
from dotenv import load_dotenv
import os

# Load .env file for API keys
load_dotenv()

# Load Groq API key from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')

# Load preprocessed dataset
data = pd.read_csv('Government_Schemes_Dataset.csv')
data.fillna('', inplace=True)

# Filter function
def filter_schemes(gender, caste, residency, differently_abled, student, scheme_category):
    filtered_data = data[
        (data['Gender'].str.contains(gender, case=False, na=False) | data['Gender'].str.contains('All')) &
        (data['Caste'].str.contains(caste, case=False, na=False) | data['Caste'].str.contains('All')) &
        (data['Residency'].str.contains(residency, case=False, na=False) | data['Residency'].str.contains('All')) &
        (data['Differently Abled'].str.contains(differently_abled, case=False, na=False) | data['Differently Abled'].str.contains('All')) &
        (data['Student'].str.contains(student, case=False, na=False) | data['Student'].str.contains('All')) &
        (data['Scheme Category'].str.contains(scheme_category, case=False, na=False) | data['Scheme Category'].str.contains('All'))
    ]
    
    return filtered_data

# Chatbot function using Groq API (LLM)
def ask_chatbot(query, context):
    client = Groq(api_key=groq_api_key)
    
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing information on government policies."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ],
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stream=False
    )
    return completion.choices[0].message.content

# Title
st.title("Government Policies Recommendation for Citizens of Maharashtra")

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')

gender = st.sidebar.selectbox('Gender', ('Male', 'Female', 'Transgender', 'All'))
caste = st.sidebar.selectbox('Caste', sorted(data['Caste'].unique()))
residency = st.sidebar.selectbox('Residency', ('Rural', 'Urban', 'Both'))
differently_abled = st.sidebar.selectbox('Differently Abled', ('Yes', 'No'))
student = st.sidebar.selectbox('Student', ('Yes', 'No'))
scheme_category = st.sidebar.selectbox('Scheme Category', sorted(data['Scheme Category'].unique()))

# Initialize session state
if 'filtered_schemes' not in st.session_state:
    st.session_state.filtered_schemes = None
if 'selected_scheme' not in st.session_state:
    st.session_state.selected_scheme = None
if 'summary' not in st.session_state:
    st.session_state.summary = None

# Submit button
if st.sidebar.button('Submit'):
    # Filter schemes based on user input
    st.session_state.filtered_schemes = filter_schemes(gender, caste, residency, differently_abled, student, scheme_category)

    if st.session_state.filtered_schemes.empty:
        st.warning("No schemes found matching your criteria. Please try adjusting your inputs.")
        st.session_state.selected_scheme = None  # Reset selected scheme
    else:
        st.success(f"Found {len(st.session_state.filtered_schemes)} matching schemes!")

# If there are filtered schemes, display them in a dropdown
if st.session_state.filtered_schemes is not None:
    # Dropdown to select a scheme
    scheme_names = st.session_state.filtered_schemes['Scheme Name'].unique()
    selected_scheme_name = st.selectbox('Select a Scheme to view details', scheme_names)

    # If a scheme is selected, store it in the session state
    if selected_scheme_name:
        st.session_state.selected_scheme = st.session_state.filtered_schemes[st.session_state.filtered_schemes['Scheme Name'] == selected_scheme_name].iloc[0]
        # Generate a summary for the policy
        st.session_state.summary = ask_chatbot("Summarize this policy", st.session_state.selected_scheme['Scheme Details'])

# Display the selected scheme details
if st.session_state.selected_scheme is not None:
    scheme_details = st.session_state.selected_scheme
    st.write(f"**Scheme Name:** {scheme_details['Scheme Name']}")
    st.write(f"**Category:** {scheme_details['Scheme Category']}")
    st.write(f"**Eligibility:** {scheme_details['Eligibility']}")
    st.write(f"**Benefits:** {scheme_details['Scheme Benefits']}")
    st.write(f"**Documents Required:** {scheme_details['Documents Required']}")
    st.write(f"**Application Process:** {scheme_details['Application Process']}")
    st.write(f"**More Details:** {scheme_details['Scheme Details']}")

    # Display policy summary
    if st.session_state.summary:
        st.write(f"**Summary:** {st.session_state.summary}")

# Chatbot interface to ask any questions regarding policy
st.header("Ask Questions About the Policy")

user_query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if st.session_state.selected_scheme is not None:
        # Get the context from the selected scheme
        context = st.session_state.selected_scheme['Scheme Details']
        # Get the chatbot response
        chatbot_answer = ask_chatbot(user_query, context)
        st.write(f"**Answer:** {chatbot_answer}")
    else:
        st.warning("Please select a scheme before asking questions.")
