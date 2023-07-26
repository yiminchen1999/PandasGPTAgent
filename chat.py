from langchain.agents import initialize_agent, AgentType
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
import streamlit as st
import pandas as pd
import os
import enum

# Custom enum to represent the agent type
class AgentType(enum.Enum):
    PANDAS_AGENT = "pandas_agent"

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,

}


def clear_submit():
    st.session_state["submit"] = False


def load_data(file_path):
    try:
        ext = os.path.splitext(file_path)[1][1:].lower()
    except:
        ext = file_path.split(".")[-1]
    if ext in file_formats:
        df = file_formats[ext](file_path)
        return df
    else:
        st.error(f"Unsupported file format: {ext}")
        return None


st.set_page_config(page_title="LangChain: Chat with Pandas DataFrame", page_icon="🦜")
st.title("🦜 LangChain: Chat with Pandas DataFrame")

data_directory = "csv"  # Replace with the path to the directory containing your files

file_list = os.listdir(data_directory)
selected_file = st.selectbox("Select a Data file", file_list, help="Various File formats are Support")

if selected_file:
    file_path = os.path.join(data_directory, selected_file)
    df = load_data(file_path)

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.text(msg["role"] + ": " + msg["content"])  # Display role and content

if prompt := st.text_input("User input", key="user_input", placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(
        temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, streaming=True
    )

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.PANDAS_AGENT,  # Use PANDAS_AGENT instead of OPENAI_COMPLETION
        handle_parsing_errors=True,
    )

    with st.spinner("Thinking..."):
        response = pandas_df_agent.run(st.session_state.messages[-1:])  # Pass the latest message to the agent
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.text("assistant: " + response)  # Display the assistant's response



