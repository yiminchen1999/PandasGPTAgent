import os
import streamlit as st
import os
import glob
import streamlit as st
from functions import *
import openai
from streamlit_chat import message
from streamlit_image_select import image_select
import environ

env = environ.Env()
environ.Env.read_env()

OPENAI_API_KEY = env("apikey")

def get_text(n):
    input_text = st.text_input('How can I help?', '', key="input{}".format(n))
    return input_text

def show_data(tabs, df_arr):
    for i, df_ in enumerate(df_arr):
        with tabs[i]:
            st.dataframe(df_)

def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    return json.loads(response)

def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.line_chart(df)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

def main():
    st.title("Pandas AI Agent - Demo")
    st.sidebar.title('🤖 Thinking Process 🤖')

    # Get the list of CSV files in a specific directory
    csv_directory = '/Users/chenyimin/PycharmProjects/PandasGPTAgent/csv'
    csv_files = glob.glob(os.path.join(csv_directory, '*.csv'))

    if csv_files:
        selected_df_names = [os.path.basename(file) for file in csv_files]
        selected_df = [pd.read_csv(file) for file in csv_files]
        st.session_state["tabs"].clear()
        for df_name in selected_df_names:
            st.session_state.tabs.append(df_name)
        tabs = st.tabs([s.center(9, "\u2001") for s in st.session_state["tabs"]])
        show_data(tabs, selected_df)

    st.header("AI Agent Output Directory")
    if st.button('Open Directory'):
        os.startfile(os.getcwd())

    imgs_png = glob.glob('*.png')
    imgs_jpg = glob.glob('*.jpg')
    imgs_jpeeg = glob.glob('*.jpeg')
    imgs_ = imgs_png + imgs_jpg + imgs_jpeeg
    if len(imgs_) > 0:
        img = image_select("Generated Charts/Graphs", imgs_, captions =imgs_, return_value = 'index')
        st.write(img)

    st.header("Query The Dataframes")
    x = 0
    user_input = get_text(x)
    if st.button('Query'):
        x+=1
        #st.write("You:", user_input)
        print(user_input, len(user_input))
        response, thought, action, action_input, observation = run_query(agent, user_input)
        #st.write("Pandas Agent: ")
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        for i in range(0, len(thought)):
            st.sidebar.write(thought[i])
            st.sidebar.write(action[i])
            st.sidebar.write(action_input[i])
            st.sidebar.write(observation[i])
            st.sidebar.write('====')

if __name__ == "__main__":
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []
    
    if 'tabs' not in st.session_state:
        st.session_state['tabs'] = []

    main()
