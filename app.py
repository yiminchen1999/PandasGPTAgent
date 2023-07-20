import os
import streamlit as st
from functions import *
import platform
import openai
from streamlit_chat import message
from streamlit_image_select import image_select
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import subprocess

def generate_plot():
    # Sample data for the plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Sample Plot')

    return fig

def setOpenAIKey(key):
    os.environ['OPENAI_API_KEY'] = key


def get_text(n):
    input_text = st.text_input('How can I help?', '', key="input{}".format(n))
    return input_text


def show_data(tabs, df_arr):
    for i, df_ in enumerate(df_arr):
        print(i, len(df_))
        with tabs[i]:
            st.dataframe(df_)


def main():
    st.title("Demo Agent")
    openai_key = st.sidebar.text_input('Open AI API KEY', key="openai_key", type="password")
    if st.sidebar.button('Update Key'):
        setOpenAIKey(openai_key)
    st.sidebar.title('Thinking Process')
    uploaded_file = st.file_uploader("Choose files to upload (csv, xls, xlsx)", type=["csv", "xls", "xlsx"],
                                     accept_multiple_files=True)
    agent = ''
    if uploaded_file:
        for file in uploaded_file:
            agent, selected_df, selected_df_names = save_uploaded_file(file)
        st.session_state["tabs"].clear()
        for df_name in selected_df_names:
            st.session_state.tabs.append(df_name)
        tabs = st.tabs([s.center(9, "\u2001") for s in st.session_state["tabs"]])
        show_data(tabs, selected_df)

    st.header("AI Agent Output Directory")
    if st.button('Open Directory'):
        current_dir = os.getcwd()
        if platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", current_dir])
        elif platform.system() == "Windows":
            subprocess.Popen(["explorer", current_dir])
        else:
            print("Directory opened:", current_dir)


    imgs_png = glob.glob('*.png')
    imgs_jpg = glob.glob('*.jpg')
    imgs_jpeeg = glob.glob('*.jpeg')
    imgs_ = imgs_png + imgs_jpg + imgs_jpeeg
    if len(imgs_) > 0:
        img = image_select("Generated Charts/Graphs", imgs_, captions=imgs_, return_value='index')
        st.write(img)

    st.header("Query The Dataframes")
    x = 0
    user_input = get_text(x)
    if st.button('Query'):
        x += 1
        print(user_input, len(user_input))
        response, thought, action, action_input, observation = run_query(agent, user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)

        # Display the generated response messages
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


        # Display the plots
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
