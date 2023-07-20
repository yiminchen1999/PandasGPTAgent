from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
import pandas as pd
import glob
import json
from datetime import datetime
import environ


def save_chart(query):
    q_s = ' If any charts or graphs or plots were created save them localy and include the save file names in your response.'
    query += ' . '+ q_s
    return query
def save_uploaded_file(uploaded_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    df_arr, df_arr_names = load_dataframe()

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df_arr, return_intermediate_steps=True, save_charts=True, verbose=True)
    return agent, df_arr, df_arr_names

def load_dataframe():
  selected_df = []

  all_files_csv = glob.glob("*.csv")
  all_files_xlsx = glob.glob("*.xlsx")
  all_files_xls = glob.glob("*.xls")
  for filename in all_files_csv:
      df = pd.read_csv(filename)
      selected_df.append(df)
  for filename in all_files_xlsx:
      df = pd.read_excel(filename)
      selected_df.append(df)
  for filename in all_files_xls:
      df = pd.read_excel(filename)
      selected_df.append(df)
  selected_df_names = all_files_csv + all_files_xlsx + all_files_xls
  return selected_df, selected_df_names

def run_query(agent, query_):
    if 'chart' or 'charts' or 'graph' or 'graphs' or 'plot' or 'plt' in query_:
        query_ = save_chart(query_)
    output = agent(query_)
    response, intermediate_steps = output['output'], output['intermediate_steps']
    thought, action, action_input, observation, steps = decode_intermediate_steps(intermediate_steps)
    store_convo(query_, steps, response)
    return response, thought, action, action_input, observation

def decode_intermediate_steps(steps):
    log, thought_, action_, action_input_, observation_ = [], [], [], [], []
    text = ''
    for step in steps:
        thought_.append(':green[{}]'.format(step[0][2].split('Action:')[0]))
        action_.append(':green[Action:] {}'.format(step[0][2].split('Action:')[1].split('Action Input:')[0]))
        action_input_.append(':green[Action Input:] {}'.format(step[0][2].split('Action:')[1].split('Action Input:')[1]))
        observation_.append(':green[Observation:] {}'.format(step[1]))
        log.append(step[0][2])
        text = step[0][2]+' Observation: {}'.format(step[1])
    return thought_, action_, action_input_, observation_, text



def query_agent(agent, query_):
    """
    Query an agent and return the response as a string.
    Args:
        agent: The agent to query.
        query: The query to ask the agent.
    Returns:
        The response from the agent as a string.
    """
    prompt = (
        """
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}
            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
            There can only be two types of chart, "bar" and "line".
            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}
            Below is the query.
            Query: 
            """
        + query_
    )

    # Run the prompt through the agent.
    response = agent.run(prompt)

    # Convert the response to a string.
    return response.__str__()

def get_convo():
    convo_file = 'convo_history.json'
    with open(convo_file, 'r',encoding='utf-8') as f:
        data = json.load(f)
    return data, convo_file

def store_convo(query, response_, response):
    data, convo_file = get_convo()
    current_dateTime = datetime.now()
    data['{}'.format(current_dateTime)] = []
    data['{}'.format(current_dateTime)].append({'Question': query, 'Answer':response, 'Steps':response_})
    
    with open(convo_file, 'w',encoding='utf-8') as f:
        json.dump(data, f,ensure_ascii=False, indent=4)