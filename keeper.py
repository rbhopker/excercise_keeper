#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 10:10:20 2021

@author: ricardobortothopker
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
from random import shuffle
import random
import string
import pathlib

from google.oauth2 import service_account
from googleapiclient.discovery import build
from gsheetsdb import connect

from datetime import date
st.session_state["date"] = date.today()


# streamlit_app.py

def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            # del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True
def retrieve_from_server(date,user):
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
        ],
    )
    conn = connect(credentials=credentials)
    
    # Perform SQL query on the Google Sheet.
    # Uses st.cache to only rerun when the query changes or after 10 min.
    @st.cache(ttl=600)
    def run_query(query):
        rows = conn.execute(query, headers=1)
        rows = rows.fetchall()
        return rows
    
    sheet_url = st.secrets["private_gsheets_url"]
    rows = run_query(f'SELECT * FROM "{sheet_url}"')
    
    # Print results.
    for row in rows:
        st.write(f"{row.name} has a :{row.pet}:")
    
def initial():
    st.session_state["date"] = st.date_input("Selecione a data que voce quer ver atividade.", value=st.session_state["date"])
    idx = 0
    if st.session_state["username"] == "isa_ribeiro":
        idx=1
    st.session_state["user"] = st.selectbox(
     'De quem voce quer ver a atividade?',
     ('rbhopker', 'isa_ribeiro'),index=idx)
    if st.button("Ver"):
        retrieve_from_server(st.session_state["date"],st.session_state["user"])
        
if check_password():
    initial()
    st.write("Here goes your normal Streamlit app...")
    st.button("Click me")

# def load_instructions():
#     txt2 = """Travel Salesperson (TSP) Problem Instructions: \n\n
# The TSP problem is composed of several points, your task is to try to find the shortest path that links all points, with straight lines, while going to each point once and only once and returning to the starting position.\n 
# You will try to solve 13 TSP problems. 
#     """
#     st.markdown(txt2)

#     from PIL import Image
#     st.markdown("Below you can find an example of a problem:")
#     image = Image.open('example_empty.jpg')
#     st.image(image, caption='Example problem')
    
#     st.markdown("Below you can find an example of a problem solved with the shortest path possible:")
#     image = Image.open('example_optimal.jpg')
#     st.image(image, caption='Example problem optimally solved')
    
#     st.markdown("Below you can find an example of a problem solved without the shortest path possible, but still a valid solution")
#     image = Image.open('example_valid.jpg')
#     st.image(image, caption='Example problem solved, with valid solution')
    
#     st.markdown("Below you can find an example of a problem solved without a valid solution")
#     image = Image.open('example_invalid.png')
#     st.image(image, caption='Example problem invalid solution. (A point was not visited)')
#     st.markdown("To create a link between two points, click on a point and then on the next.")
#     st.markdown("To delete a link between two points, click on the line.")
# def query_counter():
#     credentials = service_account.Credentials.from_service_account_info(
#         st.secrets["gcp_service_account"],
#         scopes=[
#             "https://www.googleapis.com/auth/spreadsheets",
#         ],
#     )
#     conn = connect(credentials=credentials)

#     service = build('sheets','v4',credentials=credentials)
#     sheet = service.spreadsheets()

#     Sheet0 = st.secrets["Sheet0"]
#     Sheet1 = st.secrets["Sheet1"]
#     result_counter = sheet.values().get(spreadsheetId=Sheet1,range="current_test_number!A1:A3").execute()
#     values_counter = result_counter.get('values',[])
#     df1 = pd.DataFrame(values_counter)
#     return df1
# def update_counter(df):
#     credentials = service_account.Credentials.from_service_account_info(
#         st.secrets["gcp_service_account"],
#         scopes=[
#             "https://www.googleapis.com/auth/spreadsheets",
#         ],
#     )
#     conn = connect(credentials=credentials)

#     service = build('sheets','v4',credentials=credentials)
#     sheet = service.spreadsheets()

#     Sheet1 = st.secrets["Sheet1"]
#     df_as_list = df.values.tolist()
#     dictt = {'values':df_as_list}
#     request = sheet.values().update(spreadsheetId=Sheet1,
#                                    range="current_test_number!A1",
#                                    valueInputOption='USER_ENTERED', 
#                                    body=dictt).execute()
# def query_db(last_row):
#     credentials = service_account.Credentials.from_service_account_info(
#         st.secrets["gcp_service_account"],
#         scopes=[
#             "https://www.googleapis.com/auth/spreadsheets",
#         ],
#     )
#     conn = connect(credentials=credentials)

#     service = build('sheets','v4',credentials=credentials)
#     sheet = service.spreadsheets()

#     Sheet0 = st.secrets["Sheet0"]
#     result = sheet.values().get(spreadsheetId=Sheet0,range=f"results_streamlit!A1:E{last_row}").execute()
#     values = result.get('values',[])
#     df = pd.DataFrame(values[1:],columns=values[0])
#     return df
# def update_db(df):
#     credentials = service_account.Credentials.from_service_account_info(
#         st.secrets["gcp_service_account"],
#         scopes=[
#             "https://www.googleapis.com/auth/spreadsheets",
#         ],
#     )
#     conn = connect(credentials=credentials)

#     service = build('sheets','v4',credentials=credentials)
#     sheet = service.spreadsheets()

#     Sheet0 = st.secrets["Sheet0"]
#     streamlit_csv_as_list = df.values.tolist()
#     streamlit_csv_as_list.insert(0,df.columns.tolist())
    
#     dict_write = {'values':streamlit_csv_as_list}
#     request = sheet.values().update(spreadsheetId=Sheet0,
#                                     range="results_streamlit!A1",
#                                     valueInputOption='USER_ENTERED', 
#                                     body=dict_write).execute()
    
# # Create a connection object.
# # credentials = service_account.Credentials.from_service_account_info(
# #     st.secrets["gcp_service_account"],
# #     scopes=[
# #         "https://www.googleapis.com/auth/spreadsheets",
# #     ],
# # )

# # result_counter = sheet.values().get(spreadsheetId=Sheet1,range="current_test_number!A1:A3").execute()
# # values_counter = result_counter.get('values',[])
# # df1 = pd.DataFrame(values_counter)
# # st.write(df1)
# # last_row = df1.iloc[2][0]
# # result = sheet.values().get(spreadsheetId=Sheet0,range=f"results_streamlit!A1:E{last_row}").execute()
# # values = result.get('values',[])
# # df0 = pd.DataFrame(values)
# # st.write(df0)


# # df1.iloc[1][0] = int(df1.iloc[1][0])+1
# # df_as_list = df1.values.tolist()
# # dictt = {'values':df_as_list}
# # request = sheet.values().update(spreadsheetId=Sheet1,
# #                                range="current_test_number!A1",
# #                                valueInputOption='USER_ENTERED', 
# #                                body=dictt).execute()




# if 'accepted' not in st.session_state:
#     txt ="""This is an experiment designed to analyze the relationship between system complexity and human effort for Ricardo Hopker’s SDM thesis at MIT under the supervision of Prof. Olivier De Weck. \n
#         By continuing here, you accept to be a part of this experiment and allow MIT and its affiliates to use the data collected for research purposes. \n
#         This is a volunteered experiment, and you are free exit. \n
#         The expected time for conclusion of the experiment is 30 minutes to 60 minutes, but you may take as long as you want. \n
#         The data collected and the result will be anonymized. \n
#         Please complete the experiment without interuptions. \n
#         Please complete the experiment only using a computer. (The experiment does not work correctly on smartphones)
#         """
#     st.markdown(txt)
#     agree = st.checkbox('I agree')
#     if st.button(label='Continue'):
#         if agree:
#             st.session_state['accepted'] = agree
#             st.experimental_rerun()
#         else:
#             'Please accept before continuing the experiment'
#     st.stop()
# if (st.session_state['accepted'] and 'instructions' not in st.session_state):
#     load_instructions()
#     st.markdown("The experiment will start in the next page.")
#     if st.button(label='Continue'):
#         st.session_state['instructions'] = True
#         st.experimental_rerun()
#     st.stop()
    
# if 'session_id' not in st.session_state:
#     letters = string.ascii_lowercase    
# @st.cache()
# def create_experiments():
#     from itertools import combinations
#     data = list(range(0,30,2))
#     cc = np.array(list(combinations(data,13)))
#     np.random.seed(22689)
#     random = np.random.randint(2,size=np.shape(cc))
#     notRandom = random==0
#     problems = cc+random
#     problems=np.where(problems!=29,problems,problems+1)
#     problems2 = cc+notRandom
#     problems2=np.where(problems2!=29,problems2,problems2+1)
#     problemsTot = np.concatenate([problems.T,problems2.T],axis=1).T
#     return problemsTot
# problemsTot=create_experiments()
# @st.cache()
# def load_xy_dict():
    
#     path = pathlib.Path(__file__).parents[0]
#     # path = r"/Users/ricardobortothopker/OneDrive - Massachusetts Institute of Technology/Classes/Thesis/excels/Points for TSP/"
#     file = r"TSP Problems3.pkl"
#     url = path / file
#     with open(url,'rb') as f:  # Python 3: open(..., 'rb')\n",
#         xyArrDict = pickle.load(f)
#     return xyArrDict
# xyArrDict = load_xy_dict()
# if 'current_test' not in st.session_state:
#     # credentials = service_account.Credentials.from_service_account_info(
#     #     st.secrets["gcp_service_account"],
#     #     scopes=[
#     #         "https://www.googleapis.com/auth/spreadsheets",
#     #     ],
#     # )
#     # conn = connect(credentials=credentials)

#     # service = build('sheets','v4',credentials=credentials)
#     # sheet = service.spreadsheets()

#     # Sheet0 = st.secrets["Sheet0"]
#     # Sheet1 = st.secrets["Sheet1"]
#     # result_counter = sheet.values().get(spreadsheetId=Sheet1,range="current_test_number!A1:A3").execute()
#     # values_counter = result_counter.get('values',[])
#     # df1 = pd.DataFrame(values_counter)
#     df1 = query_counter()
#     cur_test_num = int(df1.iloc[1][0])

#     if cur_test_num>=len(problemsTot):
#         cur_test_num=-1
#     df1.iloc[1][0] = cur_test_num+1
#     st.session_state['current_test'] = cur_test_num
#     update_counter(df1)
#     # df_as_list = df1.values.tolist()
#     # dictt = {'values':df_as_list}
#     # request = sheet.values().update(spreadsheetId=Sheet1,
#     #                                range="current_test_number!A1",
#     #                                valueInputOption='USER_ENTERED', 
#     #                                body=dictt).execute()

#     cur_test_num = st.session_state['current_test']
#     cur_test = problemsTot[cur_test_num,:].copy()
#     shuffle(cur_test)
#     st.session_state['current_test'] = cur_test

# if 'count' not in st.session_state:
#     st.session_state['count'] = 0
#     st.session_state['session_id'] = ''.join(random.choice(letters) for i in range(30))
# elif st.session_state['count'] == len(st.session_state['current_test']):
#     st.write('You have reached the end of the experiment')
#     st.write('Thank you for participating!')
#     st.stop()
# st.write("Once you have a valid solution a button to advance will appear. Click on it to submit your answer.")
# st.write("The goal is to find the smallest total lenght of the red lines.")
# st.write(f"excercise {st.session_state['count']+1} of {len(st.session_state['current_test'])}")
# cur_test = st.session_state['current_test']
# test_id = cur_test[st.session_state['count']]
# test_id =f"id {test_id}"
# cur_id = xyArrDict[test_id]
# x = cur_id[:,0].tolist()
# y = cur_id[:,1].tolist()
# xy = list(zip(x,y))
# # path = {'x': [[4, 0], [1, 2], [3, 4], [1, 0], [2, 3]], 'y': [[16, 0], [1, 4], [9, 16], [1, 0], [4, 9]]}
# def valid_path(path):
#     if len(x)!=len(path['x']):
#         return False
#     xy_path =[]
#     path_dict = {}
#     for i in range(len(path['x'])):
#         points = list(zip(path['x'][i],path['y'][i]))
#         point0 = np.where(np.all(points[0]==np.array(xy),axis=1))[0][0]
#         point1 = np.where(np.all(points[1]==np.array(xy),axis=1))[0][0]
#         xy_path.append([point0,point1])
#         path_dict[i] = []
#     for i in xy_path:
#         path_dict[i[0]].append(i[1])
#         path_dict[i[1]].append(i[0])
#     visited = [0]
#     tf = True
#     cur_node = 0
#     while tf:
#         # print(f'Current node: {cur_node}')
#         # print(f'Visited: {visited}')
#         # print(f'Possible nodes: {path_dict[cur_node]}')
#         # print(f'-------')
#         if path_dict[cur_node][0] not in visited:
#             cur_node = path_dict[cur_node][0]
#             visited.append(cur_node)
#         elif path_dict[cur_node][1] not in visited:
#             cur_node = path_dict[cur_node][1]
#             visited.append(cur_node)
#         else:
#             if len(visited) == len(x) and 0 in path_dict[cur_node]:
#                 tf = False
#                 return True
#             else:
#                 tf = False
#                 return False
#         if len(visited)> len(x):
#             tf = False
#             return False
# # path = {'x': [[4, 0], [1, 2], [3, 4], [1, 0], [2, 3],[2, 3],[3,2]],  
# #         'y': [[16, 0], [1, 4], [9, 16], [1, 0], [4, 9], [4, 9],[9,4]]}
# def remove_double_paths(path):
#     if len(path['x'])>1:
#         xy_path=[]
#         path_dict = {}
#         for i in range(len(path['x'])):
#             points = list(zip(path['x'][i],path['y'][i]))
#             point0 = np.where(np.all(points[0]==np.array(xy),axis=1))[0][0]
#             point1 = np.where(np.all(points[1]==np.array(xy),axis=1))[0][0]
#             xy_path.append([point0,point1])
#         for i in range(len(x)):
#             path_dict[i] = []
#         x_path = []
#         y_path = []
#         for i in xy_path:
#             if i[1] not in path_dict[i[0]]:
#                 path_dict[i[0]].append(i[1])
#             if i[0] not in path_dict[i[1]]:
#                 path_dict[i[1]].append(i[0])
#         visited = []
#         xy_path =[]
#         for key,item in path_dict.items():
#             visited.append(key)
#             if item !=[]:
#                 for i in item:
#                     if i not in visited:
#                         xy_path.append([key,i])
#         x_path = []
#         y_path = []
#         for i in xy_path:
#             x_path.append([x[i[0]],x[i[1]]])
#             y_path.append([y[i[0]],y[i[1]]])
#         outpath ={}
#         outpath['x'] = x_path
#         outpath['y'] = y_path
#         return outpath
#     return path
# def path_to_point(path):
#     xy_path =[]
#     for i in range(len(path['x'])):
#         points = list(zip(path['x'][i],path['y'][i]))
#         point0 = np.where(np.all(points[0]==np.array(xy),axis=1))[0][0]
#         point1 = np.where(np.all(points[1]==np.array(xy),axis=1))[0][0]
#         xy_path.append([point0,point1])
#     return xy_path
# # remove_double_paths(path)


# if 'last_point' not in st.session_state:
#     st.session_state['last_point'] = []
#     st.session_state['path'] = {'x':[],'y':[]}

# else:
#     if st.session_state['path']['x']!=[]:
#         # print(f"before {st.session_state['path']}")
#         new_path = remove_double_paths(st.session_state['path'])
#         # print(f"new_path {new_path}")
#         if new_path!=st.session_state['path']:
#             st.session_state['path'] = new_path
#             st.experimental_rerun()
#         # print(f"after {st.session_state['path']}")
        
# if 'start_time' not in st.session_state:
#     st.session_state['start_time'] = datetime.now()
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=x, y=y,mode='markers',text=list(range(len(x))), hovertemplate='point number:%{text}'))
# fig.update_yaxes(visible=False, showticklabels=False)
# fig.update_xaxes(visible=False, showticklabels=False)


# for i in range(len(st.session_state['path']['x'])):
#     x1 = np.linspace(st.session_state['path']['x'][i][0],st.session_state['path']['x'][i][1],10)
#     y1 = np.linspace(st.session_state['path']['y'][i][0],st.session_state['path']['y'][i][1],10)
#     fig.add_trace(go.Scatter(x=x1, y=y1, mode="lines",marker_color='rgba(255, 0, 0, 1)',text=list(range(len(x1))), hovertemplate='path (click to erase)'))
# fig.update_layout(showlegend=False)
# # fig.update_layout(
# #     autosize=False,
# #     width=200,
# #     height=200,)
# fig.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True)
# fig.update_yaxes(
#     scaleanchor = "x",
#     scaleratio = 1,
#   )
# # if st.button(label='View instructions'):
# with st.expander('View instructions'):
#     load_instructions()

