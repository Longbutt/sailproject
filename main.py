# import pandas as pd
from query import *
import streamlit as st
import plotly.express as px
# import matplotlib.pyplot as plt
# import numpy as np
# import datetime
import SM_handling
from random import randint
# import openpyxl
import joblib
import re
# import json
# import os
# import json
# import os
# import subprocess
# import psutil
# import tempfile
# import platform
import streamlit.components.v1 as components
# from azure.identity import ClientSecretCredential
# from azure.mgmt.sql import SqlManagementClient
# from azure.mgmt.sql.models import FirewallRule
# import requests
#
#
# def get_public_ip():
#     response = requests.get('https://api.ipify.org')
#     return response.text


# public_ip = get_public_ip()

# SQLuser = 'root'
# SQLPW = 'newpassword'
# SQLHost = '127.0.0.1'
# SQLport = 3306
# SQLdb = 'bac'

# # Replace these values with your own
# client_id = 'f4aaf264-83b7-481a-8c05-7865a2edfe1a'

# tenant_id = '39da5ba9-aeb3-4a9d-8bfe-9f1e2da64000'
#
# credential = ClientSecretCredential(tenant_id, client_id, client_secret)
#
# # Replace these variables with your own values
# subscription_id = '47936833-b041-464e-9654-23cd63fa217d'
# resource_group_name = 'New_resource_group'
# server_name = 'sailproject'
# firewall_rule_name = 'PythonScriptAccess'
# start_ip_address = public_ip
# end_ip_address = public_ip
#
#
# # Initialize the SqlManagementClient
# sql_client = SqlManagementClient(credential, subscription_id)
#
# # Create or update the firewall rule
# sql_client.firewall_rules.create_or_update(
#     resource_group_name,
#     server_name,
#     firewall_rule_name,
#     FirewallRule(start_ip_address=start_ip_address, end_ip_address=end_ip_address)
# )

# Run your Python script here
# ...

config = {
    'user': st.secrets["user"]["name"],
    'password': st.secrets["user"]["password"],
    'host': st.secrets["user"]["host"],
    'database': st.secrets["user"]["database"],
    'ssl_ca': 'BaltimoreCyberTrustRoot.crt.pem'
}


connection_string = (st.secrets["connection_string"]["string"])


def ChangeWidgetFontSize(wgt_txt, wch_font_size='12px'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
    for (i = 0; i < elements.length; ++i) {
        if (elements[i].innerText == |wgt_txt|) {
            elements[i].style.fontSize='""" + wch_font_size + """';}
    } </script> """
    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f" {htmlstr}", height=0, width=0)


# Create session, that can be refreshed when wanted
if "state" not in st.session_state:
    st.session_state.state = {}
if "widget_key" not in st.session_state.state:
    st.session_state.state["widget_key"] = str(randint(1000, 100000000))

# Header in App
st.title('Performance Visualization and Analysis Program')

# Text in app
st.text('Visualize and analyse sailing sessions')
ChangeWidgetFontSize('Visualize and analyse sailing sessions', '26px')

# Layout
st.sidebar.title('Navigation')

# Initialize the session state variable for the current page if it doesn't exist
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# Set up the sidebar radio button using the session state variable
options = st.sidebar.radio('Pages', options=['Home', 'Analysis', 'Data Statistics', 'Upload', 'Draft'], key='current_page')

if st.sidebar.button("Clear and reload"):
    st.experimental_rerun()

if options == 'Analysis':
    # Possible data to show
    datatype = st.radio('Plot type', options=['Graph', 'Map', 'Polar Plot', 'Bar Chart'], horizontal=True)

    # Data or plot
    plot, data, evaluation = st.tabs(['Plot', 'Data', 'Evaluation'])


    st.header('Filters:')

    col40, col41, col42 = st.columns(3)

    # Search criteria
    with col40:
        wind_cond = st.radio('Wind conditions', options=['All', 'Onshore', 'Offshore'], horizontal=True)

    with col41:
        tack = st.radio('Tack', options=['All', 'Starboard', 'Port'], horizontal=True)

    with col42:
        sail_type = st.radio('Type', options=['All', 'Upwind', 'Downwind'], horizontal=True)

    crew = st.multiselect('Select crew', options=SM_handling.fetch_crew(config=config))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_options = ['All', '0', '1', '2']
        current_idx = st.radio('Current', options=current_options, key='current')
        current = current_options[current_options.index(current_idx)] if current_idx != 'All' else None

    with col2:
        wave_options = ['All', '0', '1', '2', '3']
        wave_idx = st.radio('Waves', options=wave_options, key='waves')
        waves = wave_options[wave_options.index(wave_idx)] if wave_idx != 'All' else None

    with col3:
        rating_options = ['All', '1', '2', '3', '4', '5']
        rating_idx = st.radio('Rating', options=rating_options, key='rating')
        rating = rating_options[rating_options.index(rating_idx)] if rating_idx != 'All' else None

    with col4:
        vmg_avg = st.number_input('Minimum VMG Average:', value=0)
        wind_avg = st.number_input('Minimum Windspeed Average', value=0)
        run_length = st.number_input('Minimum Segment length', value=0, step=10)

    crew_map = {'All': None, 'Frederik/Jakob': 'Frederik/Jakob', 'Daniel/Buhl': 'Daniel/Buhl',
                'Kristian/Marcus': 'Kristian/Marcus', 'JC/Mads': 'JC/Mads'}
    tack_map = {'All': None, 'Starboard': 'Starboard', 'Port': 'Port'}
    tack_param = tack_map.get(tack)

    sail_type_map = {'All': None, 'Upwind': 'Upwind', 'Downwind': 'Downwind'}
    sail_type_param = sail_type_map.get(sail_type)

    wind_map = {'All': None, 'Onshore': 'Onshore', 'Offshore': 'Offshore'}
    wind_param = wind_map.get(wind_cond)

    col30, col31, col32, col33 = st.columns(4)

    with col30:
        available_dates = fetch_dates(config=config)['DATE']
        dates = pd.to_datetime(available_dates).dt.strftime("%Y-%m-%d").tolist()
        date = st.multiselect('Date', dates, key='option_plot')
    with col31:
        Area = st.multiselect('Select Area', options=SM_handling.fetch_area(config=config))
    df = query_database(wind=wind_param, crew=crew, vmg_avg=vmg_avg, date=date, current=current, waves=waves,
                        rating=rating, wind_avg=wind_avg, run_length=run_length, area=Area, tack=tack_param,
                        type=sail_type_param, config=config)
    with col32:
        session = st.multiselect('Select Session', options=df['session_id'].unique())

    if session:
        df = query_database(wind=wind_param, crew=crew, vmg_avg=vmg_avg, date=date, current=current, waves=waves,
                        rating=rating, wind_avg=wind_avg, run_length=run_length, area=Area, session_id=session,
                        tack=tack_param, type=sail_type_param, config=config)
    with col33:
        segment = st.multiselect('Select Segment', options=df['ident'].unique())

    if segment:
        df = query_database(wind=wind_param, crew=crew, vmg_avg=vmg_avg, date=date, current=current, waves=waves,
                        rating=rating, wind_avg=wind_avg, run_length=run_length, area=Area, session_id=session,
                        segment=segment, tack=tack_param, type=sail_type_param, config=config)
    if len(df) < 1:
        st.warning('No data found for the given input')

    with plot:
        hover_data_opt = df.columns.tolist()
        hover_data_sel = st.multiselect("Hoverdata", hover_data_opt)
        if datatype == 'Graph':
            if len(df) > 0:

                # create a list of options for the dropdown
                axis_opt = ['vmg', 'Roll', 'TWA', 'Pitch', 'WindSpeed', 'time', 'Yaw', 'SOG', 'COG', 'WindDirection', 'rel_time']

                col20, col21, col22 = st.columns(3)
                # create the dropdown using the selectbox function
                with col20:
                    yaxis_sel = st.selectbox('Y-axis:', axis_opt, index=axis_opt.index('vmg'))
                with col21:
                    xaxis_sel = st.selectbox('X-axis:', axis_opt, index=axis_opt.index('rel_time'))
                with col22:
                    # create the slider using the select_slider function
                    smoothing = st.select_slider('Smoothing:', options=range(1, 31))

                # apply smoothing to the y-axis data using a rolling mean
                df[yaxis_sel] = df[yaxis_sel].rolling(smoothing).mean()

                # Create the line chart using plotly express
                fig = px.line(df, x=xaxis_sel, y=yaxis_sel, color='ident', hover_name='Crew',
                              hover_data=hover_data_sel)
                # Update the x-axis title
                fig.update_xaxes(title_text=xaxis_sel)
                # Update the y-axis title
                fig.update_yaxes(title_text=yaxis_sel)

                # Show the plot using streamlit
                st.plotly_chart(fig)
            else:
                st.warning('No data found for the given input')

        elif datatype == 'Map':

            col50, col51 = st.columns(2)

            with col50:
                color_map_opt = list(df.columns)

                color_map_sel = st.selectbox('Colormap:', color_map_opt, index=color_map_opt.index('Crew'))

            with col51:
                # create a list of options for the dropdown
                color_opt = dir(px.colors.sequential)
                color_opt = [item for item in color_opt if '__' not in item and 'swatches' not in item]

                # create the dropdown using the selectbox function
                color_sel = st.selectbox('Color:', color_opt, index=color_opt.index('Turbo'))

            GPSplot = px.scatter_mapbox(df, lat="latitude", lon="longitude", color=color_map_sel,
                                        color_continuous_scale=color_sel,
                                        hover_data=hover_data_sel)
            GPSplot.update_layout(mapbox_style="carto-positron", mapbox_zoom=12,
                                  margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(GPSplot)

        elif datatype == 'Polar Plot':
            # create a list of options for the dropdown
            theta_opt = ['TWA', 'Yaw', 'COG', 'WindDirection']
            r_opt = ['vmg', 'Roll', 'Pitch', 'WindSpeed', 'Yaw', 'SOG', 'rel_time', 'time']
            cmap_opt = list(df.columns)

            col60, col61, col62 = st.columns(3)

            # create the dropdown using the selectbox function
            with col60:
                theta_sel = st.selectbox('Theta:', theta_opt, index=theta_opt.index('TWA'))

            with col61:
                r_sel = st.selectbox('Radius:', r_opt, index=r_opt.index('vmg'))

            with col62:
                cmap_sel = st.selectbox('Colormap:', cmap_opt, index=cmap_opt.index('ident'))

            if theta_sel == 'TWA':
                theta_range = [-180, 180]
                direction = 'counterclockwise'
            else:
                theta_range = [0, 360]
                direction = 'clockwise'

            PolarPlot = px.scatter_polar(df, r=r_sel, theta=theta_sel, range_theta=theta_range,
                                   start_angle=90, direction=direction, color=cmap_sel,
                                         hover_data=hover_data_sel)
            st.plotly_chart(PolarPlot)

        elif datatype == 'Bar Chart':
            # create a list of options for the dropdown
            x_opt = ['session_id', 'Date', 'ident', 'Crew', 'Wind']
            y_opt = ['vmg', 'Roll', 'Pitch', 'WindSpeed', 'Yaw']
            y2_opt = ['Mean', 'Standard Deviation']
            color_opt = ['Crew', 'Date', 'session_id']

            col9, col10, col11, col12 = st.columns(4)
            # create the dropdown using the selectbox function
            with col9:
                x_sel = st.selectbox('X-axis:', x_opt, index=x_opt.index('session_id'))

            with col10:
                y_sel = st.selectbox('Y-axis:', y_opt, index=y_opt.index('Roll'))

            with col11:
                y2_sel = st.selectbox('Value:', y2_opt, index=y2_opt.index('Mean'))

            with col12:
                color_sel = st.selectbox('Colormap:', color_opt, index=color_opt.index('Crew'))



            if y2_sel == 'Mean':
                values = df.groupby([x_sel, color_sel])[y_sel].mean().reset_index()
            elif y2_sel == 'Standard Deviation':
                values = df.groupby([x_sel, color_sel])[y_sel].std().reset_index()


            BarChart = px.bar(values, x=x_sel, y=y_sel, color=color_sel, barmode='group',
                              labels={y_sel: y_sel+' '+y2_sel})
            # BarChart.update_xaxes(title_font_family="Arial")
            st.plotly_chart(BarChart)

    with data:
        oversigt_opt = ['By Point', 'By Segment', 'By Session', 'just show me everythang']

        # create the dropdown using the selectbox function
        oversigt_sel = st.radio('Overview:', oversigt_opt, index=oversigt_opt.index('By Segment'), horizontal=True)

        if oversigt_sel == 'By Point':
            inter_columns = ['vmg', 'Roll', 'TWA', 'Pitch', 'WindSpeed', 'time', 'Yaw', 'SOG', 'COG', 'WindDirection']
            st.write(df[inter_columns].describe())
        elif oversigt_sel == 'By Segment':

            grouped_seg_data = df.groupby(['session_id', 'stream_id'])
            #
            # overview_seg_data = grouped_seg_data.Crew

            agg_funcs = {
                'vmg': ['min', 'mean', 'max', 'std'],
                'WindSpeed': ['min', 'mean', 'max', 'std'],
                'Roll': ['min', 'mean', 'max', 'std'],
                'Pitch': ['min', 'mean', 'max', 'std'],
                'SOG': ['min', 'mean', 'max', 'std'],
                'Crew': lambda x: x.iloc[0],
                'Date': lambda x: x.iloc[0],
                'run_length': lambda x: x.iloc[0],
                'Forestay': lambda x: x.iloc[0],
                'Team_weight': lambda x: x.iloc[0],
                'Focus': lambda x: x.iloc[0],
                'Wind': lambda x: x.iloc[0],
                'Waves': lambda x: x.iloc[0],
                'Current': lambda x: x.iloc[0],
                'Rating': lambda x: x.iloc[0],
                'ident': lambda x: x.iloc[0],
                'avg_WindDirection': lambda x: x.iloc[0],
                'avg_TWA': lambda x: x.iloc[0],
                'Tack': lambda x: x.iloc[0],
                'Type': lambda x: x.iloc[0],
                'Area': lambda x: x.iloc[0],
                'avg_COG': lambda x: x.iloc[0],
            }

            # Apply the aggregation functions to the grouped DataFrame
            grouped_seg_data_agg = grouped_seg_data.agg(agg_funcs)

            # Flatten the column names to make them easier to work with
            grouped_seg_data_agg.columns = ['_'.join(col).strip() for col in grouped_seg_data_agg.columns.values]

            # Reset the index to make the session_id and stream_id columns regular columns
            grouped_seg_data_agg = grouped_seg_data_agg.reset_index()

            # Rename the crew and date columns
            grouped_seg_data_agg = grouped_seg_data_agg.rename(
                columns={'Crew_<lambda>': 'Crew', 'Date_<lambda>': 'Date', 'run_length_<lambda>': 'run_length',
                         'Forestay_<lambda>': 'Forestay', 'Team_weight_<lambda>': 'Team_weight', 'Focus_<lambda>': 'Focus',
                         'Wind_<lambda>': 'Wind', 'Waves_<lambda>': 'Waves', 'Current_<lambda>': 'Current',
                         'Rating_<lambda>': 'Rating', 'ident_<lambda>': 'ident', 'avg_WindDirection_<lambda>':
                             'avg_WindDirection', 'avg_TWA_<lambda>': 'avg_TWA', 'Tack_<lambda>':
                             'Tack', 'Type_<lambda>': 'Type', 'Area_<lambda>': 'Area', 'avg_COG_<lambda>': 'avg_COG'})

            if "grouped_seg_data_agg_sel" not in st.session_state:
                st.session_state.grouped_seg_data_agg_sel = []

            grouped_seg_data_agg_opt = grouped_seg_data_agg.columns.tolist()

            grouped_seg_data_agg_opt = sorted(grouped_seg_data_agg_opt)

            # grouped_seg_data_agg_default_options = ["vmg_min", "Crew"]

            grouped_seg_data_agg_sel = st.multiselect("Data shown", grouped_seg_data_agg_opt, default=st.session_state.grouped_seg_data_agg_sel)

            st.session_state.grouped_seg_data_agg_sel = grouped_seg_data_agg_sel

            st.write(grouped_seg_data_agg[grouped_seg_data_agg_sel])

        elif oversigt_sel == 'By Session':
            grouped_session_data = df.groupby(['session_id'])

            agg_funcs = {
                'vmg': ['min', 'mean', 'max', 'std'],
                'WindSpeed': ['min', 'mean', 'max', 'std'],
                'Roll': ['min', 'mean', 'max', 'std'],
                'TWA': ['min', 'mean', 'max', 'std'],
                'Pitch': ['min', 'mean', 'max', 'std'],
                'Yaw': ['min', 'mean', 'max', 'std'],
                'SOG': ['min', 'mean', 'max', 'std'],
                'COG': ['min', 'mean', 'max', 'std'],
                'WindDirection': ['min', 'mean', 'max', 'std'],
                'Crew': lambda x: x.iloc[0],
                'Date': lambda x: x.iloc[0],
                'run_length': lambda x: x.iloc[0],
                'Forestay': lambda x: x.iloc[0],
                'Team_weight': lambda x: x.iloc[0],
                'Focus': lambda x: x.iloc[0],
                'Wind': lambda x: x.iloc[0],
                'Waves': lambda x: x.iloc[0],
                'Current': lambda x: x.iloc[0],
                'Rating': lambda x: x.iloc[0],
                'ident': lambda x: x.iloc[0],
            }

            # Apply the aggregation functions to the grouped DataFrame
            grouped_session_data_agg = grouped_session_data.agg(agg_funcs)

            # Flatten the column names to make them easier to work with
            grouped_session_data_agg.columns = ['_'.join(col).strip() for col in grouped_session_data_agg.columns.values]

            # Reset the index to make the session_id and stream_id columns regular columns
            grouped_session_data_agg = grouped_session_data_agg.reset_index()

            # Rename the crew and date columns
            grouped_session_data_agg = grouped_session_data_agg.rename(
                columns={'Crew_<lambda>': 'Crew', 'Date_<lambda>': 'Date', 'run_length_<lambda>': 'run_length',
                         'Forestay_<lambda>': 'Forestay', 'Team_weight_<lambda>': 'Team_weight',
                         'Focus_<lambda>': 'Focus',
                         'Wind_<lambda>': 'Wind', 'Waves_<lambda>': 'Waves', 'Current_<lambda>': 'Current',
                         'Rating_<lambda>': 'Rating', 'ident_<lambda>': 'ident'})

            if "grouped_session_data_agg_sel" not in st.session_state:
                st.session_state.grouped_session_data_agg_sel = []

            grouped_session_data_agg_opt = grouped_session_data_agg.columns.tolist()
            grouped_session_data_agg_opt = sorted(grouped_session_data_agg_opt)

            # grouped_seg_data_agg_default_options = ["vmg_min", "Crew"]

            grouped_session_data_agg_sel = st.multiselect("Data shown", grouped_session_data_agg_opt,
                                                      default=st.session_state.grouped_session_data_agg_sel)

            st.session_state.grouped_session_data_agg_sel = grouped_session_data_agg_sel

            st.write(grouped_session_data_agg[grouped_session_data_agg_sel])




        elif oversigt_sel == 'just show me everythang':
            st.write(df)

    with evaluation:

        Eval_opt = ['Overview', 'Boat Management', 'TWA Management']
        Eval_sel = st.radio('Evaluation options:', options=Eval_opt, horizontal=True)

        ChangeWidgetFontSize('Evaluation options:', '26px')

        if Eval_sel == 'Boat Management':
            res = SM_handling.boatmanagement_vmg_pred(df)

            # st.write(res)

            graph_opt = ['All'] + res.ident.unique().tolist()

            unique_ident_length = len(res.ident.unique().tolist())

            if unique_ident_length == 1:
                graph_sel = res.ident.unique().tolist()[0]
            else:
                graph_sel = st.selectbox('Options:', options=graph_opt)

            if graph_sel == 'All':
                axis_opt = ['vmg', 'pVMG', 'BoatManagementScore']

                yaxis_sel2 = st.selectbox('Y-axis:', axis_opt, index=axis_opt.index('BoatManagementScore'))

                # create the slider using the select_slider function
                smoothingBMS = st.select_slider('Smoothing:', options=range(1, 31), key=3)

                # apply smoothing to the y-axis data using a rolling mean
                df[yaxis_sel2] = df[yaxis_sel2].rolling(smoothingBMS).mean()

                all_BMS = px.line(df, x='rel_time', y=yaxis_sel2, color='ident')

                st.plotly_chart(all_BMS)
            elif graph_sel != 'All':
                chosen_segment = df.loc[df['ident'] == graph_sel, :]

                # create the slider using the select_slider function
                smoothingSpecBMS = st.select_slider('Smoothing:', options=range(1, 31), key=4)

                # apply smoothing to the y-axis data using a rolling mean
                chosen_segment['vmg2'] = chosen_segment['vmg'].rolling(smoothingSpecBMS).mean()
                chosen_segment['pVMG2'] = chosen_segment['pVMG'].rolling(smoothingSpecBMS).mean()
                chosen_segment['BoatManagementScore2'] = chosen_segment['BoatManagementScore'].rolling(smoothingSpecBMS).mean()

                specific_BMS = px.line(chosen_segment, x='rel_time', y=['vmg2', 'pVMG2', 'BoatManagementScore2'],
                                       title='VMG vs. VMG Predict vs. Boat Management Score')
                st.plotly_chart(specific_BMS)

        elif Eval_sel == 'TWA Management':
            res = SM_handling.TWAmanagement_vmg_pred(df)

            # st.write(res)

            graph_opt = ['All'] + res.ident.unique().tolist()

            unique_ident_length = len(res.ident.unique().tolist())

            if unique_ident_length == 1:
                graph_sel = res.ident.unique().tolist()[0]
            else:
                graph_sel = st.selectbox('Options:', options=graph_opt)

            if graph_sel == 'All':
                axis_opt = ['vmg', 'p_TWA_VMG', 'TWAManagementScore']

                yaxis_sel2 = st.selectbox('Y-axis:', axis_opt, index=axis_opt.index('TWAManagementScore'))

                # create the slider using the select_slider function
                smoothingBMS = st.select_slider('Smoothing:', options=range(1, 31), key=3)

                # apply smoothing to the y-axis data using a rolling mean
                df[yaxis_sel2] = df[yaxis_sel2].rolling(smoothingBMS).mean()

                all_BMS = px.line(df, x='rel_time', y=yaxis_sel2, color='ident')

                st.plotly_chart(all_BMS)
            elif graph_sel != 'All':
                chosen_segment = df.loc[df['ident'] == graph_sel, :]

                # create the slider using the select_slider function
                smoothingSpecBMS = st.select_slider('Smoothing:', options=range(1, 31), key=4)

                # apply smoothing to the y-axis data using a rolling mean
                chosen_segment['vmg2'] = chosen_segment['vmg'].rolling(smoothingSpecBMS).mean()
                chosen_segment['p_TWA_VMG2'] = chosen_segment['p_TWA_VMG'].rolling(smoothingSpecBMS).mean()
                chosen_segment['TWAManagementScore2'] = chosen_segment['TWAManagementScore'].rolling(
                    smoothingSpecBMS).mean()

                specific_BMS = px.line(chosen_segment, x='rel_time', y=['vmg2', 'p_TWA_VMG2', 'TWAManagementScore2'],
                                       title='VMG vs. VMG Predict vs. TWA Management Score')
                st.plotly_chart(specific_BMS)

        elif Eval_sel == 'Overview':
            grouped_seg_data = df.groupby(['session_id', 'stream_id'])

            if len(grouped_seg_data) > 10000:
                st.warning('Too many segments, please refine search to max. 10 segments', icon="🧮")
            else:

                res = SM_handling.boatmanagement_vmg_pred(df)
                res = SM_handling.TWAmanagement_vmg_pred(res)

                res['Total Score'] = 0.5 * res['BoatManagementScore'] + 0.5 * res['TWAManagementScore']

                agg_funcs = {
                    'Total Score': ['mean'],
                    'BoatManagementScore': ['mean'],
                    'TWAManagementScore': ['mean'],
                    'vmg': ['mean'],
                    'WindSpeed': ['mean'],
                    'ident': lambda x: x.iloc[0],
                }

                # Apply the aggregation functions to the grouped DataFrame
                grouped_seg_data_agg = grouped_seg_data.agg(agg_funcs)

                # Flatten the column names to make them easier to work with
                grouped_seg_data_agg.columns = ['_'.join(col).strip() for col in grouped_seg_data_agg.columns.values]

                # Reset the index to make the session_id and stream_id columns regular columns
                grouped_seg_data_agg = grouped_seg_data_agg.reset_index()

                # Rename the crew and date columns
                grouped_seg_data_agg = grouped_seg_data_agg.rename(
                    columns={'BoatManagementScore_mean': 'BMS_mean', 'TWAManagementScore_mean': 'TWAS_mean',
                             'Rating_<lambda>': 'Rating', 'ident_<lambda>': 'ident'})

                st.write(grouped_seg_data_agg[['ident', 'Total Score_mean', 'BMS_mean', 'TWAS_mean']])



elif options == 'Home':
    # Load an image from a file path
    image = 'PVAP.png'

    # Display the image in the Streamlit app
    st.image(image, use_column_width=True)

    st.write('Contact: johan.langballe@outlook.com or nikolaj.hougaard1999@gmail.com')

    ChangeWidgetFontSize('Contact: johan.langballe@outlook.com or nikolaj.hougaard1999@gmail.com', '12px')

elif options == 'Upload':
    st.header("Upload Files")

    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0

    uploaded_csv_file_1 = st.file_uploader(
        "Upload Sailmon CSV file",
        type=["csv"],
        key=f"csv1_{st.session_state.file_uploader_key}"
    )
    uploaded_csv_file_2 = st.file_uploader(
        "Upload WindBot CSV file",
        type=["csv"],
        key=f"csv2_{st.session_state.file_uploader_key}"
    )

    if uploaded_csv_file_1 and uploaded_csv_file_2:
        st.success("All files have been uploaded successfully!")

        # Read and display the content of the first CSV file
        df_csv_1 = pd.read_csv(uploaded_csv_file_1)
        # st.write("Content of the first CSV file:")
        # st.write(df_csv_1)

        # Read and display the content of the second CSV file
        df_csv_2 = pd.read_csv(uploaded_csv_file_2)
        # st.write("Content of the second CSV file:")
        # st.write(df_csv_2)

        # load the trained model from a file
        SegPredModel = joblib.load('BIGMODE.joblib')

        SM = SM_handling.upload_session_step1(SM_Rawfile=df_csv_1, Wind_Rawfile=df_csv_2, _SegmentPredictionmodel=SegPredModel)

        vmg_min = st.number_input('Minimum VMG Average:', value=4.0, step=0.1)
        length_min = st.number_input('Minimum Segment Length', value=20, step=5)

        SM = SM_handling.join_rolling_lines(SM, vmg=vmg_min, length=length_min)

        # Drop negative wind if happens
        grouped_means = SM.groupby('stream_id')['WindSpeed'].transform('mean')
        SM = SM[grouped_means >= 0]



        # create a list of options for the dropdown
        color_opt = dir(px.colors.sequential)
        color_opt = [item for item in color_opt if '__' not in item and 'swatches' not in item]

        # create the dropdown using the selectbox function
        color_sel = st.selectbox('Color:', color_opt, index=color_opt.index('Turbo'))

        GPSplot = px.scatter_mapbox(SM, lat="latitude", lon="longitude", color="vmg",
                                    color_continuous_scale=color_sel)
        GPSplot.update_layout(mapbox_style="carto-positron", mapbox_zoom=12,
                              margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(GPSplot)

        st.header("Create Log")
        old_new_crew_opt = ['Old', 'New']
        old_new_crew_sel = st.radio('Crew:', options=old_new_crew_opt, horizontal=True,
                                    index=old_new_crew_opt.index('Old'))

        if old_new_crew_sel == 'New':
            Crew = st.text_input("Type crew:")
        elif old_new_crew_sel == 'Old':
            Crew = st.radio('Select crew', options=SM_handling.fetch_crew(config=config),
                            horizontal=True)

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            current_opt = ['0 - None', '1', '2 - Worst']
            current_sel = st.radio('Current', options=current_opt)

        with col2:
            wave_opt = ['0 - None', '1', '2', '3 - Worst']
            wave_sel = st.radio('Waves', options=wave_opt)

        with col3:
            rating_opt = ['1 - Worst', '2', '3', '4', '5 - Best']
            rating_sel = st.radio('Rating', options=rating_opt)

        with col4:
            Focus_opt = ['Race', 'Other']
            Focus_sel = st.radio('Focus:', options=Focus_opt, index=Focus_opt.index('Other'))

        with col5:
            Wind_opt = ['Onshore', 'Offshore']
            Wind_sel = st.radio('Focus:', options=Wind_opt, index=Wind_opt.index('Onshore'))

        with col6:
            Forestay = st.number_input('Forestay length:', value=430, step=5)
            Team_weight = st.number_input('Team weight:', value=165, step=1)

        col7, col8 = st.columns(2)

        with col7:
            Wind_range_min = st.number_input('Minimum wind range:', value=5, step=1)

        with col8:
            Wind_range_max = st.number_input('Maximum wind range:', value=15, step=1)

        notes = st.text_input("Notes:")

        if type(current_sel) != int:
            match = re.search(r'\d+', current_sel)
            current_sel = int(match.group())

        if type(wave_sel) != int:
            match = re.search(r'\d+', wave_sel)
            wave_sel = int(match.group())

        if type(rating_sel) != int:
            match = re.search(r'\d+', rating_sel)
            rating_sel = int(match.group())

        session_overview = pd.DataFrame([Crew, Forestay, Team_weight, Focus_sel, Wind_sel, Wind_range_min,
                                         Wind_range_max, wave_sel, current_sel, rating_sel, notes],
                                        ['Crew', 'Forestay', 'Team_weight', 'Focus', 'Wind', 'Wind_range_min',
                                         'Wind_range_max', 'Waves', 'Current', 'Rating', 'Notes'])

        session_overview = session_overview.transpose()

        delta = SM.time.iloc[-1] - SM.time.iloc[0]

        session_overview['Date'] = SM.time.iloc[0].date()
        session_overview['session_length'] = delta.total_seconds()
        session_overview['Median_Wind'] = SM.WindSpeed.median()

        coords = (SM.latitude.iloc[0], SM.longitude.iloc[0])
        session_overview['Area'] = SM_handling.nearest_area(coords)

        if_exist_opt = ['append', 'replace', 'fail']
        if_exist_sel = st.selectbox(label='If exist behavior:', options=if_exist_opt)

        if if_exist_sel == 'replace':
            Session_id = 1
            common_rows = []
        else:
            Session_id = max(SM_handling.fetch_session_id(config=config)) + 1
            Test_exist = SM_handling.fetch_crew_date_list(config=config)

            Test_new = pd.DataFrame([session_overview['Crew'], session_overview['Date']]).transpose()

            # st.write(Test_exist)
            # st.write(Test_new)

            # Assuming df1 and df2 are your dataframes
            merged_df = Test_exist.merge(Test_new, how='left', indicator=True)

            # Filter the rows that are present in both dataframes
            common_rows = merged_df[merged_df['_merge'] == 'both']

        session_overview['session_id'] = Session_id

        run_overview, SM = SM_handling.make_RO_and_Prep_SM(SM, Session_id)

        st.write(run_overview)

        st.header('Session ID: ' + str(Session_id))

        # Check if there are any common rows
        if len(common_rows) > 0:
            st.warning('Warning, this combination of Date and Crew already exists in the database!', icon="⚠️")
            if st.button("Ok", key=5):
                # Create the button
                if st.button("Upload Session to server", key=4):
                    SM_handling.upload_data(File=run_overview, Type="run_overview", connection_string=connection_string,
                                            ifexist=if_exist_sel)
                    SM_handling.upload_data(File=session_overview, Type="session_overview", connection_string=connection_string,
                                            ifexist=if_exist_sel)
                    SM_handling.upload_data(File=SM, Type="saildata", connection_string=connection_string,
                                            ifexist=if_exist_sel)
        else:
            if st.button("Upload Session to server", key=2):
                SM_handling.upload_data(File=run_overview, Type="run_overview", connection_string=connection_string,
                                        ifexist=if_exist_sel)
                SM_handling.upload_data(File=session_overview, Type="session_overview",
                                        connection_string=connection_string,
                                        ifexist=if_exist_sel)
                SM_handling.upload_data(File=SM, Type="saildata", connection_string=connection_string,
                                        ifexist=if_exist_sel)
                uploaded_csv_file_1 = None
                uploaded_csv_file_2 = None
                st.session_state.file_uploader_key += 1
                st.experimental_rerun()
        if st.button("Clear and reload", key=1):
            uploaded_csv_file_1 = None
            uploaded_csv_file_2 = None
            st.session_state.file_uploader_key += 1
            st.experimental_rerun()





elif options == 'Draft':
    # Function to update the 'options' variable
    def my_function():
        st.session_state.current_page = 'Home'


    # Create the button
    if st.button("Click me!"):
        my_function()
        st.experimental_rerun()

    options = ['Option 1', 'Option 2', 'Option 3']

    selected_options = []

    for option in options:
        if st.checkbox(option):
            selected_options.append(option)

    st.write(f"You have selected: {selected_options}")

    import streamlit as st

    options = ['Option 1', 'Option 2', 'Option 3']

    smoothing = st.select_slider('Smoothing:', options=options)

    st.write(f"You have selected: {selected_options}")

# Remove the firewall rule after your script is done
# sql_client.firewall_rules.delete(resource_group_name, server_name, firewall_rule_name)
st.stop()
