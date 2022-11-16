import json

import pytz
import plotly
import requests
import sqlalchemy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import HTTPException
from plotly.subplots import make_subplots
from postgres.crud import hour_convert_scada, hour_convert_forecast
from postgres.database import ScadaSessionLocal, DataSessionLocal


def server_status(url):
    status = requests.get(url).status_code
    if status == 200:
        code = 'Active'
    else:
        code = 'Unable to connect to server'

    return code


def calc_frame(dataframe):
    dataframe['Date'] = dataframe['Date'].apply(lambda x: x.strftime("%Y-%m-%d"))
    dataframe.set_index('Date', inplace=True)
    dataframe['NMAE_MEAN'] = dataframe.apply(lambda x: np.round(np.mean(x), 3), axis=1)
    dataframe.loc['NMAE_MEAN'] = dataframe.apply(lambda x: np.round(np.mean(x), 3))
    dataframe = dataframe.apply(lambda x: np.round(x, 3))
    dataframe.reset_index(inplace=True)
    dataframe.sort_values('Date', inplace=True)

    return dataframe


def main_data(data_list: list, date: str, now: str):
    dataframe = pd.DataFrame()
    pdataframe = pd.DataFrame()
    for n, label in enumerate(data_list):
        try:
            all_frame = hour_frame(label, date, now)
            all_frame['Date'] = pd.to_datetime(all_frame['Date'])
            all_frame.set_index('Date', inplace=True)
            all_frame.rename(columns={'NMAE': label}, inplace=True)
            production_frame = all_frame.query('actual >= 200')
            all_frame = all_frame.resample('1D').mean()
            production_frame = production_frame.resample('1D').mean()
            all_frame.dropna(inplace=True)
            production_frame.dropna(inplace=True)

            all_frame = all_frame[[label]]
            production_frame = production_frame[[label]]
            all_frame.reset_index(inplace=True)
            production_frame.reset_index(inplace=True)

            if n == 0:
                dataframe = pd.concat([dataframe, all_frame], axis=1)
                pdataframe = pd.concat([pdataframe, production_frame], axis=1)
            else:
                dataframe = pd.merge(dataframe, all_frame, on='Date', how='outer')
                pdataframe = pd.merge(pdataframe, production_frame, on='Date', how='outer')

        except sqlalchemy.exc.ProgrammingError:
            raise HTTPException(status_code=404, detail='Input data is fault')

    # todo: 발전기 추가 시 해당 바꿔야 할 곳 (2000 -> 각 발전기별 CAPACITY)
    dataframe = calc_frame(dataframe)
    pdataframe = calc_frame(pdataframe)

    return dataframe, pdataframe


def hour_frame(data: str, start: str, end: str):
    with ScadaSessionLocal() as db:
        scada = hour_convert_scada(db, data, start, end)

    with DataSessionLocal() as db:
        forecast = hour_convert_forecast(db, data, start, end)

    scada_df = pd.DataFrame([*scada])
    forecast_df = pd.DataFrame([*forecast])
    timezone = pytz.timezone('utc')
    forecast_df['ds'] = forecast_df['ds'].apply(lambda x: timezone.localize(x, is_dst=None))

    dataframe = pd.merge(scada_df, forecast_df, on='ds', how='outer')
    # todo: 발전기 추가 시 해당 바꿔야 할 곳 (2000 -> 각 발전기별 CAPACITY)
    dataframe['NMAE'] = dataframe.apply(lambda x: np.abs(np.subtract(x['forecast'], x['actual'])) / 2000 * 100, axis=1)
    dataframe.rename(columns={'ds': 'Date'}, inplace=True)
    dataframe['Date'] = dataframe['Date'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    dataframe.set_index('Date', inplace=True)
    dataframe = dataframe.apply(lambda x: np.round(x, 3))
    dataframe.reset_index(inplace=True)

    return dataframe


def generator_frame(data: str, st: str, ed: str):
    dataframe = hour_frame(data, st, ed)
    # todo: 발전기 추가 시 해당 바꿔야 할 곳 (200 -> 각 발전기 CAPACITY * 0.1)
    error_rate_10 = dataframe.query('actual >= 200')
    # 행열 추가
    error_rate_10.set_index('Date', inplace=True)
    error_rate_10.loc['NMAE_MEAN'] = error_rate_10.apply(lambda x: np.round(np.mean(x), 3))
    error_rate_10.reset_index(inplace=True)

    if not dataframe.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['actual'], name='actual', mode='lines'))
        fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['forecast'], name='forecast', mode='lines'))
        fig.add_trace(go.Bar(x=dataframe['Date'], y=dataframe['NMAE'], name='NMAE', opacity=0.3), secondary_y=True)
        fig.add_trace(go.Bar(x=error_rate_10['Date'], y=error_rate_10['NMAE'], name='NMAE_10', opacity=0.3),
                      secondary_y=True)

        fig.update_traces(hovertemplate=None)
        fig.update_layout(title=f'{data}: {st} - {ed}', hovermode="x unified", xaxis=dict(title='DateTime'),
                          yaxis=dict(title='Power(Kwh)'), yaxis2=dict(title='NMAE(%)', range=[0, 100], showgrid=False),
                          legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="right", x=0.95),
                          margin=dict(l=0, r=0, b=0, t=30))
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        graph_json = 'Empty Database'

    return dataframe, graph_json, error_rate_10
