import json
from datetime import datetime

import numpy as np
import plotly.utils
import pytz
import sqlalchemy.exc
import uvicorn

import requests
import pandas as pd

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from postgres.database import ScadaSessionLocal, DataSessionLocal
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from postgres.crud import day_convert_scada, day_convert_forecast, hour_convert_scada, hour_convert_forecast
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings

warnings.filterwarnings('ignore')

app = FastAPI()


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if (request.url.path.startswith("/v1") and
            request.headers.get('X-Token', None) != "expected_token"):
        return JSONResponse(status_code=403)
    response = await call_next(request)
    return response


app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory="templates")


def server_status(url):
    status = requests.get(url).status_code
    if status == 200:
        code = 'Active'
    else:
        code = 'Unable to connect to server'

    return code


def hour_frame(data, start, end):
    with ScadaSessionLocal() as db:
        scada = hour_convert_scada(db, data, start, end)

    with DataSessionLocal() as db:
        forecast = hour_convert_forecast(db, data, start, end)

    scada_df = pd.DataFrame([*scada])
    forecast_df = pd.DataFrame([*forecast])
    timezone = pytz.timezone('utc')
    forecast_df['ds'] = forecast_df['ds'].apply(lambda x: timezone.localize(x, is_dst=None))

    merge_df = pd.merge(scada_df, forecast_df, on='ds', how='outer')
    merge_df['NMAE'] = merge_df.apply(lambda x: np.abs(np.subtract(x['forecast'], x['actual'])) / 2000 * 100, axis=1)
    merge_df.rename(columns={'ds': 'Date'}, inplace=True)
    merge_df['Date'] = merge_df['Date'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    merge_df.set_index('Date', inplace=True)
    merge_df = merge_df.apply(lambda x: np.round(x, 3))
    merge_df.reset_index(inplace=True)

    return merge_df


api_url = 'api_url'
mlflow_url = 'mlflow_url'
api = server_status(api_url)
mlflow = server_status(mlflow_url)
data_list = [f'DB_HJ{i:02d}' for i in range(1, 16)]


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    date = '2022-11-01'
    scada_df = pd.DataFrame()
    forecast_df = pd.DataFrame()
    for n, label in enumerate(data_list):
        try:
            with ScadaSessionLocal() as db:
                scada = day_convert_scada(db, label, date)

            with DataSessionLocal() as db:
                forecast = day_convert_forecast(db, label, date)

            if n == 0:
                scada_df = pd.concat([scada_df, pd.DataFrame([*scada])], axis=1)
                forecast_df = pd.concat([forecast_df, pd.DataFrame([*forecast])], axis=1)
            else:
                scada_df = pd.merge(scada_df, pd.DataFrame([*scada]), on='ds', how='outer')
                forecast_df = pd.merge(forecast_df, pd.DataFrame([*forecast]), on='ds', how='outer')

        except sqlalchemy.exc.ProgrammingError:
            raise HTTPException(status_code=404, detail='Input data is fault')

    scada_df['ds'] = scada_df['ds'].astype('str')
    forecast_df['ds'] = forecast_df['ds'].astype('str')

    scada_df['ds'] = scada_df['ds'].astype('str')
    forecast_df['ds'] = forecast_df['ds'].astype('str')

    scada_df['ds'] = scada_df['ds'].apply(lambda x: x[:10])

    scada_df.set_index('ds', inplace=True)
    forecast_df.set_index('ds', inplace=True)

    nmae_df = abs(forecast_df - scada_df) / 2000 * 100
    nmae_df.dropna(inplace=True)
    nmae_df['NMAE_MEAN'] = nmae_df.apply(lambda x: np.mean(x), axis=1)
    nmae_df.loc['NMAE_MEAN'] = nmae_df.apply(lambda x: np.mean(x))
    nmae_df = nmae_df.apply(lambda x: np.round(x, 3))
    nmae_df.reset_index(inplace=True)
    nmae_df.rename(columns={'ds': 'Date'}, inplace=True)

    if not nmae_df.empty:
        fig = go.Figure()
        for label in data_list:
            fig.add_trace(go.Bar(x=nmae_df['Date'], y=nmae_df[label], name=label))

        fig.update_traces(hovertemplate=None)
        fig.update_layout(title=date[:7], hovermode="x unified", xaxis=dict(title='DateTime'),
                          yaxis=dict(title='NMAE(%)'),
                          legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="right", x=0.95),
                          margin=dict(l=0, r=0, b=0, t=30))

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    else:
        graphJSON = 'Empty Database'

    return templates.TemplateResponse('dashboard.html',
                                      {'request': request, 'result': graphJSON, 'tables': nmae_df, 'status': api,
                                       'mlflow': mlflow, 'data_list': data_list})


@app.get("/generator/{data}", response_class=HTMLResponse)
def gen(request: Request, data: str):
    now = datetime.now()
    st = datetime(year=now.year, month=now.month, day=now.day - 1).strftime("%Y-%m-%d")
    ed = datetime(year=now.year, month=now.month, day=now.day).strftime("%Y-%m-%d")
    merge_df = hour_frame(data, st, ed)

    if not merge_df.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=merge_df['Date'], y=merge_df['actual'], name='actual', mode='lines'))
        fig.add_trace(go.Scatter(x=merge_df['Date'], y=merge_df['forecast'], name='forecast', mode='lines'))
        fig.add_trace(go.Bar(x=merge_df['Date'], y=merge_df['NMAE'], name='NMAE', opacity=0.3), secondary_y=True)
        fig.update_traces(hovertemplate=None)
        fig.update_layout(title=f'{data}: {st} - {ed}', hovermode="x unified", xaxis=dict(title='DateTime'),
                          yaxis=dict(title='Power(Kwh)'), yaxis2=dict(title='NMAE(%)', range=[0, 100], showgrid=False),
                          legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="right", x=0.95),
                          margin=dict(l=0, r=0, b=0, t=30))
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        graphJSON = 'Empty Database'

    return templates.TemplateResponse('items.html',
                                      {'request': request, 'data': data, 'status': api,
                                       'mlflow': mlflow, 'data_list': data_list,
                                       'result': graphJSON,
                                       'tables': merge_df})


@app.post("/generator/{data}", response_class=HTMLResponse)
def search(request: Request, data: str, start: str = Form(...), end: str = Form(...)):
    merge_df = hour_frame(data, start, end)

    if not merge_df.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=merge_df['Date'], y=merge_df['actual'], name='actual', mode='lines'))
        fig.add_trace(go.Scatter(x=merge_df['Date'], y=merge_df['forecast'], name='forecast', mode='lines'))
        fig.add_trace(go.Bar(x=merge_df['Date'], y=merge_df['NMAE'], name='NMAE', opacity=0.3), secondary_y=True)
        fig.update_traces(hovertemplate=None)
        fig.update_layout(title=f'{data}: {start} - {end}', hovermode="x unified", xaxis=dict(title='DateTime'),
                          yaxis=dict(title='Power(Kwh)'), yaxis2=dict(title='NMAE(%)', range=[0, 100], showgrid=False),
                          legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="right", x=0.95),
                          margin=dict(l=0, r=0, b=0, t=30))
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        graphJSON = 'Empty Database'

    return templates.TemplateResponse('items.html',
                                      {'request': request, 'data': data, 'status': api,
                                       'mlflow': mlflow, 'data_list': data_list, 'result': graphJSON,
                                       'tables': merge_df})


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8123, reload=True, debug=True)
