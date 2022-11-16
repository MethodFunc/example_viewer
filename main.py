import json
import warnings
from datetime import datetime

import plotly.graph_objects as go
import plotly.utils
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from bin.tools import server_status, main_data, generator_frame

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

APIURL = 'APIURL'
MLFLOWURL = 'MLFLOWURL'
DATEFORMAT = '%Y-%m-%d'

api = server_status(APIURL)
mlflow = server_status(MLFLOWURL)
# 명칭 리스트 적기
data_list = [f'DB_HJ{i:02d}' for i in range(1, 16)]


@app.get('/dashboard', response_class=HTMLResponse)
async def index(request: Request):
    now = datetime.now()
    date = datetime(now.year, now.month, day=1).strftime(DATEFORMAT)
    now = now.strftime(DATEFORMAT)
    nmae_df, production = main_data(data_list, date, now)

    if not nmae_df.empty:
        fig = go.Figure()
        for label in data_list:
            fig.add_trace(go.Bar(x=nmae_df['Date'], y=nmae_df[label], name=label))

        fig.update_traces(hovertemplate=None)
        fig.update_layout(title=date[:7], hovermode="x unified", xaxis=dict(title='DateTime'),
                          yaxis=dict(title='NMAE(%)'),
                          legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="right", x=0.95),
                          margin=dict(l=0, r=0, b=0, t=30))

        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    else:
        graph_json = 'Empty Database'

    return templates.TemplateResponse('dashboard.html',
                                      {'request': request, 'result': graph_json, 'tables': nmae_df, 'status': api,
                                       'mlflow': mlflow, 'data_list': data_list, 'tables2': production})


@app.get("/generator/{data}", response_class=HTMLResponse)
def gen(request: Request, data: str):
    now = datetime.now()
    st = datetime(year=now.year, month=now.month, day=now.day - 1).strftime(DATEFORMAT)
    ed = datetime(year=now.year, month=now.month, day=now.day).strftime(DATEFORMAT)
    dataframe, graph_json, error_rate = generator_frame(data, st, ed)

    return templates.TemplateResponse('items.html',
                                      {'request': request, 'data': data, 'status': api,
                                       'mlflow': mlflow, 'data_list': data_list,
                                       'result': graph_json,
                                       'tables': dataframe, 'tables2': error_rate})


@app.post("/generator/{data}", response_class=HTMLResponse)
def search(request: Request, data: str, start: str = Form(...), end: str = Form(...)):
    dataframe, graph_json, error_rate = generator_frame(data, start, end)

    return templates.TemplateResponse('items.html',
                                      {'request': request, 'data': data, 'status': api,
                                       'mlflow': mlflow, 'data_list': data_list, 'result': graph_json,
                                       'tables': dataframe, 'tables2': error_rate})


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8123, reload=True, debug=True)
