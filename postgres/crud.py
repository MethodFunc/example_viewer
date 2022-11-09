import re

from sqlalchemy.orm import Session
from postgres import models
from datetime import datetime, timedelta
from sqlalchemy import func, text

pattern = r'[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]'


def day_convert_scada(db: Session, table_name: str, start_time: str):
    table = models.create_models(table_name)
    result = db.query(func.date_trunc('day', func.timezone('UTC', table.record_date)) \
                      .label('ds'), func.avg(table.active_power).label(table_name)) \
        .order_by(text('1')) \
        .group_by(text('1')) \
        .filter(table.record_date >= start_time) \
        .all()

    return result


def day_convert_forecast(db: Session, table_name: str, start_time: str):
    table = models.create_forcast(table_name)
    result = db.query(func.date_trunc('day', func.timezone('KST', func.timezone('UTC', table.record_date))) \
                      .label('ds'), func.avg(table.forecast).label(table_name)) \
        .order_by(text('1')) \
        .group_by(text('1')) \
        .filter(table.record_date >= start_time) \
        .all()

    return result


def hour_convert_scada(db: Session, table_name: str, start_time: str, end_time: str):
    table = models.create_models(table_name)
    result = db.query(func.date_trunc('hour', func.timezone('UTC', table.record_date)) \
                      .label('ds'), func.avg(table.active_power).label('actual')) \
        .order_by(text('1')) \
        .group_by(text('1')) \
        .filter(table.record_date >= start_time) \
        .filter(table.record_date < end_time) \
        .all()

    return result


def hour_convert_forecast(db: Session, table_name: str, start_time: str, end_time: str):
    start_time = re.sub(pattern, '', start_time)
    end_time = re.sub(pattern, '', end_time)
    start_time = datetime.strptime(start_time, "%Y%m%d")
    end_time = datetime.strptime(end_time, "%Y%m%d")
    end_time = (end_time - timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
    start_time = (start_time - timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
    table = models.create_forcast(table_name)
    result = db.query(func.date_trunc('hour', func.timezone('KST', func.timezone('UTC', table.record_date))) \
                      .label('ds'), func.avg(table.forecast).label('forecast')) \
        .order_by(text('1')) \
        .group_by(text('1')) \
        .filter(table.record_date >= start_time) \
        .filter(table.record_date < end_time) \
        .all()

    return result
