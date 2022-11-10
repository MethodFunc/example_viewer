import re
from datetime import datetime
from datetime import timedelta
from typing import Optional

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from postgres import schemas, database, models


def utc_convert(start_time: str, end_time: Optional[str] = None):
    pattern = r'[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]'
    start_time = re.sub(pattern, '', start_time)
    start_time = datetime.strptime(start_time, "%Y%m%d")
    start_time = (start_time - timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
    if end_time is not None:
        end_time = re.sub(pattern, '', end_time)
        end_time = datetime.strptime(end_time, "%Y%m%d")
        end_time = (end_time - timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")

        return start_time, end_time

    return start_time


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
    start_time = utc_convert(start_time)
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
    start_time, end_time = utc_convert(start_time, end_time)
    table = models.create_forcast(table_name)
    result = db.query(func.date_trunc('hour', func.timezone('KST', func.timezone('UTC', table.record_date))) \
                      .label('ds'), func.avg(table.forecast).label('forecast')) \
        .order_by(text('1')) \
        .group_by(text('1')) \
        .filter(table.record_date >= start_time) \
        .filter(table.record_date < end_time) \
        .all()

    return result
