"""Database and Minio models."""
import datetime as _dt
from dataclasses import dataclass
from enum import Enum

import sqlalchemy as _sql

import db as _database


@dataclass
class Users(_database.Base):
    """Users model. Inherit from the base class to create a table in the database."""

    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}
    id = _sql.Column(_sql.Integer, primary_key=True, index=True)
    first_name = _sql.Column(_sql.String, index=True)
    last_name = _sql.Column(_sql.String, index=True)
    password = _sql.Column(_sql.String, index=True)
    email = _sql.Column(_sql.String, index=True, unique=True)
    data_created = _sql.Column(_sql.DateTime, default=_dt.datetime.utcnow)


@dataclass
class Coins(_database.Base):
    """Coins informations. Contains the informations about the coins."""

    __tablename__ = "coins"
    __table_args__ = {"extend_existing": True}
    id: int = _sql.Column(_sql.Integer, primary_key=True, index=True)
    folder_path = _sql.Column(_sql.String, index=True)
    price = _sql.Column(_sql.Float, index=True)
    tirage = _sql.Column(_sql.Integer, index=True)
    country = _sql.Column(_sql.String, index=True)
    currency = _sql.Column(_sql.String, index=True)
    amount = _sql.Column(_sql.Float, index=True)
    year = _sql.Column(_sql.Integer, index=True)
    coin_name = _sql.Column(_sql.String, index=True)


@dataclass
class Predictions(_database.Base):
    """Predictions model. Will contain the predictions made by the AI model."""

    __tablename__ = "predictions"
    __table_args__ = {"extend_existing": True}
    id = _sql.Column(_sql.Integer, primary_key=True, index=True)
    id_users = _sql.Column(_sql.Integer, _sql.ForeignKey("users.id"), index=True)
    id_coins = _sql.Column(_sql.Integer, _sql.ForeignKey("coins.id"), index=True)
    class_name = _sql.Column(_sql.String, index=True)
    degree_of_certainty = _sql.Column(_sql.Float, index=True)
    region_of_interest = _sql.Column(_sql.String, index=True)
    predicted_picture_path = _sql.Column(_sql.String, index=True)
    date = _sql.Column(_sql.DateTime, default=_dt.datetime.utcnow)
    folder_path = _sql.Column(_sql.String, index=True)
    price = _sql.Column(_sql.Float, index=True)
    tirage = _sql.Column(_sql.Integer, index=True)
    country = _sql.Column(_sql.String, index=True)
    currency = _sql.Column(_sql.String, index=True)
    amount = _sql.Column(_sql.Float, index=True)
    year = _sql.Column(_sql.Integer, index=True)
    coin_name = _sql.Column(_sql.String, index=True)


###########################################################
# Models for Minio storage
###########################################################


class Buckets(str, Enum):
    """Define buckets on minio"""

    BASED_PICTURES = "based-pictures"
    USERS_PICTURES = "users-pictures"
