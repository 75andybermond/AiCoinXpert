"""Database  and Minio models."""
import datetime as _dt
from dataclasses import dataclass
from enum import Enum

import sqlalchemy as db
from flask import Flask
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from flask_login import LoginManager, UserMixin
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect

# pylint: disable=E0401
# import db as _database

app = Flask(__name__, template_folder='/workspaces/AiCoinXpert/src/frontend/templates', static_folder='/workspaces/AiCoinXpert/src/frontend/static')
app.secret_key = "replace later"
app.config[
    "SQLALCHEMY_DATABASE_URI"
] = "postgresql://root:password@localhost:5432/coins_db"
app.config["WTF_CSRF_ENABLED"] = False
db = SQLAlchemy(app)
admin = Admin(app)
login = LoginManager(app)

csrf = CSRFProtect(app)
login.init_app(app)
login.login_view = "login"


@dataclass
class Users(UserMixin, db.Model):
    """Users model. Inherit from the base class to create a table in the database."""

    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}
    id = db.Column(db.Integer, primary_key=True, index=True)
    first_name = db.Column(db.String, index=True)
    last_name = db.Column(db.String, index=True)
    password = db.Column(db.String, index=True)
    email = db.Column(db.String, index=True, unique=True)
    data_created = db.Column(db.DateTime, default=_dt.datetime.utcnow)
    predictions = db.relationship("Predictions", back_populates="user")

    def __str__(self):
        return self.first_name


@dataclass
class Predictions(db.Model):
    """Predictions model. Will contain the predictions made by the AI model."""

    __tablename__ = "predictions"
    __table_args__ = {"extend_existing": True}
    id = db.Column(db.Integer, primary_key=True, index=True)
    id_users = db.Column(db.Integer, db.ForeignKey("users.id"), index=True)
    id_coins = db.Column(db.Integer, db.ForeignKey("coins.id"), index=True)
    class_name = db.Column(db.String, index=True)
    degree_of_certainty = db.Column(db.Float, index=True)
    region_of_interest = db.Column(db.String, index=True)
    predicted_picture_path = db.Column(db.String, index=True)
    date = db.Column(db.DateTime, default=_dt.datetime.utcnow)
    user = db.relationship("Users", back_populates="predictions")
    folder_path = db.Column(db.String, index=True)
    price = db.Column(db.Float, index=True)
    tirage = db.Column(db.Integer, index=True)
    country = db.Column(db.String, index=True)
    currency = db.Column(db.String, index=True)
    amount = db.Column(db.Float, index=True)
    year = db.Column(db.Integer, index=True)
    coin_name = db.Column(db.String, index=True)


@dataclass
class Coins(db.Model):
    """Coins informations. Contains the informations about the coins."""

    __tablename__ = "coins"
    # __table_args__ = {"extend_existing": True}
    id = db.Column(db.Integer, primary_key=True, index=True)
    folder_path = db.Column(db.String, index=True)
    price = db.Column(db.Float, index=True)
    tirage = db.Column(db.Integer, index=True)
    country = db.Column(db.String, index=True)
    currency = db.Column(db.String, index=True)
    amount = db.Column(db.Float, index=True)
    year = db.Column(db.Integer, index=True)
    coin_name = db.Column(db.String, index=True)


class PostView(ModelView):
    """Custom view for the ModelView class in the app, route /admin.

    Args:
        ModelView (Object Model): Object ModelView from flask_admin.
    """

    can_delete = True
    form_columns = [
        "degree_of_certainty",
        "region_of_interest",
        "predicted_picture_path",
        "user",
    ]
    column_list = [
        "degree_of_certainty",
        "region_of_interest",
        "predicted_picture_path",
        "user",
    ]


admin.add_view(ModelView(Users, db.session))
admin.add_view(ModelView(Coins, db.session))
admin.add_view(PostView(Predictions, db.session))


###########################################################
# Models for Minio storage
###########################################################


class Buckets(str, Enum):
    """Define buckets on minio"""

    BASED_PICTURES = "based-pictures"
    USERS_PICTURES = "users-pictures"
