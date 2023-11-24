import inspect
import sys

sys.path.append("src/backend")

import psycopg2
import sqlalchemy as _sql

# import models as _models
from sqlalchemy.exc import OperationalError
from sqlalchemy.schema import CreateSchema
from sqlalchemy.sql import text

# Global variables for database credentials
DB_USER = "root"
DB_PASSWORD = "password"
DB_HOST = "localhost"
DB_PORT = 5432


def create_engine():
    """Create a SQLAlchemy engine using the global database credentials."""
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres"
    engine = _sql.create_engine(db_url)
    return engine


def create_database(db_name):
    """Create a new database with the given name and credentials."""

    engine = create_engine()

    # Create a connection
    conn = engine.connect()
    # Create a transaction
    trans = conn.begin()
    try:
        # Create the new database
        conn.execute(CreateSchema(db_name))
        # Commit the transaction
        trans.commit()
        print(f"Database '{db_name}' created successfully!")
    except Exception as e:
        # Roll back the transaction on error
        trans.rollback()
        raise e
    finally:
        # Close the connection
        conn.close()


def remove_database(db_name):
    """Remove a database with the given name and credentials."""

    conn = psycopg2.connect(
        dbname="postgres",
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )

    try:
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
        cursor.execute(f"DROP SCHEMA IF EXISTS {db_name} CASCADE")
    except Exception as e:
        raise e
    finally:
        cursor.close()
        conn.close()


def get_all_database_names():
    """Get the names of all the databases present."""
    engine = create_engine()

    with engine.connect() as conn:
        # Get the names of all the databases present
        query = conn.execute(text("SELECT datname FROM pg_database"))
        list_of_databases = query.fetchall()

        # check in the list of databases if the named database exists
        list_of_databases = [db[0] for db in list_of_databases]

        return list_of_databases


def create_fake_table():
    """Create a fake table in the database."""

    metadata = _sql.MetaData()
    fake_table = _sql.Table(
        "fake_table",
        metadata,
        _sql.Column("id", _sql.Integer, primary_key=True),
        _sql.Column("name", _sql.String(50)),
        _sql.Column("surname", _sql.String(50)),
        _sql.Column("email", _sql.String(50)),
        _sql.Column("phone_number", _sql.String(50)),
    )
    engine = create_engine()
    metadata.create_all(engine)


def database_exists(db_name):
    """Check if a database exists."""
    engine = create_engine()

    with engine.connect() as conn:
        # Get the names of all the databases present
        query = conn.execute(text("SELECT datname FROM pg_database"))
        list_of_databases = query.fetchall()

        # check if the named database exists
        return db_name in [db[0] for db in list_of_databases]
