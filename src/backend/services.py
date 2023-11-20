"""Services and funtions for database."""
# pylint: disable=E0401
import gzip
import pickle

import jellyfish
import minio_minio as _minio
import models as _models
import pandas as pd
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError

import db as _database

#### Services for database ####


def _add_table():
    """Add a table based on the model to the database."""

    return _database.Base.metadata.create_all(bind=_database.engine)


def _drop_table():
    """Drop all tables in the database."""

    confirmation = input(
        "Are you sure you want to drop all tables in the database? (y/n): "
    )
    if confirmation.lower() == "y":
        return _database.Base.metadata.drop_all(bind=_database.engine)
    print("Table drop cancelled.")


def check_if_table_exists(table_name):
    """Check if a table exists in the database."""

    inspector = inspect(_database.engine)
    return table_name in inspector.get_table_names()


def is_database_responding():
    """Check if the database is responding."""
    try:
        with _database.engine.connect() as connection:
            connection.execute("SELECT 1")
        return True
    except OperationalError:
        return False


def insert_csv(file_path: str, table_name: str) -> None:
    """Insert data from a CSV file into a PostgreSQL database table.

    Args:
        table_name (str): The name of the table.
        csv_file_path (str): The path to the CSV file.
    """
    engine = _database.engine
    data = pd.read_csv(file_path)
    data.to_sql(table_name, engine, if_exists="append", index=False)


def insert_df(dataframe: pd.DataFrame, table_name: str) -> None:
    """Insert data from a DataFrame pandas into a PostgreSQL database table.

    Args:
        df (pd.DataFrame): The dataframe to insert
        table_name (str): The name of the table to insert the data into.
    """
    engine = _database.engine
    dataframe.to_sql(table_name, engine, if_exists="append", index=False)


def display_data_from_table(table_name: str) -> pd.DataFrame:
    """Display data from a table in the database.
    Args:
         table_name (str): The name of the table.
    """
    with _database.engine.connect() as connection:
        if not connection.dialect.has_table(connection, table_name):
            return print("Table does not exist.")
        return pd.read_sql_table(table_name, connection)


def find_coins_in_database(table_name: str, coin_class: str):
    """
    Find coins details in the database based on the model predictions.

    Args:
        table_name (str): The name of the database table to search.
        coin_class (str): The class name of the coin to search for.

    Returns:
        list: A list of dictionaries containing the details for all coins matching the class.
    """
    with _database.engine.connect() as connection:
        coin_data = pd.read_sql_table(table_name, connection)
        coin_data_filtered = coin_data[
            coin_data["folder_path"].str.startswith(coin_class)
        ]
        results = []
        for _, row in coin_data_filtered.iterrows():
            result = row.to_dict()
            results.append(result)
        return results


#### Services for minio ####


def _make_bucket():
    """Create a bucket on minio."""
    client = _minio.MinioClient()
    client.create_bucket(bucket_name=_models.Buckets.BASED_PICTURES.value)
    client.create_bucket(bucket_name=_models.Buckets.USERS_PICTURES.value)


def _send_pictures(bucket_name: str, folder_path: str, object_names: list = None):
    """Send based pictures to minio."""
    client = _minio.MinioClient()
    client.send_directory_of_pictures(
        bucket_name=bucket_name, folder_path=folder_path, object_names=object_names
    )


#### Services for for synchronize Minio/DB ####


def get_coin_with_picture(coin_name: str):
    """
    Get a specific coin with its corresponding picture.

    Args:
        coin_name (str): The name of the coin.

    Returns:
        dict: A dictionary containing the coin data and the picture URL.
    """
    table_name = "coins"
    picture_bucket = _models.Buckets.BASED_PICTURES.value

    with _database.engine.connect() as connection:
        if not connection.dialect.has_table(connection, table_name):
            print("Table does not exist.")

        # Retrieve the coin data from the table
        coin_data = pd.read_sql_table(table_name, connection)
        coin_data = coin_data[coin_data["coin_name"] == coin_name]
        # once the coin is found retrieve all its data
        coin = coin_data.to_dict(orient="records")[0]
        print(coin)
        if coin_data.empty:
            print("Coin not found.")

        # Retrieve the picture path from the coins database
        picture_path = coin_data.iloc[0]["folder_path"]
        # Retrieve the picture from the Minio bucket using the
        # picture path
        minio_client = _minio.DisplayMinio()
        picture_url = minio_client.get_all_files(picture_bucket)

        best_match = None
        best_match_score = 0

        for picture in picture_url:
            score = jellyfish.jaro_winkler(picture, picture_path)
            if score > best_match_score:
                best_match = picture
                best_match_score = score

        if best_match is not None:
            print(f"Picture {best_match} found with score {best_match_score}.")
            img = _minio.DisplayMinio().display_picture_by_name(
                _models.Buckets.BASED_PICTURES.value, best_match
            )
            return {
                "coin_infos": coin,
                "picture_path": picture_path,
                "picture": img,
            }
        print("Picture not found.")


#### services to reduce model size ####


def compress_model(model, path: str) -> None:
    """Compress .pth model to lighter vesions.

    Args:
        model (pth): Model to compress.
        path (str): Path to save the model.
    """
    with gzip.GzipFile(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> object:
    """Load compressed model.

    Args:
        path (str): Path to the model.

    Returns:
        (Pickle_object): Model loaded.
    """
    with gzip.GzipFile(path, "rb") as f:
        model = pickle.load(f)
    return model


def main():
    """Main function."""
    #_add_table()
    #_make_bucket()
    insert_csv("/workspaces/AiCoinXpert/algo/webscraping/coins_to_db.csv", "coins")
    _send_pictures(
        _models.Buckets.BASED_PICTURES.value,
        "/workspaces/AiCoinXpert/algo/webscraping/data/selected_coins_above20",
    )


#### Scrpit sending datas to database and pictures to minio synchronyously ####

# I want to take my pictures from locally and send them to minio
# at the same time I feed the database.

if __name__ == "__main__":
    main()
