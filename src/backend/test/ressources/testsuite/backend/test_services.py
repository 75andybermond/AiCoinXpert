import json
import time

import pytest

from backend import minio_minio as minio
from backend.log_processor import DataProcessor
from backend.services import display_data_from_table, find_coins_in_database
from backend.test.ressources.backend import services_db as _services
from backend.test.ressources.backend.flask_integration import (
    extract_all_coordinates_in_logs,
    prepare_data_for_bdd,
)


def test_add_table(test_db):
    """Test adding a table to the database."""
    # Add a table to the database
    _services.create_fake_table()

    # # Check if the table exists in the database
    if not _services.database_exists("postgres"):
        pytest.fail("Table does not exist in the database")


def test_data_saved_to_db_and_minio(clean_data):
    """Check if the captured coins saved in the logs are the same saved the database and minio."""
    time.sleep(5)  # wait for the databse to be updated

    check_log_last_elements = extract_all_coordinates_in_logs(
        log_path="/workspaces/AiCoinXpert/src/backend/video/tmp/detection_log.txt"
    )

    # Get the last 10 elements in the log file by the latest date
    check_log_last_elements = json.loads(check_log_last_elements)
    check_log_last_elements = list(check_log_last_elements)
    check_log_last_elements.reverse()
    check_log_last_elements = str(check_log_last_elements[:10])

    # Get the last 10 elements in the database by the latest date
    database_elements = display_data_from_table("predictions")
    elements_in_database = prepare_data_for_bdd(database_elements)

    # check if at least one element of the log is present in the database
    for element in check_log_last_elements:
        if any(element == db_element for db_element in elements_in_database):
            break
    else:
        pytest.fail(
            f"No elements {check_log_last_elements} found in the logs that are present in the database elements: {elements_in_database}."
        )

    minio_display = minio.DisplayMinio()
    list_of_files = minio_display.get_all_files("users-pictures")
    last_element_saved_minio = list_of_files[-1]
    if elements_in_database != last_element_saved_minio:
        pytest.fail(
            f"The last element in the log is: {elements_in_database} and in minio is {last_element_saved_minio}"
        )


def test_coin_found_in_db_from_predictions():
    """Test if the coin is found in the database based on model prediction output."""
    expected_result = [
        {
            "id": 295,
            "folder_path": "Autriche-1-Euro-2014-3022800-155540004211316.jpg",
            "price": 27.74,
            "tirage": 52500,
            "country": "Austria",
            "currency": "Euro",
            "amount": 1.0,
            "year": 2014,
            "coin_name": "Autriche-1-Euro-2014",
        }
    ]

    coins = [
        {
            "class": "Autriche-1-Euro-2014-3022800-155540004211316.jpg",
            "prob": 0.40283939242362976,
        }
    ]

    result = find_coins_in_database(
        table_name="coins",
        coin_class="Autriche-1-Euro-2014-3022800-155540004211316.jpg",
    )

    if result != expected_result:
        pytest.fail(f"Expected result: {expected_result} but got {result}")


def test_coin_not_found_in_db_from_captures():
    """Test if the coin is found in the database based on model prediction output."""
    expected_result = []

    coins = [
        {
            "class": "Unknown",
            "prob": 0.000,
        }
    ]

    result = find_coins_in_database(table_name="coins", coin_class="Unknown")

    if result != expected_result:
        pytest.fail(f"Expected result: {expected_result} but got {result}")
