import uuid
from dataclasses import dataclass

import pandas as pd
from faker import Faker
import json
import re

fake = Faker()


@dataclass
class Users_Creation:
    """Users model. Inherit from the base class to create a table in the database."""

    email: str
    first_name: str
    last_name: str
    password: str

    @classmethod
    def create_random_user_dict(cls):
        unique_id = str(uuid.uuid4())[:5]
        email = f"test_{unique_id}@example.com"
        first_name = fake.first_name()
        last_name = fake.last_name()
        password = "password"

        return {
            "email": email,
            "first_name": first_name,
            "last_name": last_name,
            "password": password,
            "confirm_pswd": password,
        }


def prepare_data_for_bdd(database_column: pd.DataFrame):
    """Function transforming data to be compare in the same format of the logs.

    Args:
        database_column (pd.DataFrame): A dataframe with the data from the database.

    Returns:
        dict: Latest date from the database.
    """
    transform = database_column.sort_values(by="date", ascending=False)
    # transform = database_column.head(1)
    transform.loc[:, "date"] = transform["predicted_picture_path"].values[0]
    # change column order to match the one in the database
    transform = transform.drop(
        columns=["id", "id_users", "id_coins", "predicted_picture_path"]
    )
    column_order = ["class_name", "degree_of_certainty", "region_of_interest", "date"]
    transform = transform[column_order]
    transform_dict = transform.to_dict("records")[0]
    return transform_dict["date"]


def extract_all_coordinates_in_logs(log_path, min_certainty=None):
    """Get all the logs and transform them in a json format.

    Args:
        log_path (str): Path to the log file.
        min_certainty (float, optional): Filter the logs by the minimum degree of certainty.

    Returns:
        json: A json containing the logs.
    """
    try:
        with open(log_path, "r", encoding="utf-8") as log_file:
            log_text = log_file.read()

        pattern = r"(\d{8}_\d{6}) - INFO - Class: (.*?), Degree Of Certainty: (\d+), Region Of Interest:\[x1: (\d+\.\d+),y1: (\d+\.\d+), x2: (\d+\.\d+), y2: (\d+\.\d+)\]"
        matches = re.findall(pattern, log_text)

        if matches:
            results = []

            for match in matches:
                date, class_name, degree_of_certainty, x1, y1, x2, y2 = match
                degree_of_certainty = float(degree_of_certainty)

                result = {
                    "class_name": class_name,
                    "degree_of_certainty": degree_of_certainty,
                    "region_of_interest": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    },
                    "date": date,
                }
                last_element_formatted_for_db = [date, x1, y1, x2, y2]
                last_element_formatted_for_db = (
                    ",".join(last_element_formatted_for_db[0:5]) + ".jpg"
                )
                results.append(last_element_formatted_for_db)

            if not results:
                print("No matching log entries found.")
            return json.dumps(results, indent=4)
        else:
            print("No matching log entries found.")
    except Exception as error:
        print(f"An error occurred: {str(error)}")
