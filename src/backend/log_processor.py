"""Process logs and pictures path"""
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Set

import pandas as pd

import backend.services as _services
from backend.models import Buckets


# pylint: disable=line-too-long
# pylint: disable=invalid-name
@dataclass
class DataProcessor:
    """Reads a log file and a picture path and extracts data from them.
    The data can be sent to the database and Minio."""

    log_path: str = None
    picture_path: str = None
    send_to_database: bool = False
    save_to_minio: bool = False
    min_certainty: str = None

    def extract_date_classname_coordinates(self):
        """Search in the logs the best entry per class and return the date, class name and coordinates.

        Returns:
            json:
        """
        try:
            with open(self.log_path, "r", encoding="utf-8") as log_file:
                log_text = log_file.read()

            pattern = r"(\d{8}_\d{6}) - INFO - Class: (.*?), Degree Of Certainty: (\d+), Region Of Interest:\[x1: (\d+\.\d+),y1: (\d+\.\d+), x2: (\d+\.\d+), y2: (\d+\.\d+)\]"
            matches = re.findall(pattern, log_text)

            if matches:
                best_entries = {}  # Dictionary to store the best entry per class
                results = []

                for match in matches:
                    date, class_name, degree_of_certainty, x1, y1, x2, y2 = match
                    degree_of_certainty = float(degree_of_certainty)
                    # Check if this entry has a better degree_of_certainty for its class
                    if (
                        class_name not in best_entries
                        or (
                            self.min_certainty is not None
                            and degree_of_certainty > self.min_certainty
                        )
                        or (
                            self.min_certainty is None
                            and degree_of_certainty
                            > best_entries[class_name]["degree_of_certainty"]
                        )
                    ):
                        best_entries[class_name] = {
                            "degree_of_certainty": degree_of_certainty,
                            "region_of_interest": {
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                            },
                            "date": date,
                        }

                for class_name, entry in best_entries.items():
                    # if self.min_certainty is None or entry["degree_of_certainty"] > self.min_certainty:
                    if (
                        self.min_certainty is None
                        or entry["degree_of_certainty"] > self.min_certainty
                    ):
                        result = {
                            "class_name": class_name,
                            "degree_of_certainty": entry["degree_of_certainty"],
                            "region_of_interest": entry["region_of_interest"],
                            "date": entry["date"],
                        }
                        results.append(result)

                if not results:
                    print(
                        "No matching log entries found with the specified conditions."
                    )
                return json.dumps(results, indent=4)
            else:
                print("No matching log entries found.")
        except Exception as error:
            print(f"An error occurred: {str(error)}")

    def extract_pictures_names_coordinates(self):
        """Extracts pictures names and coordinates from a picture path.

        Returns:
            str: A JSON string containing the pictures names and coordinates.
        """
        try:
            if not os.path.exists(self.picture_path):
                print("Picture path does not exist.")
            picture_path = os.listdir(self.picture_path)
            if not picture_path:
                print("No pictures found in the picture path.")
                return json.dumps({"pictures_data": []}, indent=4)

            # Extract and store each element as date, x1, y1, x2, y2
            extracted_data = []
            for p in picture_path:
                filename = os.path.splitext(p)[0]
                parts = filename.split(",")
                if len(parts) == 5:
                    date, x1, y1, x2, y2 = parts
                    extracted_data.append(
                        {
                            "date": date,
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                        }
                    )

            return json.dumps({"pictures_data": extracted_data}, indent=4)
        except Exception as error:
            print(f"An error occurred: {str(error)}")

    @staticmethod
    def compare_dates(date1: Dict[str, str], date2: Dict[str, str]) -> Set[str]:
        """Compare two dates and return the common dates.

        Args:
            date1 (Dict[str, str]): The first date.
            date2 (Dict[str, str]): The second date.

        Returns:
            Set[str]: A set containing the common dates.
        """
        return set(date1) & set(date2)

    async def process_data(self):
        """
        Processes data from a log file and a picture path
        and sends it to the database and Minio.
        """
        log_dates_json = self.extract_date_classname_coordinates()
        log_dates = json.loads(log_dates_json)
        log_dates = {entry["date"]: entry for entry in log_dates}

        picture_dates_json = self.extract_pictures_names_coordinates()
        picture_dates = json.loads(picture_dates_json)
        picture_dates = {
            entry["date"]: entry for entry in picture_dates["pictures_data"]
        }

        common_dates = self.compare_dates(
            set(log_dates.keys()), set(picture_dates.keys())
        )
        if common_dates:
            results = []
            for date in common_dates:
                log_entry = log_dates[date]
                roi = log_entry["region_of_interest"]
                roi_str = f"{roi['x1']},{roi['y1']},{roi['x2']},{roi['y2']}"
                predicted_picture_path = f"{date},{roi_str}.jpg"
                result = {
                    "degree_of_certainty": log_entry["degree_of_certainty"],
                    "region_of_interest": str(log_entry["region_of_interest"]),
                    "class_name": log_entry[
                        "class_name"
                    ],  # To add after if edit the Models in de database
                    "predicted_picture_path": predicted_picture_path,
                    "date": date,
                }
                results.append(result)
            if self.send_to_database and self.save_to_minio:
                result_df = pd.DataFrame(results)
                picture_paths_to_send = [
                    result["predicted_picture_path"] for result in results
                ]
                if len(result_df) != len(picture_paths_to_send):
                    raise IndexError(
                        f"The number of pictures are : {len(picture_paths_to_send)} \
                            and the number of rows in the dataframe are {len(result_df)}."
                    )
                _services.insert_df(result_df, "predictions")
                print("Data sent to the database.")
                _services._send_pictures(
                    Buckets.USERS_PICTURES.value,
                    self.picture_path,
                    picture_paths_to_send,
                )
                print("Data saved to Minio.")


# def main():
#     log = DataProcessor(
#         log_path="/workspaces/AICoinXpert/src/backend/video/tmp/detection_log.txt",
#         picture_path="/workspaces/AICoinXpert/src/backend/video/tmp/images",
#         send_to_database=True,
#         save_to_minio=True,
#     )
#     log = log.extract_date_classname_coordinates()
#     print(log)


# #     #test = print(log.extract_date_classname_coordinate(70))
# # #     #print(f'Pictures: {log.extract_pictures_names_coordinates()}')
# # # # check = json.loads(check)
# # # # #print(check)
# # # # #log.process_data(70)
# if __name__ == "__main__":
#     main()
