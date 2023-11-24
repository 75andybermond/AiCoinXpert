import os
import re
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class SmtpServer(str, Enum):
    office365 = "smtp.office365.com"
    gmail = "smtp.gmail.com"
    yahoo = "smtp.mail.yahoo.com"
    outlook = "smtp-mail.outlook.com"


def create_dataframe_from_logs():
    # Read the existing data from the CSV file into a DataFrame
    if os.path.exists("scripts/test_results.csv"):
        df = pd.read_csv("scripts/test_results.csv")
    else:
        df = pd.DataFrame(columns=["test_name", "result", "datetime"])

    # Read the test results from standard input into a list of dictionaries
    for line in sys.stdin:
        # Search for lines that contain test results
        match = re.search(r"(test_\w+)\s(PASSED|FAILED)", line)
        if match:
            test_name = match.group(1)
            result = match.group(2)
            df.loc[len(df)] = [test_name, result, datetime.now()]

    # Save the DataFrame to the CSV file
    df.to_csv("scripts/test_results.csv", index=False)


def check_if_all_tests_passed() -> list:
    """Function that checks if any test failed and returns a boolean"""
    df = pd.read_csv("scripts/test_results.csv")
    failed_results = df[df.result == "FAILED"]
    print(f"Number of failed tests: {len(failed_results)}")
    return failed_results


def send_email_notification(
    subject: str, failed_tests: list, receiver_email="andy.bermond75@gmail.com"
):
    """Function that sends an email notification if any test failed"""
    sender_email = "andy.bermond75@gmail.com"
    password = os.getenv("EMAIL_PASSWORD")

    # Set up the MIME object
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Attach the body of the email
    body = failed_tests.to_html()
    message.attach(MIMEText(body, "html"))

    with smtplib.SMTP(SmtpServer.gmail, 587) as server:
        # Start TLS for security
        server.starttls()

        # Log in to your Gmail account
        server.login(sender_email, password)

        # Send the email
        server.send_message(message)

    print("Email sent successfully")


def main():
    create_dataframe_from_logs()
    failed_tests = check_if_all_tests_passed()
    if not failed_tests.empty:
        send_email_notification("Test results", failed_tests)


if __name__ == "__main__":
    main()
