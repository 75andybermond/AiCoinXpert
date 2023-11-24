import pytest
import requests
from bs4 import BeautifulSoup

from backend.test.ressources.backend.flask_integration import Users_Creation

BASE_URL = "http://127.0.0.1:5000"


# Helper function to log in a user
def login_user(email: str, password: str) -> requests.Session:
    session = requests.Session()
    session.post(
        f"{BASE_URL}/login",
        data={
            "email": email,
            "password": password,
        },
    )
    return session


def test_hello_world():
    """Simple test to check the index is accessible."""
    response = requests.get(f"{BASE_URL}/hello_world")
    if response.status_code == 200:
        print(response.content, "is the response data")
    else:
        pytest.fail("The index is not accessible.")


def test_registration():
    """Check the registration is successful."""
    create_fake_user = Users_Creation.create_random_user_dict()
    # Create a dictionary with the form data
    form_data = {
        "email": create_fake_user["email"],
        "first_name": create_fake_user["first_name"],
        "last_name": create_fake_user["last_name"],
        "password": create_fake_user["password"],
        "confirm_pswd": create_fake_user["confirm_pswd"],
    }
    # Send the form data as part of the request
    response = requests.post(
        url=f"{BASE_URL}",
        data=form_data,
    )

    if response.status_code != 200:
        pytest.fail(f"Status code is {response.status_code} and should be 200")
    if b"Registered successfully. Please login." not in response.content:
        pytest.fail("Registration failed.")


def test_login():
    """Check the login is successful."""
    create_fake_user = Users_Creation.create_random_user_dict()
    session = login_user(create_fake_user["email"], create_fake_user["password"])
    response = session.get(BASE_URL)
    if response.status_code != 200:
        pytest.fail(f"Status code is: {response.status_code} and should be 200")


def test_dashboard():
    """Verify that the dashboard is accessible and the buttons are present."""
    create_fake_user = Users_Creation.create_random_user_dict()
    # Log in the user and get the session
    session = login_user(create_fake_user["email"], create_fake_user["password"])
    # Access the dashboard page
    response = session.get(f"{BASE_URL}/dashboard")
    if response.status_code != 200:
        pytest.fail("The dashboard is not accessible.")

    # parse the HTML content using BeautifulSoup
    bs = BeautifulSoup(response.content, "html.parser")

    # Verify that dashboard contains the Wallet and Launch Video buttons
    wallet_button = bs.find("a", {"href": "/wallet"})
    launch_video_button = bs.find("a", {"href": "/streaming"})

    if not wallet_button or not launch_video_button:
        pytest.fail(
            "The dashboard does not contain the Wallet and Launch Video buttons."
        )


def test_logout():
    """Check the logout is successful."""
    # test_login()
    # Perform the logout action
    response = requests.get(f"{BASE_URL}/logout")
    # Verify the status code (200 for success)
    if response.status_code != 200:
        pytest.fail("The logout is not accessible.")
    # Parse the HTML content using BeautifulSoup to check the success message
    bs = BeautifulSoup(response.content, "html.parser")
    # Find the success message
    success_message = bs.find("div", {"class": "alert alert-success"})
    # Verify that the success message contains the expected text
    if (
        success_message
        and "You have logged out successfully" not in success_message.text
    ):
        pytest.fail("Logout was not successful or message is incorrect.")


def test_wallet():
    """Check the wallet is accessible."""
    # test_login()
    response = requests.get(f"{BASE_URL}/wallet")
    if response.status_code != 200:
        pytest.fail("The wallet is not accessible.")


def test_webcam_streaming():
    """Check the webcam streaming is accessible."""
    response = requests.get(f"{BASE_URL}/streaming", stream=True)
    if response.status_code != 200:
        pytest.fail("The webcam streaming is not accessible.")
