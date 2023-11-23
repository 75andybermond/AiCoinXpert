"""Form fields and routes for the Flask application."""
import asyncio
from datetime import datetime

from flask import Response, flash, redirect, render_template, url_for
from flask_login import current_user, login_required, login_user, logout_user

from backend.camera.fake import FakeCamera
from backend.infer_model import ImageClassifier
from backend.log_processor import DataProcessor
from backend.minio_minio import DisplayMinio
from backend.models import Buckets, Predictions, Users, app, db, login
from backend.services import find_coins_in_database
from backend.video.run_yolo_video import MODEL_PATH, YOLODetector
from backend.wtform_fields import LoginForm, RegistrationForm, pbkdf2_sha256

# pylint: disable=unexpected-keyword-arg
MIN_CERTAINTY = 50  # Minimum certainty for a coin to be considered valid (in %)
PROBABILITY_THRESHOLD = (
    0.5  # Threshold for the classifier to consider a prediction valid
)
classifier = ImageClassifier()  # Initialize the classifier


@login.user_loader
def load_user(ids):
    """
    This function is used to load a user from the database given its ID.

    Args:
        ID (int): The ID of the user to load.

    Returns:
        Users: The user object corresponding to the given ID.
    """
    return db.session.get(Users, int(ids))


@app.route("/hello_world", methods=["GET"])
def hello_world():
    """Simple hello world for testing API

    Returns:
        str: An 'Hello, World' message
    """
    return "Hello, World!"


@app.route("/", methods=["GET", "POST"])
def index():
    """
    This function handles the registration form and renders the index page.

    Returns:
        str: The HTML content of the index page.
    """
    reg_form = RegistrationForm()

    # Update database if validation success
    if reg_form.validate_on_submit():
        email = reg_form.email.data
        first_name = reg_form.first_name.data
        last_name = reg_form.last_name.data
        password_user = reg_form.password.data

        # Hash password
        hashed_pswd = pbkdf2_sha256.hash(password_user)

        # Add email & hashed password to DB
        user = Users(
            email=email,
            first_name=first_name,
            last_name=last_name,
            password=hashed_pswd,
        )
        db.session.add(user)
        db.session.commit()

        flash("Registered successfully. Please login.", "success")
        return redirect(url_for("index"))

    return render_template("index.html", form=reg_form), 200


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    This function handles the login form and renders the login page.

    Returns:
        str: The HTML content of the login page.
    """
    login_form = LoginForm()

    # Allow login if validation success
    if login_form.validate_on_submit():
        user_object = Users.query.filter_by(email=login_form.email.data).first()
        login_user(user_object)
        return redirect(url_for("dashboard"))

    return render_template("login.html", form=login_form), 200


@app.route("/logout", methods=["GET"])
def logout():
    """
    This function logs out the user and redirects to the login page.

    Returns:
        str: A redirect to the login page.
    """
    # Logout user
    logout_user()
    flash("You have logged out successfully", "success")
    return redirect(url_for("login"))


@app.route("/dashboard", methods=["GET", "POST"])
# @login_required To add after finding out how to test it
def dashboard():
    """
    This function handles the dashboard page and displays the user's predictions.

    Returns:
        str: The HTML content of the dashboard page.
    """
    return render_template("dashboard.html"), 200


@app.route("/wallet", methods=["GET", "POST"])
@login_required
async def wallet():
    """
    This function handles the wallet page and displays the user's wallet.

    Returns:
        str: The HTML content of the wallet page.
    """
    display_minio = DisplayMinio()
    # Check if the user is authenticated
    if not current_user.is_authenticated:
        # Handle the case where the user is not authenticated (not logged in)
        flash("You need to log in to access this page.", "danger")
        return redirect(
            url_for("login")
        )  # Redirect to the login page or another appropriate page

    user_id = current_user.id
    # Retrieve the user's wallet from the database
    predictions = Predictions.query.filter_by(id_users=user_id).all()
    # i want to display only one value per class_name
    predictions = Predictions.query.filter_by(id_users=user_id).distinct(
        Predictions.class_name
    )

    if not predictions:
        flash("You have no predictions yet.", "info")  # Provide an info message
        predictions = []  # Set predictions to an empty list

    # Await the result of classify_image()
    prediction_results = await classify_image()

    # Attach predictions to the wallet
    await attach_predictions_to_wallet(prediction_results, id_users=user_id)

    return (
        render_template("wallet.html", wallet=predictions, display_minio=display_minio),
        200,
    )


def generate_frames():
    """Generate bytes frames from a video feed using YOLO object detection

    Yields:
        bytes: Frames in HTTP multipart format
    """
    fake_camera = FakeCamera()
    detector = YOLODetector(camera=fake_camera, model_path=MODEL_PATH, threshold=0.1)

    try:
        while True:
            frames = detector.extract_frames()
            for frame in frames:
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
    except GeneratorExit:
        # Release resources when the generator is closed
        detector.cap.release()
        detector.on_close()


@app.route("/streaming", methods=["GET"])
def video():
    """Stream video frames route.

    Returns:
        Response: A streaming response with video frames and the sending data processing thread.
    """

    # Generate frames and return a streaming response
    response = Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        status=200,
    )

    # Run data processing after the response is sent
    response.call_on_close(run_data_processing_thread)

    return response


async def run_data_processing():
    """Run data processing asynchronously."""
    data_processor = DataProcessor(
        log_path="/workspaces/AiCoinXpert/src/backend/video/tmp/detection_log.txt",
        picture_path="/workspaces/AiCoinXpert/src/backend/video/tmp/images",
        send_to_database=True,
        save_to_minio=True,
        min_certainty=MIN_CERTAINTY,
    )
    await data_processor.process_data()
    # await classify_image()


def run_data_processing_thread():
    """Run data processing in a separate thread."""
    with app.app_context():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_data_processing())
        video_stream_done.set_result(False)


async def classify_image():
    """Classify a list of images using a trained model.

    Returns:
        json: JSON object with the class and the probability of the prediction.
    """
    minio_pic = DisplayMinio()
    pic_names = minio_pic.get_all_files(Buckets.USERS_PICTURES.value)
    predictions = []

    # Use asyncio.gather to concurrently process predictions
    async def process_image(image_name):
        # Fetch the image data from Minio
        image_data = minio_pic.get_object_by_name(
            Buckets.USERS_PICTURES.value, image_name
        )
        # Predict the class of the image using the classifier
        prediction = await classifier.predict_image(
            probability_threshold=PROBABILITY_THRESHOLD,
            image_data=image_data,
            image_name=image_name,
        )
        predictions.append(prediction)

    # Create a list of coroutines to process images concurrently
    coroutines = [process_image(image_name) for image_name in pic_names]

    # Use asyncio.gather to run the coroutines concurrently
    await asyncio.gather(*coroutines)

    # Use find_coins_in_database here if needed
    return predictions


async def attach_predictions_to_wallet(predictions: list, id_users: int):
    """Attach predictions to the user's wallet and add additional details.

    Args:
        predictions (list): List of predictions made by the classifier.
        id_users (int): The ID of the logged-in user.
    """
    for prediction in predictions:
        # Extract relevant prediction data
        class_name = prediction["class"]
        degree_of_certainty = prediction["prob"]
        degree_of_certainty = round(degree_of_certainty * 100, 2)
        class_name = class_name + ".jpg"
        if class_name == "Unknown.jpg":
            continue
        coin_details = find_coins_in_database("coins", coin_class=class_name)
        if not coin_details:
            # Skip the prediction if the coin is not found in the database
            KeyError(f"Coin {coin_details} not found in the database.")

        for coin_detail in coin_details:
            # Create a new prediction entry with additional details
            new_prediction = Predictions(
                id_users=id_users,
                class_name=class_name,
                degree_of_certainty=degree_of_certainty,
                date=datetime.utcnow(),
                folder_path=coin_detail["folder_path"],
                price=coin_detail["price"],
                tirage=coin_detail["tirage"],
                country=coin_detail["country"],
                currency=coin_detail["currency"],
                amount=coin_detail["amount"],
                year=coin_detail["year"],
                coin_name=coin_detail["coin_name"],
            )
            # Add the new prediction to the database session and commit
            db.session.add(new_prediction)
            db.session.commit()


if __name__ == "__main__":
    video_stream_done = asyncio.Future()
    app.run(debug=True)
