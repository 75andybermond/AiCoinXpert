"""Module for wtform fields"""
from flask_wtf import FlaskForm
from passlib.hash import pbkdf2_sha256
from wtforms import PasswordField, StringField
from wtforms.validators import EqualTo, InputRequired, Length, ValidationError

from backend.models import Users


def invalid_credentials(form, field):
    """Username and password checker"""

    password = field.data
    email = form.email.data

    # Check email is invalid
    user_data = Users.query.filter_by(email=email).first()
    if user_data is None:
        raise ValidationError("Username or email is incorrect")

    # Check password in invalid
    if not pbkdf2_sha256.verify(password, user_data.password):
        raise ValidationError("Username or password is incorrect")


class RegistrationForm(FlaskForm):
    """Registration form"""

    email = StringField(
        "email",
        validators=[
            InputRequired(message="email required"),
            Length(min=4, max=25, message="Email must be between 4 and 25 characters"),
        ],
    )

    first_name = StringField(
        "first_name",
        validators=[
            InputRequired(message="First name required"),
            Length(
                min=4, max=25, message="First name must be between 4 and 25 characters"
            ),
        ],
    )

    last_name = StringField(
        "last_name",
        validators=[
            InputRequired(message="Last name required"),
            Length(
                min=2, max=25, message="Last name must be between 4 and 25 characters"
            ),
        ],
    )

    password = PasswordField(
        "password",
        validators=[
            InputRequired(message="Password required"),
            Length(
                min=4, max=25, message="Password must be between 4 and 25 characters"
            ),
        ],
    )
    confirm_pswd = PasswordField(
        "confirm_pswd",
        validators=[
            InputRequired(message="Password required"),
            EqualTo("password", message="Passwords must match"),
        ],
    )

    def validate_username(self, email):
        """
        Validate username

        Args:
            username (str): The username to validate.
        """
        user_object = Users.query.filter_by(email=email.data).first()
        if user_object:
            raise ValidationError(
                "Username already exists. Select a different username."
            )


class LoginForm(FlaskForm):
    """Login form"""

    email = StringField("email", validators=[InputRequired(message="Email required")])
    password = PasswordField(
        "password",
        validators=[InputRequired(message="Password required"), invalid_credentials],
    )
