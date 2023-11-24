"""Configuration file for pytest."""
import pytest

import backend.test.ressources.backend.services_db as _services
from backend.minio_minio import MinioClient
from backend.models import Buckets


@pytest.fixture(scope="session")
def test_db():
    """Create a test database."""
    _services.create_database("test_db")
    yield
    _services.remove_database("test_db")


@pytest.fixture(scope="function")
def clean_data():
    minio_client = MinioClient()
    yield
    minio_client.delete_all_objects(bucket_name=Buckets.USERS_PICTURES.value)
