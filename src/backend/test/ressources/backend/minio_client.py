"""Functions used for minio tests"""
import pytest
from minio import Minio
from minio.deleteobjects import DeleteObject
from minio.error import S3Error, ServerError


def compare_bucket_list(minio_client: Minio, bucket_app_list: list):
    """Get current list of bucket and compare against expected list.

    Args:
        minio_client (Minio): An instance of minio client.
        bucket_name (str): Name of the bucket.

    """
    bucket_current_list = []
    buckets = minio_client.list_buckets()
    for bucket in buckets:
        bucket_current_list.append(bucket.name)
    if bucket_current_list != bucket_app_list:
        pytest.fail(
            f"Found difference on bucket list : {bucket_current_list} instead of {bucket_app_list}"
        )


def clean_minio_bucket(minio_client: Minio, bucket_name: str):
    """Delete all objects recursively in a minio bucket.

    Args:
        minio_client (Minio): An instance of minio client.
        bucket_name (str): Name of the bucket.

    """
    delete_object_list = [
        DeleteObject(x.object_name)
        for x in minio_client.list_objects(bucket_name, recursive=True)
    ]
    try:
        # remove all the objects in the bucket using the list of of DeleteObject instances
        errors = minio_client.remove_objects(bucket_name, delete_object_list)
        for error in errors:
            pytest.fail(f"Error occurred when deleting object: {error}")
    # handle any exceptions that may occur during object deletion
    except (S3Error, ServerError) as err:
        pytest.fail(f"Encountered error: {err}")


def most_recent_object(minio_client: Minio, bucket_name: str) -> object | None:
    """Get the most recently added object in the bucket.

    Args:
        minio_client (Minio): An instance of minio client.
        bucket_name (str): Name of the bucket to inspect.

    Return:
        object | None: Minio object or None.
    """
    try:
        objects = minio_client.list_objects(bucket_name, recursive=True)
    except (S3Error, ServerError) as err:
        pytest.fail(f"Encountered error: {err}")
    return sorted(objects, key=lambda obj: obj.last_modified)[0]


def count_object(minio_client: Minio, bucket_name: str, prefix: str | None) -> int:
    """Get the number of object in bucket with prefix {reference}/{label} sub-folders.

    Args:
        minio_client (Minio): An instance of minio client.
        bucket_name (str): Name of the bucket to inspect.
        prefix (str): Prefix in the bucket to inspect.

    Returns:
        int: Number of objects.
    """
    try:
        objects = minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)
        objects_list = list(objects)
    except (S3Error, ServerError) as err:
        pytest.fail(f"Encountered error: {err}")
    return len(objects_list)
