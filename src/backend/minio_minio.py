"""Minio client."""
import base64
import logging
import os
from io import BytesIO

import requests
from minio import Minio
from minio.error import S3Error
from PIL import Image


class MinioClient:
    """Minio client.
    Class to interact with the minio server.
    """

    def __init__(
        self,
        endpoint="172.18.0.4:9000",  # It changes at each docker compose up TODO: Fix this
        access_key="minioadmin",  # Could be 172.19.0.3:9000 can be a range up to a lot
        secret_key="minioadmin",
        secure=False,
    ):
        """Initialize the minio client.
        Args:
            endpoint (str): The endpoint of the minio server.
            access_key (str): The access key of the minio server.
            secret_key (str): The secret key of the minio server.
            secure (bool): Whether to use secure connection or not.
        """
        self.client = Minio(endpoint, access_key, secret_key, secure=secure)

    def create_bucket(self, bucket_name):
        """Create a bucket on the minio server.

        Args:
            bucket_name( str): The name of the bucket to be created.
        """
        try:
            self.client.make_bucket(bucket_name)
            print(f"Bucket {bucket_name} created successfully.")
        except S3Error as err:
            print(err)

    def delete_bucket(self, bucket_name):
        """Delete a bucket from the minio server.


        Args:
            bucket_name (str): The name of the bucket to be deleted.
        """
        try:
            self.client.remove_bucket(bucket_name)
            print(f"Bucket {bucket_name} deleted successfully.")
        except S3Error as err:
            print(err)

    def list_buckets(self):
        """List all buckets on the minio server."""
        try:
            buckets = self.client.list_buckets()
            for bucket in buckets:
                print(bucket.name, bucket.creation_date)
        except S3Error as err:
            print(err)

    def delete_policy(self, bucket_name):
        """Remove the policy of a bucket.

        Args:
            bucket_name (str): The name of the bucket.
        """
        try:
            self.client.delete_bucket_policy(bucket_name)
            print(f"Bucket policy deleted successfully for {bucket_name}.")
        except S3Error as err:
            print(err)

    def add_single_object(self, bucket_name, object_name, file_path):
        """Add a single object to a bucket.
        Args:
            bucket_name (str): The name of the bucket.
            object_name (str): The name of the object to add.
            file_path (str): The local path to the file to add."""
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
            print(f"Object {object_name} added successfully to bucket {bucket_name}.")
        except S3Error as err:
            print(err)

    def add_multiple_objects(self, bucket_name, folder_path, object_names=None):
        """Add multiple objects to a bucket.

        Args:
            bucket_name (str): The name of the bucket.
            folder_path (str): Path to the folder containing the files to add.
            object_names (list[str], optional): List of object names to use instead of file names.
        """
        try:
            object_names = object_names or []
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    if object_names is not None and file_name in object_names:
                        object_name = object_names[object_names.index(file_name)]
                        self.client.fput_object(bucket_name, object_name, file_path)
        except S3Error as err:
            print(err)

    def send_directory_of_pictures(self, bucket_name, folder_path, object_names=None):
        """Send a directory of pictures to a bucket on the first run of the application.

        Args:
            bucket_name (str): The name of the bucket.
            folder_path (str): Path to the folder containing the files to add.
            object_names (lis[str]): List of object names to use instead of file names.
        """
        try:
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    if not object_names or file_name in object_names:
                        object_name = (
                            file_name
                            if not object_names
                            else object_names[object_names.index(file_name)]
                        )
                        self.client.fput_object(bucket_name, object_name, file_path)
        except S3Error as err:
            print(err)

    def get_object_by_name(self, bucket_name: str, object_name: str):
        """Get an object from a bucket by its name.

        Args:
            bucket_name (str): The name of the bucket.
            object_name (str): The name of the object to get.

        Returns:
            bytes: The contents of the object.
        """
        try:
            data = self.client.get_object(bucket_name, object_name)
            return data.read()
        except S3Error as err:
            return err

    def download_object(self, bucket_name, object_name, file_path):
        """Download an object from a bucket.

        Args:
            bucket_name (str): The name of the bucket.
            object_name (str): Object name to download.
            file_path (str): File path to download the object to.
        """
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            print(
                f"Object {object_name} downloaded successfully from bucket {bucket_name}."
            )
        except S3Error as err:
            print(err)

    def delete_all_objects(self, bucket_name):
        """Delete all objects from a bucket.

        Args:
            bucket_name (str): The name of the bucket.
        """
        try:
            objects = self.client.list_objects(bucket_name, recursive=True)
            for obj in objects:
                self.client.remove_object(bucket_name, obj.object_name)
            print(f"All objects deleted successfully from bucket {bucket_name}.")
        except S3Error as err:
            print(err)


class DisplayMinio(MinioClient):
    """Class to display objects from a minio server.

    Args:
        MinioClient (obj): The MinioClient object.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_all_files(self, bucket_name: str):
        """Get all files in a bucket.

        Args:
            bucket_name (str): The name of the bucket.

        Returns:
            list: A list of all object names in the bucket.
        """
        try:
            objects = self.client.list_objects(bucket_name, recursive=True)
            object_names = []
            for obj in objects:
                object_names.append(obj.object_name)
            return object_names
        except S3Error as err:
            return err

    def display_pictures(self, bucket_name: str, num_pictures: int):
        """Display a specified number of pictures from a bucket.

        Args:
            bucket_name (str): The name of the bucket.
            num_pictures (int): The number of pictures to display."""
        try:
            objects = self.client.list_objects(bucket_name, recursive=True)
            count = 0
            for obj in objects:
                if obj.is_dir:
                    continue
                response = requests.get(
                    self.client.presigned_get_object(bucket_name, obj.object_name),
                    timeout=60,
                )
                img = Image.open(BytesIO(response.content))
                img.show()
                count += 1
                if count >= num_pictures:
                    break
        except S3Error as err:
            print(err)

    def display_picture_by_name(self, bucket_name: str, object_name: str) -> None:
        """Display a picture from a bucket by its name.

        Args:
            bucket_name (str): The name of the bucket.
            object_name (str): The name of the object to display.
        """
        # change extension from anything to webp
        object_name = object_name.split(".")[0] + ".webp"
        try:
            response = requests.get(
                self.client.presigned_get_object(bucket_name, object_name),
                timeout=60,
            )
            img = Image.open(BytesIO(response.content))
            img.show()
        except S3Error as err:
            print(err)

    def get_image_data_url(self, bucket_name: str, object_name: str) -> str:
        """Transform an image in BytesIO format to a data URL.

        Args:
            bucket_name: The name of the bucket.
            object_name: The name of the object to get.

        Returns:
            str: The data URL of the image.
        """
        object_name = object_name.split(".")[0] + ".webp"
        response = self.client.get_object(bucket_name, object_name)
        img = Image.open(BytesIO(response.data))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{img_str}"
