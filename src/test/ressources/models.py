"""Define models for orchestrator and inference API"""
from enum import Enum

from pydantic import BaseModel, Field


# Minio
class Buckets(str, Enum):
    """Define buckets on minio"""

    BASED_COINS = "based-coins"
    USER_COINS = "user-coins"
