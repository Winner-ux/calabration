"""RGB/IR processing core shared by the two desktop applications."""

from .models import PairSpec, ProcessMessage, VideoMetadata
from .paths import ProjectPaths

__all__ = ["PairSpec", "ProcessMessage", "ProjectPaths", "VideoMetadata"]
