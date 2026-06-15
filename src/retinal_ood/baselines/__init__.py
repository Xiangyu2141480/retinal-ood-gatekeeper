"""Lightweight unsupervised OOD baselines for dissertation comparisons."""

from retinal_ood.baselines.feature_distance import FeatureDistanceDetector
from retinal_ood.baselines.image_statistics import ImageStatisticsDetector

__all__ = ["FeatureDistanceDetector", "ImageStatisticsDetector"]
