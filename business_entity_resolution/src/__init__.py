"""
Business Entity Resolution Package
Amazon ML Challenge 2026
"""

from .config import config
from .preprocessing import preprocess_record
from .blocking import MultiChannelBlocker
from .features import extract_pair_features, compute_group_relative_features
from .model import EntityMatchingModel
from .evaluation import evaluate_macro_f05
