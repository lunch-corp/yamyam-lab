# -*- coding: utf-8 -*-
"""
Region preprocessing module for yamyam-lab.

This module provides functionality for creating walking regions based on H3 cells
for restaurant recommendation systems.
"""

from .builder import build_walking_regions

__all__ = ["build_walking_regions"]
