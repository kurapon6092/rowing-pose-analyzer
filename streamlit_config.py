"""
Streamlit設定の調整
"""

import os
import sys

# OpenCV環境変数の設定
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'

# エラーを回避するための設定
import cv2
import streamlit as st

# OpenCVのバックエンドを設定
if hasattr(cv2, 'CAP_V4L'):
    cv2.CAP_V4L

print("OpenCV configuration completed successfully")
