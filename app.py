"""
動画姿勢解析Webアプリケーション - メインエントリーポイント
公開版では main.py の内容をそのまま実行
"""

import streamlit as st

# main.pyの内容をインポート
import sys
import os
sys.path.append(os.path.dirname(__file__))

# main.pyを実行
exec(open('main.py').read())
