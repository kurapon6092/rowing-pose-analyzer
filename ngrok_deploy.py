#!/usr/bin/env python3
"""
ngrokを使用した簡単Web公開スクリプト
"""

import subprocess
import sys
import time
import threading

def install_ngrok():
    """ngrokをインストール"""
    try:
        # Homebrewでngrokをインストール
        subprocess.run(["brew", "install", "ngrok"], check=True)
        print("✅ ngrokのインストールが完了しました")
        return True
    except subprocess.CalledProcessError:
        print("❌ ngrokのインストールに失敗しました")
        print("手動でインストールしてください: https://ngrok.com/download")
        return False
    except FileNotFoundError:
        print("❌ Homebrewが見つかりません")
        print("手動でngrokをインストールしてください: https://ngrok.com/download")
        return False

def start_streamlit():
    """Streamlitアプリを起動"""
    print("🚀 Streamlitアプリを起動中...")
    subprocess.Popen([sys.executable, "-m", "streamlit", "run", "main.py", "--server.port=8501"])
    time.sleep(3)  # 起動待機

def start_ngrok():
    """ngrokでトンネルを作成"""
    print("🌐 ngrokトンネルを作成中...")
    try:
        # ngrokを起動
        result = subprocess.run(
            ["ngrok", "http", "8501", "--log=stdout"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return True
    except Exception as e:
        print(f"ngrok起動エラー: {e}")
        return False

def main():
    print("🌐 ngrok Web公開ツール")
    print("=" * 40)
    
    # ngrokがインストールされているかチェック
    try:
        subprocess.run(["ngrok", "version"], capture_output=True, check=True)
        print("✅ ngrokが利用可能です")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ngrokが見つかりません")
        if input("インストールしますか？ (y/n): ").lower() == 'y':
            if not install_ngrok():
                return
        else:
            print("ngrokをインストールしてから再度お試しください")
            return
    
    # アプリを起動
    start_streamlit()
    
    print("\n📋 使用方法:")
    print("1. 別のターミナルで以下を実行:")
    print("   ngrok http 8501")
    print("2. 表示されるPublic URLをコピー")
    print("3. そのURLを共有すれば世界中からアクセス可能！")
    print("\n⚠️  終了するには Ctrl+C を押してください")
    
    try:
        # アプリを維持
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py", "--server.port=8501"])
    except KeyboardInterrupt:
        print("\n👋 アプリケーションを終了しています...")

if __name__ == "__main__":
    main()
