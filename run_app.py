#!/usr/bin/env python3
"""
動画姿勢解析アプリケーション起動スクリプト
"""

import subprocess
import sys
import os

def check_dependencies():
    """必要な依存関係がインストールされているかチェック"""
    required_packages = [
        'streamlit',
        'opencv-python',
        'mediapipe',
        'numpy',
        'matplotlib',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """依存関係をインストール"""
    print("依存関係をインストール中...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("依存関係のインストールが完了しました。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"依存関係のインストールに失敗しました: {e}")
        return False

def generate_test_video():
    """テスト動画を生成"""
    if not os.path.exists('test_video.mp4'):
        print("テスト動画を生成中...")
        try:
            import generate_test_video
            generate_test_video.generate_test_video()
        except Exception as e:
            print(f"テスト動画の生成に失敗しました: {e}")

def main():
    """メイン関数"""
    print("🏃‍♂️ 動画姿勢解析アプリケーション")
    print("=" * 50)
    
    # 依存関係をチェック
    missing = check_dependencies()
    if missing:
        print(f"以下のパッケージが見つかりません: {', '.join(missing)}")
        print("依存関係をインストールしますか？ (y/n): ", end="")
        
        if input().lower() in ['y', 'yes']:
            if not install_dependencies():
                print("依存関係のインストールに失敗しました。手動でインストールしてください。")
                return
        else:
            print("依存関係をインストールしてからもう一度お試しください。")
            return
    
    # テスト動画を生成（存在しない場合）
    generate_test_video()
    
    # アプリケーションの説明
    print("\n📋 アプリケーション機能:")
    print("  ✅ 骨格トレース（MediaPipe使用）")
    print("  ✅ 頭の高さに水平線描画")
    print("  ✅ 腰の角度測定（リアルタイム）")
    print("  ✅ 最大・最小角度の記録")
    print("  ✅ 目線検出と表示")
    print("  ✅ 時系列データの可視化")
    
    print("\n🚀 アプリケーションを起動しています...")
    print("ブラウザで http://localhost:8501 にアクセスしてください")
    print("\n⚠️  アプリケーションを停止するには Ctrl+C を押してください")
    
    # Streamlitアプリを起動
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'main.py'])
    except KeyboardInterrupt:
        print("\n\n👋 アプリケーションを終了します")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
