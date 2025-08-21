#!/usr/bin/env python3
"""
比較アプリケーション起動スクリプト
"""

import subprocess
import sys
import webbrowser
import time

def main():
    print("🆚 動画比較解析アプリを起動します")
    print("=" * 40)
    
    print("📋 起動中...")
    print("- 比較アプリ: http://localhost:8503")
    print("- 単体アプリ: http://localhost:8502 (既存)")
    
    try:
        # 比較アプリを起動
        print("\n🚀 比較アプリを起動中...")
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "comparison_app.py", "--server.port", "8503"
        ])
        
        print("✅ 比較アプリが起動しました")
        print("\n📱 アクセス方法:")
        print("- 比較解析: http://localhost:8503")
        print("- 単体解析: http://localhost:8502")
        
        # 3秒後にブラウザを開く
        time.sleep(3)
        try:
            webbrowser.open("http://localhost:8503")
        except:
            pass
        
        input("\n⚠️  アプリを停止するには Enter を押してください...")
        
    except KeyboardInterrupt:
        print("\n👋 アプリケーションを終了します")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
