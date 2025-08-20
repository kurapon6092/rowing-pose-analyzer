# GitHub公開手順

## 1. GitHubでリポジトリ作成後、以下のコマンドを実行:

```bash
# デフォルトブランチをmainに変更
git branch -M main

# GitHubリポジトリと連携（YOUR_USERNAMEを実際のユーザー名に変更）
git remote add origin https://github.com/YOUR_USERNAME/rowing-pose-analyzer.git

# プッシュ
git push -u origin main
```

## 2. Streamlit Community Cloud で公開:

1. https://share.streamlit.io にアクセス
2. GitHubアカウントでログイン
3. "New app" をクリック
4. 設定入力:
   - Repository: YOUR_USERNAME/rowing-pose-analyzer
   - Branch: main  
   - Main file path: main.py
5. "Deploy!" をクリック

## 3. 公開完了！

数分後に以下のようなURLでアクセス可能:
https://YOUR_USERNAME-rowing-pose-analyzer-main-xxxxx.streamlit.app

## 4. 更新方法

コードを修正したら:
```bash
git add .
git commit -m "機能追加: 新機能の説明"
git push
```

自動的にアプリも更新されます！
