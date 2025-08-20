# 🚀 Webアプリケーション公開ガイド

## Streamlit Community Cloudで無料公開

### 📋 事前準備

1. **GitHubアカウント**が必要
2. **Streamlit Community Cloud**アカウント（GitHubでサインアップ可能）

### 🔧 公開手順

#### Step 1: GitHubリポジトリ作成
```bash
# 1. GitHubで新規リポジトリを作成（例：rowing-pose-analyzer）
# 2. ローカルでGit初期化
cd rowingapp_new
git init
git add .
git commit -m "Initial commit: Rowing pose analysis app"

# 3. GitHubリポジトリと連携
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/rowing-pose-analyzer.git
git push -u origin main
```

#### Step 2: Streamlit Community Cloud設定
1. **[share.streamlit.io](https://share.streamlit.io)** にアクセス
2. GitHubアカウントでサインイン
3. 「New app」をクリック
4. リポジトリ情報を入力：
   - **Repository**: `YOUR_USERNAME/rowing-pose-analyzer`
   - **Branch**: `main`
   - **Main file path**: `main.py`

#### Step 3: デプロイ
- 「Deploy!」ボタンをクリック
- 数分でアプリが公開されます
- URLは `https://YOUR_USERNAME-rowing-pose-analyzer-main-xxxxx.streamlit.app`

### 🔧 設定ファイル

#### `.streamlit/config.toml`
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
maxUploadSize = 200
maxMessageSize = 200

[browser]
gatherUsageStats = false
```

### 📊 公開後のURL例
```
https://your-username-rowing-pose-analyzer-main-abc123.streamlit.app
```

### ⚡ 更新方法
GitHubリポジトリにプッシュすると自動的にアプリが更新されます：
```bash
git add .
git commit -m "Update features"
git push
```

### 🛠️ トラブルシューティング

#### よくある問題
1. **OpenCV エラー**: `opencv-python-headless`を使用
2. **メモリ不足**: 大きな動画ファイルは制限される場合あり  
3. **タイムアウト**: 処理時間の長い動画は分割推奨

#### リソース制限
- **CPU**: 1 vCPU
- **メモリ**: 1GB RAM
- **ストレージ**: 限定的
- **ファイルサイズ**: 最大200MB

### 🌟 カスタムドメイン（オプション）
有料プランでカスタムドメインを設定可能：
```
https://your-domain.com
```

## 他の公開オプション

### 💰 有料オプション
- **Heroku**: $5-25/月
- **Google Cloud Run**: 使用量ベース
- **AWS ECS**: 使用量ベース
- **Railway**: $5/月から

### 🏠 セルフホスティング
```bash
# Docker使用
docker build -t rowing-analyzer .
docker run -p 8501:8501 rowing-analyzer
```
