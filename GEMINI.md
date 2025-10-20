# GEMINI.md

**Speak in Japanese!!**

## プロジェクト概要

このプロジェクトは、Kaggleのコンペティション「[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)」のためのものです。目的は、アイオワ州エイムズ市の住宅に関する79種類の説明変数に基づき、住宅の販売価格を予測することです。

これは回帰問題であり、評価指標は、予測値の対数と観測された販売価格の対数との間の二乗平均平方根誤差（RMSE）です。

## 利用開始

### 1. データのダウンロード

このプロジェクトでは、公式のKaggle CLIを使用してデータセットをダウンロード・管理します。

**前提条件:**
1.  **Kaggle CLIのインストール:** もしまだなら、ライブラリをインストールします。
    ```bash
    pip install kaggle
    ```
2.  **API認証情報:** KaggleアカウントページからAPI認証情報（`kaggle.json`）をセットアップしておく必要があります。
3.  **コンペ規約への同意:** ダウンロードの前に、Kaggleのウェブサイトでコンペティションの規約に同意する必要があります。
    [https://www.kaggle.com/c/house-prices-advanced-regression-techniques/rules](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/rules)

**ダウンロードコマンド:**
前提条件が満たされたら、プロジェクトのルートから以下のコマンドを実行して、`data/`ディレクトリにデータをダウンロードします。
```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p data
```
これによりzipファイルがダウンロードされるので、`data`ディレクトリに展開してください。

### 2. プロジェクト構造

推奨されるプロジェクト構造は以下の通りです。

```
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── data_description.txt
│   └── sample_submission.csv
├── notebooks/
│   ├── 01_data_exploration.py
│   ├── 02_feature_engineering.py
│   └── 03_model_training.py
├── src/
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/
│       └── visualize.py
├── requirements.txt
└── GEMINI.md
```

## ビルドと実行

*TODO: コード開発後に、プロジェクトのビルドと実行に関する指示を追加します。*

## 提出ファイルの作成

モデルの学習と評価が完了したら、Kaggleに提出するためのファイルを作成します。

1.  **ベストモデルの選択:** `03_model_training.py`で評価したモデルの中から、最も性能の良いモデル（例: XGBoost）を選択します。
2.  **モデルの再学習:** 選択したモデルを、今度は**すべての**学習データ（`X`と`y_train_log`）を使って学習させます。
3.  **テストデータでの予測:** 学習済みモデルを使い、テストデータ（`X_test`）の住宅価格を予測します。
4.  **予測値の逆変換:** 予測された値は対数スケールになっているため、`np.expm1`を使って元の価格スケールに戻します。
5.  **`submission.csv`の作成:** Kaggleの指定するフォーマット（`Id`と`SalePrice`の2列）に従って、提出用CSVファイルを作成します。

## 開発規約

*   **コードスタイル:** PythonコードはPEP 8に従います。
*   **ノートブック:** このプロジェクトでは、インタラクティブなノートブックとして [Marimo](https://marimo.io/) を使用します。すべてのノートブックは、標準のPython（`.py`）ファイルとして`notebooks`ディレクトリに保存されます。
*   **禁忌 (Marimoのルール):** 各セルで定義される変数は、ノートブック全体で一意である必要があります。異なるセルで同じ名前の変数を再定義できません。
*   **ソースコード:** 再利用可能なコードは`src`ディレクトリに配置します。
*   **データ:** 生データは`data`ディレクトリに保管します。
*   **依存関係:** すべてのPython依存関係は`requirements.txt`ファイルに記載します。

## 学習した教訓 (Marimo利用時の注意点)

このプロジェクトを通して、`marimo`ノートブックを利用する上での重要な教訓がいくつか得られました。

1.  **セルの出力は`return`文で:**
    Jupyterとは異なり、Marimoのセルで計算結果（表、グラフなど）をブラウザ上に表示するには、そのオブジェクトをセルの`return`文で返す必要があります。セルの途中で評価されただけの式は表示されません。複数のオブジェクトを表示したい場合は、`mo.vstack([...])`などで一つの要素にまとめて返します。

2.  **変数の単一定義ルール (禁忌):**
    既に「開発規約」にも記載しましたが、Marimoでは同じ変数名を複数のセルで定義（再定義）することはできません。これはノートブック全体のリアクティブな動作を保証するための重要なルールです。変数名はセルごとに一意にする必要があります。

3.  **日本語フォントと`matplotlib`:**
    `japanize_matplotlib`は日本語表示に便利なライブラリですが、`seaborn`の`sns.set_theme()`のようなスタイル設定関数によって、フォント設定が上書きされてしまう場合があります。文字化けが発生した際は、`matplotlib`の`rcParams`がどのように設定されているか、また設定が上書きされていないかを確認することが重要です。今回は`sns.set_theme()`を削除することで解決しました。