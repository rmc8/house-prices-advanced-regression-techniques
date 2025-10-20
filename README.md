# 住宅価格予測：高度な回帰分析テクニック

このプロジェクトは、Kaggleコンペティション「[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)」のためのものです。アイオワ州エイムズ市の住宅に関する79種類の説明変数を用いて、住宅の販売価格を予測することを目的としています。

## はじめに

### 前提条件

- Python 3.8+
- [Kaggleアカウント](https://www.kaggle.com/) と APIトークン (`kaggle.json`)

### インストール

1.  **Kaggle APIのセットアップ:**
    まず、公式のKaggle CLIをインストールします。
    ```bash
    pip install kaggle
    ```
    次に、`~/.kaggle/` ディレクトリに `kaggle.json` APIトークンを配置します。

2.  **依存関係のインストール:**
    仮想環境の作成を推奨します。このプロジェクトでは、ノートブックに [Marimo](https://marimo.io/) を使用します。
    ```bash
    # 後で requirements.txt を作成することもできます
    pip install marimo pandas numpy scikit-learn matplotlib seaborn
    ```

3.  **データのダウンロード:**
    ダウンロードの前に、[Kaggleウェブサイト](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/rules)でコンペティションの規約に同意する必要があります。

    その後、以下のコマンドを実行してデータをダウンロードし、展開します。
    ```bash
    mkdir -p data
    kaggle competitions download -c house-prices-advanced-regression-techniques -p data
    unzip data/house-prices-advanced-regression-techniques.zip -d data
    rm data/house-prices-advanced-regression-techniques.zip
    ```

## プロジェクト構造

```
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── data_description.txt
├── notebooks/
│   ├── 01_data_exploration.py
│   ├── 02_feature_engineering.py
│   └── 03_model_training.py
├── src/
├── README.md
└── GEMINI.md
```

## 使い方

`notebooks/` ディレクトリ内のノートブックはMarimoで構築されています。ノートブックを実行するには、ターミナルで以下のコマンドを実行してください。

```bash
marimo run notebooks/01_data_exploration.py
```