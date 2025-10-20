import marimo as mo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import pathlib
import io

__generated_with = "0.17.0"
app = mo.App(width="full")


# --- Setup Cell ---
@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pathlib
    import io
    import japanize_matplotlib

    # Define paths
    THIS_DIR = pathlib.Path(__file__).parent
    DATA_DIR = THIS_DIR.parent / "data"

    # Load data
    df_train_raw = pd.read_csv(DATA_DIR / "train.csv")
    df_test_raw = pd.read_csv(DATA_DIR / "test.csv")

    return DATA_DIR, df_test_raw, df_train_raw, io, mo, np, pd, plt, sns


# --- Main Title ---
@app.cell
def __(mo):
    return mo.md("# データクリーニングと特徴量エンジニアリング - Part 1")


# --- Outlier Removal ---
@app.cell
def __(df_train_raw, mo):
    mo.md("""## 1. 外れ値の除去
    
    EDAで確認した `GrLivArea` が4000以上で価格が不釣り合いに低い外れ値を除去します。
    """)

    original_shape = df_train_raw.shape
    df_train = df_train_raw.drop(
        df_train_raw[
            (df_train_raw["GrLivArea"] > 4000) & (df_train_raw["SalePrice"] < 300000)
        ].index
    )
    cleaned_shape = df_train.shape

    result_md_outlier = mo.md(f"""
    - 元のデータ数: {original_shape[0]}
    - 除去後のデータ数: {cleaned_shape[0]}
    - 除去されたデータ数: {original_shape[0] - cleaned_shape[0]}
    """)

    return cleaned_shape, df_train, original_shape, result_md_outlier


# --- Log Transform Target ---
@app.cell
def __(df_train, mo, np, plt, sns):
    mo.md("""## 2. 目的変数の対数変換
    
    `SalePrice`を対数変換し、分布を正規分布に近づけます。
    """)
    df_train["SalePrice_Log"] = np.log1p(df_train["SalePrice"])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_train["SalePrice_Log"], kde=True, ax=ax)
    ax.set_title("対数変換後のSalePriceの分布")

    return ax, df_train, fig


# --- Combine Data ---
@app.cell
def __(df_test_raw, df_train, mo, pd):
    mo.md("""## 3. データの結合
    
    欠損値処理や特徴量エンジニアリングを学習データとテストデータで一貫して行うため、
    一度2つのデータセットを結合します。
    """)

    y_train_log = df_train["SalePrice_Log"]
    train_ids = df_train["Id"]
    test_ids = df_test_raw["Id"]

    df_train_features = df_train.drop(["Id", "SalePrice", "SalePrice_Log"], axis=1)
    df_test_features = df_test_raw.drop("Id", axis=1)

    all_data = pd.concat((df_train_features, df_test_features)).reset_index(drop=True)

    result_md_combine = mo.md(f"""
    - 結合後の全データサイズ: {all_data.shape}
    - 学習データの特徴量数: {df_train_features.shape}
    - テストデータの特徴量数: {df_test_features.shape}
    """)

    return (
        all_data,
        df_test_features,
        df_train_features,
        result_md_combine,
        test_ids,
        train_ids,
        y_train_log,
    )


# --- Impute Missing Values ---
@app.cell
def __(all_data, mo, np, pd):
    mo.md("## 4. 欠損値の補完")

    # カテゴリカル変数: NA -> 'None'
    for col in (
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "MasVnrType",
    ):
        all_data[col] = all_data[col].fillna("None")

    # 数値変数: NA -> 0
    for col in (
        "GarageYrBlt",
        "GarageArea",
        "GarageCars",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "MasVnrArea",
    ):
        all_data[col] = all_data[col].fillna(0)

    # LotFrontage: Neighborhoodの中央値で補完
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    # カテゴリカル変数: NA -> モード (最頻値)
    for col in (
        "MSZoning",
        "Electrical",
        "KitchenQual",
        "Exterior1st",
        "Exterior2nd",
        "SaleType",
    ):
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    # Functional: NA -> 'Typ'
    all_data["Functional"] = all_data["Functional"].fillna("Typ")

    # 確認
    missing_after = all_data.isnull().sum().sum()
    result_md_impute = mo.md(f"補完後の残りの欠損値の数: **{missing_after}**")

    return all_data, missing_after, result_md_impute


# --- Skewness Transform ---
@app.cell
def __(all_data, mo, np):
    mo.md("## 5. 歪んだ数値特徴量の対数変換")

    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = all_data[numeric_feats].apply(lambda x: x.skew())
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    skew_result_md = mo.md(f"歪度を修正した特徴量の数: **{len(skewed_feats)}**")

    return all_data, skew_result_md, skewed_feats


# --- One-Hot Encoding ---
@app.cell
def __(all_data, mo, pd):
    mo.md("## 6. カテゴリカル特徴量の数値化 (One-Hot Encoding)")

    all_data_final = pd.get_dummies(all_data).reset_index(drop=True)

    final_shape_md = mo.md(
        f"One-Hotエンコーディング後の最終的な特徴量数: **{all_data_final.shape[1]}**"
    )

    return all_data_final, final_shape_md


# --- Save Processed Data ---
@app.cell
def __(DATA_DIR, all_data_final, mo, test_ids, y_train_log):
    mo.md("## 7. 処理済みデータの保存")

    # Define save paths
    processed_data_path = DATA_DIR / "all_data_final.pkl"
    target_data_path = DATA_DIR / "y_train_log.pkl"
    test_ids_path = DATA_DIR / "test_ids.pkl"

    # Save the dataframes
    all_data_final.to_pickle(processed_data_path)
    y_train_log.to_pickle(target_data_path)
    test_ids.to_pickle(test_ids_path)

    save_md = mo.md(
        "特徴量エンジニアリング済みのデータを、次のモデリングノートブックで"
        "利用するためにファイルに保存しました。\n"
        f"- 特徴量データ: `{processed_data_path}`\n"
        f"- 目的変数データ: `{target_data_path}`\n"
        f"- テストIDデータ: `{test_ids_path}`"
    )
    return processed_data_path, save_md, target_data_path, test_ids_path


if __name__ == "__main__":
    app.run()
