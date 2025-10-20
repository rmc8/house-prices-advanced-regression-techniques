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
    return mo.md("# 高度な特徴量エンジニアリング v2")

# --- Outlier Removal ---
@app.cell
def __(df_train_raw, mo):
    mo.md("## 1. 外れ値の除去")
    df_train = df_train_raw.drop(
        df_train_raw[(df_train_raw['GrLivArea'] > 4000) & (df_train_raw['SalePrice'] < 300000)].index
    )
    md_outlier = mo.md(f"外れ値を2件除去しました。データ数: {len(df_train_raw)} -> {len(df_train)}")
    return df_train, md_outlier

# --- Log Transform Target ---
@app.cell
def __(df_train, mo, np, plt, sns):
    mo.md("## 2. 目的変数の対数変換")
    df_train["SalePrice_Log"] = np.log1p(df_train["SalePrice"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_train["SalePrice_Log"], kde=True, ax=ax)
    ax.set_title("対数変換後のSalePriceの分布")
    return ax, df_train, fig

# --- Combine Data ---
@app.cell
def __(df_test_raw, df_train, mo, pd):
    mo.md("## 3. データの結合")
    y_train_log = df_train["SalePrice_Log"]
    train_ids = df_train["Id"]
    test_ids = df_test_raw["Id"]
    df_train_features = df_train.drop(["Id", "SalePrice", "SalePrice_Log"], axis=1)
    df_test_features = df_test_raw.drop("Id", axis=1)
    all_data = pd.concat((df_train_features, df_test_features)).reset_index(drop=True)
    md_combine = mo.md(f"結合後の全データサイズ: {all_data.shape}")
    return all_data, md_combine, test_ids, train_ids, y_train_log

# --- Impute Missing Values ---
@app.cell
def __(all_data, mo, np, pd):
    mo.md("## 4. 欠損値の補完")
    # (Omitted for brevity, same as before)
    for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType'):
        all_data[col] = all_data[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
        all_data[col] = all_data[col].fillna(0)
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType'):
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    md_impute = mo.md(f"欠損値の補完が完了しました。")
    return all_data, md_impute

# --- Advanced Feature Engineering v2 ---
@app.cell
def __(all_data, mo):
    mo.md("## 5. 高度な特徴量エンジニアリング v2")
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    all_data['OverallQual_sq'] = all_data['OverallQual'] ** 2
    all_data['GrLivArea_sq'] = all_data['GrLivArea'] ** 2
    all_data['Qual_x_GrLivArea'] = all_data['OverallQual'] * all_data['GrLivArea']
    md_feature = mo.md("総面積、2乗、交互作用などの新しい特徴量を作成しました。")
    return all_data, md_feature

# --- Skewness Transform ---
@app.cell
def __(all_data, mo, np):
    mo.md("## 6. 歪んだ数値特徴量の対数変換")
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = all_data[numeric_feats].apply(lambda x: x.skew())
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
    md_skew = mo.md(f"歪度を修正した特徴量の数: **{len(skewed_feats)}**")
    return all_data, md_skew

# --- One-Hot Encoding ---
@app.cell
def __(all_data, mo, pd):
    mo.md("## 7. カテゴリカル特徴量の数値化 (One-Hot Encoding)")
    all_data_final = pd.get_dummies(all_data).reset_index(drop=True)
    md_ohe = mo.md(f"One-Hotエンコーディング後の最終的な特徴量数: **{all_data_final.shape[1]}**")
    return all_data_final, md_ohe

# --- Save Processed Data v2 ---
@app.cell
def __(DATA_DIR, all_data_final, mo, test_ids, y_train_log):
    mo.md("## 8. 処理済みデータ(v2)の保存")
    processed_data_path = DATA_DIR / "all_data_v2.pkl"
    target_data_path = DATA_DIR / "y_train_log_v2.pkl"
    test_ids_path = DATA_DIR / "test_ids_v2.pkl"
    all_data_final.to_pickle(processed_data_path)
    y_train_log.to_pickle(target_data_path)
    test_ids.to_pickle(test_ids_path)
    md_save = mo.md("さらに新しい特徴量を含む処理済みデータをファイルに保存しました。")
    return md_save,


if __name__ == "__main__":
    app.run()