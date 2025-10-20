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

    return DATA_DIR, io, mo, np, pd, plt, sns


# --- Main Title ---
@app.cell
def __(mo):
    return mo.md("# 探索的データ分析 (EDA) - Part 1")


# --- Load Data ---
@app.cell
def __(DATA_DIR, mo, pd):
    mo.md("## 1. データの読み込み")
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    return (df_train,)


# --- Head & Describe ---
@app.cell
def __(df_train, mo):
    mo.md("### データフレームの概観と基本統計量")

    head_view = mo.vstack(
        [mo.md("**`head()`による最初の5行:**"), mo.ui.table(df_train.head())]
    )

    describe_view = mo.vstack(
        [mo.md("**`describe()`による基本統計量:**"), mo.ui.table(df_train.describe())]
    )

    return describe_view, head_view


# --- Info ---
@app.cell
def __(df_train, io, mo):
    mo.md("### データ型と欠損値の概要 (info)")
    buffer = io.StringIO()
    df_train.info(buf=buffer)
    info_str = buffer.getvalue()
    return (mo.ui.code_editor(code=info_str, language="text"),)


# --- Missing Values Plot ---
@app.cell
def __(df_train, mo, plt):
    mo.md("## 2. 欠損値の可視化")
    missing = df_train.isnull().sum()
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)

    fig_missing, ax_missing = plt.subplots(figsize=(10, 8))
    missing.plot.barh(ax=ax_missing)
    ax_missing.set_title("欠損値のある列とその数")
    ax_missing.set_xlabel("欠損値の数")

    return (fig_missing,)


# --- SalePrice Distribution Plot ---
@app.cell
def __(df_train, mo, np, plt, sns):
    mo.md("## 3. 目的変数 (SalePrice) の分布")
    fig_dist, axes_dist = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(df_train["SalePrice"], kde=True, ax=axes_dist[0])
    axes_dist[0].set_title("SalePrice の分布")
    axes_dist[0].set_xlabel("価格")

    sns.histplot(np.log1p(df_train["SalePrice"]), kde=True, ax=axes_dist[1])
    axes_dist[1].set_title("log(SalePrice + 1) の分布")
    axes_dist[1].set_xlabel("対数変換後の価格")

    return (fig_dist,)


# --- Correlation Heatmap ---
@app.cell
def __(df_train, mo, np, plt, sns):
    mo.md("## 4. SalePriceとの相関が高い特徴量")
    k = 11
    corrmat = df_train.select_dtypes(include=np.number).corr()
    cols = corrmat.nlargest(k, "SalePrice")["SalePrice"].index
    cm = np.corrcoef(df_train[cols].values.T)

    fig_corr, ax_corr = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        cm,
        cbar=True,
        annot=True,
        square=True,
        fmt=".2f",
        annot_kws={"size": 10},
        yticklabels=cols.values,
        xticklabels=cols.values,
        ax=ax_corr,
    )
    ax_corr.set_title("SalePriceと相関の高い特徴量")

    return cols, fig_corr


# --- Scatter Plots ---
@app.cell
def __(cols, df_train, mo, plt, sns):
    mo.md("## 5. 主要特徴量とSalePriceの散布図")

    scatter_cols = cols[1:5]

    fig_scatter, axes_scatter = plt.subplots(2, 2, figsize=(15, 12))
    axes_scatter = axes_scatter.flatten()

    for i, col in enumerate(scatter_cols):
        sns.scatterplot(x=df_train[col], y=df_train["SalePrice"], ax=axes_scatter[i])
        axes_scatter[i].set_title(f"{col} vs SalePrice")

    plt.tight_layout()
    return (fig_scatter,)


if __name__ == "__main__":
    app.run()
