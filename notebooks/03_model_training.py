import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import pathlib
    import japanize_matplotlib
    from sklearn.linear_model import Ridge, Lasso
    import xgboost as xgb
    import lightgbm as lgb

    # Define paths
    THIS_DIR = pathlib.Path(__file__).parent
    DATA_DIR = THIS_DIR.parent / "data"

    # Load ADVANCED v2 data
    all_data_final = pd.read_pickle(DATA_DIR / "all_data_v2.pkl")
    y_train_log = pd.read_pickle(DATA_DIR / "y_train_log_v2.pkl")
    test_ids = pd.read_pickle(DATA_DIR / "test_ids_v2.pkl")

    # Recreate train/test sets
    X = all_data_final.iloc[: len(y_train_log)]
    X_test = all_data_final.iloc[len(y_train_log) :]
    return (
        Lasso,
        Ridge,
        X,
        X_test,
        lgb,
        mo,
        np,
        pathlib,
        pd,
        test_ids,
        xgb,
        y_train_log,
    )


@app.cell
def _():
    return


@app.cell
def _(Lasso, Ridge, X, X_test, lgb, mo, xgb, y_train_log):
    mo.md("## 1. 各モデルの学習と予測")

    # Use best parameters from previous tuning
    ridge = Ridge(alpha=15)
    lasso = Lasso(alpha=0.0005, max_iter=10000)
    xgboost = xgb.XGBRegressor(random_state=42, n_jobs=-1, learning_rate=0.05, max_depth=3, n_estimators=400)
    lightgbm = lgb.LGBMRegressor(objective='regression', random_state=42, n_jobs=-1, learning_rate=0.05, n_estimators=800, num_leaves=5)

    # Fit models
    ridge.fit(X, y_train_log)
    lasso.fit(X, y_train_log)
    xgboost.fit(X, y_train_log)
    lightgbm.fit(X, y_train_log)

    # Get predictions (log scale)
    ridge_preds = ridge.predict(X_test)
    lasso_preds = lasso.predict(X_test)
    xgb_preds = xgboost.predict(X_test)
    lgb_preds = lightgbm.predict(X_test)

    md1 = mo.md("4つのベースモデルを学習させ、テストデータに対する予測値を出力しました。")
    return lasso_preds, lgb_preds, ridge_preds, xgb_preds


@app.cell
def _(
    lasso_preds,
    lgb_preds,
    mo,
    np,
    pathlib,
    pd,
    ridge_preds,
    test_ids,
    xgb_preds,
):
    mo.md("## 2. ブレンディングと最終提出")

    # Simple average blend
    # blended_preds_log = (ridge_preds + lasso_preds + xgb_preds + lgb_preds) / 4.0

    # Weighted average blend (giving more weight to tree-based models)
    blended_preds_log = (
        0.1 * ridge_preds + 0.1 * lasso_preds + 0.4 * xgb_preds + 0.4 * lgb_preds
    )

    # Inverse transform
    blended_predictions = np.expm1(blended_preds_log)

    # Create submission file
    submission_blended = pd.DataFrame(
        {"Id": test_ids, "SalePrice": blended_predictions}
    )

    submission_dir = "submissions"
    pathlib.Path(submission_dir).mkdir(exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    submission_path = f"{submission_dir}/submission_blended_{timestamp}.csv"

    submission_blended.to_csv(submission_path, index=False)

    md2 = mo.md(
        f"ブレンディングによる最終提出ファイルを作成しました！\n"
        f"ファイルパス: `{submission_path}`"
    )
    return


if __name__ == "__main__":
    app.run()
