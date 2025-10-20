import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import pathlib
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import StackingRegressor
    import xgboost as xgb
    import lightgbm as lgb

    # Define paths
    THIS_DIR = pathlib.Path(__file__).parent
    DATA_DIR = THIS_DIR.parent / "data"

    # Load our best dataset
    all_data_final = pd.read_pickle(DATA_DIR / "all_data_v2.pkl")
    y_train_log = pd.read_pickle(DATA_DIR / "y_train_log_v2.pkl")
    test_ids = pd.read_pickle(DATA_DIR / "test_ids_v2.pkl")

    # Recreate original train/test sets
    X = all_data_final.iloc[:len(y_train_log)]
    X_test = all_data_final.iloc[len(y_train_log):]
    return (
        Lasso,
        Ridge,
        StackingRegressor,
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
def _(Lasso, Ridge, StackingRegressor, lgb, mo, xgb):
    mo.md("## 1. 最強モデルの定義")
    # These are our best models from notebook 03
    ridge_best = Ridge(alpha=15)
    lasso_best = Lasso(alpha=0.0005, max_iter=10000)
    xgb_best = xgb.XGBRegressor(random_state=42, n_jobs=-1, learning_rate=0.05, max_depth=3, n_estimators=400)
    lgb_best = lgb.LGBMRegressor(objective='regression', random_state=42, n_jobs=-1, learning_rate=0.05, n_estimators=800, num_leaves=5)

    # This is our best stacking model
    base_model = StackingRegressor(
        estimators=[
            ('ridge', ridge_best), 
            ('lasso', lasso_best), 
            ('xgb', xgb_best),
            ('lgb', lgb_best)
        ],
        final_estimator=Ridge(alpha=10)
    )
    return (base_model,)


@app.cell
def _(X, X_test, base_model, mo, np, y_train_log):
    mo.md("## 2. テストデータの予測と疑似ラベルの選択")

    # Fit the base model on original training data
    base_model.fit(X.values, y_train_log.values)

    # Predict on the test set
    test_preds_log = base_model.predict(X_test.values)

    # Select high-confidence predictions (e.g., top and bottom 5%)
    N_PSEUDO = int(len(X_test) * 0.05)
    high_conf_indices = np.concatenate([
        np.argsort(test_preds_log)[-N_PSEUDO:],
        np.argsort(test_preds_log)[:N_PSEUDO]
    ])

    # Create pseudo-labeled data
    X_pseudo = X_test.iloc[high_conf_indices]
    y_pseudo = test_preds_log[high_conf_indices]

    md1 = mo.md(f"テストデータの予測から、信頼度の高い**{len(X_pseudo)}件**を疑似ラベルとして選択しました。")
    return X_pseudo, y_pseudo


@app.cell
def _(X, X_pseudo, base_model, mo, np, pd, y_pseudo, y_train_log):
    mo.md("## 3. 疑似ラベルを使ったモデルの再学習")

    # Augment the training data
    X_augmented = pd.concat([X, X_pseudo])
    y_augmented = np.concatenate([y_train_log, y_pseudo])

    # Re-train the model on the augmented data
    retrained_model = base_model
    retrained_model.fit(X_augmented.values, y_augmented)

    md2 = mo.md(f"元の学習データと疑似ラベルデータを結合し、モデルを再学習させました。")
    return (retrained_model,)


@app.cell
def _(X_test, mo, np, pathlib, pd, retrained_model, test_ids):
    mo.md("## 4. 最終提出ファイルの作成 (疑似ラベルモデル)")

    # Final predictions
    final_predictions_log = retrained_model.predict(X_test.values)
    final_predictions = np.expm1(final_predictions_log)

    # Create submission file
    final_submission = pd.DataFrame({"Id": test_ids, "SalePrice": final_predictions})

    submission_dir = "submissions"
    pathlib.Path(submission_dir).mkdir(exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    submission_path = f"{submission_dir}/submission_pseudo_{timestamp}.csv"

    final_submission.to_csv(submission_path, index=False)

    md3 = mo.md(
        f"疑似ラベリングで再学習したモデルで、最終提出ファイルを作成しました！\n"
        f"ファイルパス: `{submission_path}`"
    )
    return


if __name__ == "__main__":
    app.run()
