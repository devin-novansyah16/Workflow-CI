"""
modelling.py (MLProject version)
==================================
Script pemodelan untuk dijalankan via MLflow Project.
Mendukung parameter via argparse sehingga bisa dikontrol
dari MLProject entry point maupun GitHub Actions.

Author  : Devin Novansyah
Dataset : Titanic - Machine Learning from Disaster
"""

import os
import json
import argparse
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# ARGPARSE
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Titanic - MLProject Modelling')
    parser.add_argument('--n_estimators'    , type=int  , default=100   )
    parser.add_argument('--max_depth'       , type=int  , default=6     )
    parser.add_argument('--min_samples_split', type=int , default=2     )
    parser.add_argument('--max_features'    , type=str  , default='sqrt')
    parser.add_argument('--test_size'       , type=float, default=0.2   )
    parser.add_argument('--random_state'    , type=int  , default=42    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# HELPER: BUAT ARTEFAK
# ──────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, tmp_dir: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['Not Survived', 'Survived'])
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('Confusion Matrix\nDevin Novansyah', fontsize=13)
    plt.tight_layout()
    path = os.path.join(tmp_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


def plot_feature_importance(model, feature_names, tmp_dir: str) -> str:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(12, len(feature_names))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(top_n), importances[indices[:top_n]],
           color='steelblue', edgecolor='white')
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in indices[:top_n]],
                       rotation=40, ha='right', fontsize=10)
    ax.set_title('Feature Importance\nDevin Novansyah', fontsize=13)
    ax.set_ylabel('Importance Score')
    plt.tight_layout()
    path = os.path.join(tmp_dir, 'feature_importance.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


def save_classification_report(y_true, y_pred, tmp_dir: str) -> str:
    report = classification_report(y_true, y_pred,
                                   target_names=['Not Survived', 'Survived'])
    path = os.path.join(tmp_dir, 'classification_report.txt')
    with open(path, 'w') as f:
        f.write('Classification Report - Titanic\n')
        f.write('Author: Devin Novansyah\n')
        f.write('=' * 45 + '\n')
        f.write(report)
    return path


def save_params_json(params: dict, tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, 'params.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2)
    return path


# ──────────────────────────────────────────────
# MAIN TRAINING
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    print('=' * 55)
    print('  MLPROJECT MODELLING - TITANIC')
    print('  Author: Devin Novansyah')
    print('=' * 55)

    # ── Load data ──────────────────────────────
    data_path = os.path.join(os.path.dirname(__file__),
                             'titanic_preprocessing', 'train_preprocessed.csv')
    print(f'\n[1/4] Memuat data dari: {data_path}')
    df = pd.read_csv(data_path)
    df = df.select_dtypes(include='number')

    X = df.drop(columns=['Survived'])
    y = df['Survived']
    print(f'      ✅ Shape X: {X.shape} | Shape y: {y.shape}')

    # ── Split ──────────────────────────────────
    print('[2/4] Train/test split...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )
    print(f'      Train: {X_train.shape} | Test: {X_test.shape}')

    # ── Train ──────────────────────────────────
    print('[3/4] Melatih model...')
    model = RandomForestClassifier(
        n_estimators     = args.n_estimators,
        max_depth        = args.max_depth,
        min_samples_split= args.min_samples_split,
        max_features     = args.max_features,
        random_state     = args.random_state
    )
    model.fit(X_train, y_train)

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    cv_scores   = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    roc  = roc_auc_score(y_test, y_pred_prob)

    print(f'      Accuracy  : {acc:.4f}')
    print(f'      Precision : {prec:.4f}')
    print(f'      Recall    : {rec:.4f}')
    print(f'      F1-Score  : {f1:.4f}')
    print(f'      ROC-AUC   : {roc:.4f}')
    print(f'      CV Mean   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

    # ── MLflow Logging ─────────────────────────
    print('[4/4] Manual logging ke MLflow...')
    mlflow.set_experiment('Titanic_WorkflowCI_Devin-Novansyah')

    with mlflow.start_run(run_name='MLProject_CI_Run') as run:
        print(f'      Run ID: {run.info.run_id}')

        mlflow.set_tags({
            'author' : 'Devin Novansyah',
            'model'  : 'RandomForestClassifier',
            'dataset': 'Titanic',
            'trigger': 'GitHub Actions CI'
        })

        # Log params
        mlflow.log_params({
            'n_estimators'     : args.n_estimators,
            'max_depth'        : args.max_depth,
            'min_samples_split': args.min_samples_split,
            'max_features'     : args.max_features,
            'test_size'        : args.test_size,
            'random_state'     : args.random_state,
        })

        # Log metrics
        mlflow.log_metrics({
            'accuracy'        : acc,
            'precision'       : prec,
            'recall'          : rec,
            'f1_score'        : f1,
            'roc_auc'         : roc,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy' : cv_scores.std(),
        })

        # Log model
        mlflow.sklearn.log_model(
            sk_model      = model,
            artifact_path = 'model',
            input_example = X_train.head(5)
        )

        # Log artefak
        with tempfile.TemporaryDirectory() as tmp:
            cm_path = plot_confusion_matrix(y_test, y_pred, tmp)
            mlflow.log_artifact(cm_path, artifact_path='plots')

            fi_path = plot_feature_importance(model, X.columns.tolist(), tmp)
            mlflow.log_artifact(fi_path, artifact_path='plots')

            cr_path = save_classification_report(y_test, y_pred, tmp)
            mlflow.log_artifact(cr_path, artifact_path='reports')

            p_path  = save_params_json(vars(args), tmp)
            mlflow.log_artifact(p_path, artifact_path='reports')

        # Simpan run_id untuk digunakan workflow
        run_id = run.info.run_id
        with open('run_id.txt', 'w') as f:
            f.write(run_id)
        print(f'\n✅ Run ID disimpan ke run_id.txt: {run_id}')

    print('\n✅ Modelling selesai!')
    print('=' * 55)


if __name__ == '__main__':
    main()
