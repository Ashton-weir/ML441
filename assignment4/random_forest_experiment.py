from pathlib import Path
from time import perf_counter
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'results'
CATEGORIES = ['proto', 'state', 'service']
LABELS = list(range(10))
SEED = 42
CUSTOM_WEIGHTS = {2: 1.5, 5: 1.5, 7: 2.0, 8: 1.5}


def make_pipeline(encoding, parameters, tree=False):
    parameters = parameters.copy()
    if isinstance(parameters.get('class_weight'), dict):
        parameters['class_weight'] = {int(k): v for k, v in parameters['class_weight'].items()}
    # Fit the imputer and encoder inside each training fold.
    encoder = (OneHotEncoder(handle_unknown='ignore') if encoding == 'onehot'
               else OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    categorical = Pipeline([
        ('fill', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('encode', encoder),
    ])
    numeric = Pipeline([('fill', SimpleImputer(strategy='median'))])
    preprocessing = ColumnTransformer([
        ('numeric', numeric, [c for c in FEATURES if c not in CATEGORIES]),
        ('categorical', categorical, CATEGORIES),
    ])
    model = (DecisionTreeClassifier(random_state=SEED, **parameters) if tree
             else RandomForestClassifier(random_state=SEED, n_jobs=-1,
                                         bootstrap=True, **parameters))
    return Pipeline([('preprocessing', preprocessing), ('model', model)])


def measures(y, prediction):
    result = {'accuracy': accuracy_score(y, prediction),
              'balanced_accuracy': balanced_accuracy_score(y, prediction)}
    for average in ['macro', 'weighted']:
        p, r, f, _ = precision_recall_fscore_support(
            y, prediction, labels=LABELS, average=average, zero_division=0)
        result.update({f'{average}_precision': p, f'{average}_recall': r,
                       f'{average}_f1': f})
    return result


def evaluate_candidate(name, encoding, parameters, X, y, folds):
    # Pick up from the last completed fold.
    fold_file = OUT / f'{name}_folds.csv'
    rows = pd.read_csv(fold_file).to_dict('records') if fold_file.exists() else []
    completed = {row['fold'] for row in rows}
    for fold, (fit, validation) in enumerate(folds, 1):
        if fold in completed:
            continue
        model = make_pipeline(encoding, parameters)
        start = perf_counter()
        model.fit(X.iloc[fit], y.iloc[fit])
        fit_time = perf_counter() - start
        prediction = model.predict(X.iloc[validation])
        training = model.predict(X.iloc[fit])
        rows.append({'candidate': name, 'encoding': encoding, 'fold': fold,
                     **measures(y.iloc[validation], prediction),
                     'training_macro_f1': measures(y.iloc[fit], training)['macro_f1'],
                     'fit_seconds': fit_time})
        
        temporary = fold_file.with_suffix('.tmp')
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(fold_file)
    frame = pd.DataFrame(rows)
    summary = {'candidate': name, 'encoding': encoding,
               'parameters': json.dumps(parameters)}
    for column in frame.select_dtypes(include='number').columns.drop('fold'):
        summary[f'mean_{column}'] = frame[column].mean()
        summary[f'std_{column}'] = frame[column].std(ddof=1)
    print(f"{name}: CV macro-F1 = {summary['mean_macro_f1']:.4f}")
    return summary


def best(rows):
    return max(rows, key=lambda row: (row['mean_macro_f1'], row['mean_balanced_accuracy']))


def save_evaluation(name, model, X_train, y_train, X_test, y_test, test_indices, names):
    start = perf_counter()
    model.fit(X_train, y_train)
    fit_time = perf_counter() - start
    rows = []
    for split, X, y in [('training', X_train, y_train), ('test', X_test, y_test)]:
        start = perf_counter()
        prediction = model.predict(X)
        rows.append({'dataset': split, **measures(y, prediction),
                     'prediction_seconds': perf_counter() - start,
                     'fit_seconds': fit_time})
    pd.DataFrame(rows).to_csv(OUT / f'{name}_train_test.csv', index=False)
    report = classification_report(y_test, prediction, labels=LABELS,
                                   target_names=names, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(OUT / f'{name}_classification_report.csv')
    pd.DataFrame({'row_index': test_indices, 'actual': y_test.to_numpy(),
                  'predicted': prediction}).to_csv(OUT / f'{name}_predictions.csv', index=False)
    counts = confusion_matrix(y_test, prediction, labels=LABELS)
    normalised = confusion_matrix(y_test, prediction, labels=LABELS, normalize='true')
    for title, matrix in [('counts', counts), ('normalised', normalised)]:
        pd.DataFrame(matrix, index=names, columns=names).to_csv(
            OUT / f'{name}_confusion_{title}.csv')
    fig, ax = plt.subplots(figsize=(9, 8))
    picture = ax.imshow(normalised, cmap='Blues', vmin=0, vmax=1)
    for i in LABELS:
        for j in LABELS:
            ax.text(j, i, f'{normalised[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if normalised[i, j] > .5 else 'black')
    ax.set_xticks(LABELS, names, rotation=90)
    ax.set_yticks(LABELS, names)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('Actual class')
    fig.colorbar(picture, ax=ax, label='Proportion of actual class')
    fig.tight_layout()
    fig.savefig(OUT / f'{name}_confusion.png', dpi=250)
    plt.close(fig)
    joblib.dump(model, OUT / f'{name}_model.joblib')
    return {'model': name, **measures(y_test, prediction)}


def main():
    global FEATURES
    OUT.mkdir(exist_ok=True)
    # Keep the original row order so the saved split still matches.
    frame = pd.read_csv(DATA / 'networkTraffic.csv', na_values='?')
    train = np.load(DATA / 'train_indices.npy')
    test = np.load(DATA / 'test_indices.npy')

    combined = np.concatenate([train, test])
    if not np.array_equal(np.sort(combined), np.arange(len(frame))):
        raise ValueError('The saved split does not match the dataset rows.')
    mapping = pd.read_csv(DATA / 'attack_category_map.csv')
    names = mapping.set_index('Mapping')['Attack Category Name'].loc[LABELS].tolist()
    X = frame.drop(columns=['id', 'attack_cat'])
    y = frame['attack_cat']
    FEATURES = X.columns.tolist()
    X_train, y_train = X.iloc[train], y.iloc[train]
    X_test, y_test = X.iloc[test], y.iloc[test]
    pd.DataFrame({'training': y_train.value_counts(), 'test': y_test.value_counts()}).sort_index().to_csv(
        OUT / 'class_counts.csv')
    X_train.isna().sum().to_csv(OUT / 'training_missing_values.csv')
    X_train[CATEGORIES].nunique().to_csv(OUT / 'training_category_counts.csv')
    # Use the same five folds for every candidate.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(X_train, y_train))
    base = dict(n_estimators=100, criterion='gini', max_depth=None,
                min_samples_leaf=1, max_features='sqrt', class_weight=None)
    rows = []
    # Compare compact integer codes with one-hot encoding.
    for encoding in ['ordinal', 'onehot']:
        rows.append(evaluate_candidate(f'encoding_{encoding}', encoding, base,
                                       X_train, y_train, folds))
    encoding = best(rows)['encoding']
    # Test tree complexity, feature sampling and class weighting together.
    grid = {'max_depth': [20, 30, None], 'min_samples_leaf': [1, 3, 5],
            'max_features': ['sqrt', 0.5, 1.0],
            'class_weight': [None, 'balanced_subsample', CUSTOM_WEIGHTS]}
    tuning = [best(rows)]
    for i, parameters in enumerate(ParameterSampler(grid, n_iter=12, random_state=SEED), 1):
        row = evaluate_candidate(f'tuning_{i:02d}', encoding, {**base, **parameters},
                                 X_train, y_train, folds)
        rows.append(row)
        tuning.append(row)
        pd.DataFrame(rows).to_csv(OUT / 'cv_summary.csv', index=False)
    selected = json.loads(best(tuning)['parameters'])
    # Check whether a larger forest improves validation performance.
    size_rows = [best(tuning)]
    for count in [200, 400]:
        row = evaluate_candidate(f'trees_{count}', encoding,
                                 {**selected, 'n_estimators': count}, X_train, y_train, folds)
        rows.append(row)
        size_rows.append(row)
    winner = best(size_rows)
    selected = json.loads(winner['parameters'])

    if isinstance(selected['class_weight'], dict):
        selected['class_weight'] = {int(k): v for k, v in selected['class_weight'].items()}
    pd.DataFrame(rows).to_csv(OUT / 'cv_summary.csv', index=False)
    (OUT / 'selected_parameters.json').write_text(
        json.dumps({'encoding': encoding, 'parameters': selected}, indent=2))
    # Evaluate
    forest = make_pipeline(encoding, selected)
    results = [save_evaluation('random_forest', forest, X_train, y_train,
                               X_test, y_test, test, names)]
    # Recreate the single tree
    tree_parameters = dict(criterion='entropy', max_depth=30, min_samples_split=10,
                           min_samples_leaf=1, class_weight=CUSTOM_WEIGHTS,
                           ccp_alpha=3.2245e-5)
    tree = make_pipeline('onehot', tree_parameters, tree=True)
    results.append(save_evaluation('assignment2_tree_reproduced', tree, X_train, y_train,
                                   X_test, y_test, test, names))
    pd.DataFrame(results).to_csv(OUT / 'model_comparison.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    summary = pd.DataFrame(rows)
    ax.errorbar(np.arange(len(rows)), summary.mean_macro_f1,
                yerr=summary.std_macro_f1, fmt='o', capsize=3)
    ax.set_xticks(np.arange(len(rows)), summary.candidate, rotation=70, ha='right')
    ax.set_ylabel('Validation macro-F1 (mean and standard deviation)')
    fig.tight_layout()
    fig.savefig(OUT / 'tuning_macro_f1.png', dpi=250)
    plt.close(fig)
    print(pd.DataFrame(results)[['model', 'accuracy', 'balanced_accuracy', 'macro_f1']].to_string(index=False))


if __name__ == '__main__':
    main()
