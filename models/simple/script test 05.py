import os
import glob
import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.inspection import permutation_importance

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
N_JOBS = 1  # Ajusta según tu hardware

# --- 0. Detectar/Combinar CSVs en la carpeta actual ---
def load_and_combine_csvs(path='.', verbose=True):
    csv_paths = sorted(glob.glob(os.path.join(path, '*.csv')))
    if verbose:
        print(f"Archivos .csv encontrados ({len(csv_paths)}):")
        for p in csv_paths:
            print("  -", os.path.basename(p))
    if len(csv_paths) == 0:
        raise FileNotFoundError("No se encontraron archivos .csv en el directorio actual.")

    dfs, cols_list, rows_count = [], [], []
    for p in csv_paths:
        try:
            full_df = pd.read_csv(p, comment='#')
        except Exception as e:
            print(f"⚠️ Error leyendo {p}: {e}. Se ignora este archivo.")
            continue
        dfs.append((p, full_df))
        cols_list.append(set(full_df.columns))
        rows_count.append((p, full_df.shape[0]))

    if len(dfs) == 0:
        raise FileNotFoundError("No se pudieron leer los archivos .csv disponibles.")

    all_same = all(cols_list[0] == cols for cols in cols_list)
    if all_same and len(dfs) > 1:
        if verbose:
            print("Todos los CSVs comparten las mismas columnas: concatenando verticalmente.")
        concatenated = pd.concat([df for _, df in dfs], axis=0, ignore_index=True)
        return concatenated, [p for p, _ in dfs], "concatenated"

    intersec = set.intersection(*cols_list)
    min_cols = min(len(cols) for cols in cols_list)
    if len(intersec) >= 0.5 * min_cols and len(intersec) > 0:
        if verbose:
            print(f"Intersección significativa ({len(intersec)} cols). Intentando merge outer.")
        base_p, base_df = dfs[0]
        merged = base_df.copy()
        for p, df_full in dfs[1:]:
            shared = list(merged.columns.intersection(df_full.columns))
            if len(shared) == 0:
                continue
            merged = merged.merge(df_full, how='outer', on=shared, suffixes=('', '_r'))
        return merged, [p for p, _ in dfs], "merged_on_intersection"

    best_path = sorted(rows_count, key=lambda x: x[1], reverse=True)[0][0]
    chosen_df = next(df_full for p, df_full in dfs if p == best_path)
    if verbose:
        print(f"No fue posible combinar. Se usará: {os.path.basename(best_path)}.")
    return chosen_df, [best_path], "single"


# --- Cargar datasets ---
df, used_files, combine_mode = load_and_combine_csvs('.', verbose=True)
print(f"\nModo de carga: {combine_mode}. Archivos usados: {', '.join([os.path.basename(p) for p in used_files])}")
print(f"Dimensiones del dataset: {df.shape}")

# --- Variables importantes ---
TARGET_COLUMN = 'koi_disposition'
FP_COLUMNS = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']

# --- Limpieza y preparación ---
def base_preprocess(df, target_col=TARGET_COLUMN, fp_cols=FP_COLUMNS):
    cols_to_drop = [col for col in df.columns if 'err' in col or 'lim' in col]
    cols_to_drop += [col for col in df.columns if any(c in col for c in ['kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition', 'koi_score'])]
    cols_to_drop += [col for col in fp_cols if col in df.columns]  # 🔥 eliminar columnas de falso positivo
    
    df_cleaned = df.drop(columns=cols_to_drop, errors='ignore')

    if target_col not in df_cleaned.columns:
        raise KeyError(f"La columna objetivo '{target_col}' no está en el DataFrame.")
    df_cleaned = df_cleaned[df_cleaned[target_col].isin(['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])].copy()

    text_cols = df_cleaned.select_dtypes(include=['object']).columns.drop(target_col, errors='ignore')
    df_cleaned = df_cleaned.drop(columns=text_cols, errors='ignore')

    le = LabelEncoder()
    df_cleaned['y_encoded'] = le.fit_transform(df_cleaned[target_col])

    X = df_cleaned.drop(columns=[target_col, 'y_encoded'])
    y = df_cleaned['y_encoded']
    return X, y, le


X_base, y_base, label_encoder = base_preprocess(df)
print(f"✅ Variables eliminadas (falsos positivos): {FP_COLUMNS}")
print(f"Clases objetivo: {label_encoder.classes_}")
print(f"Tamaño del dataset procesado: {X_base.shape}")

# --- Train/test split ---
X_train_base, X_test, y_train_base, y_test = train_test_split(
    X_base, y_base, test_size=0.2, random_state=RANDOM_STATE, stratify=y_base
)

# --- Modelos ---
models = {
    'XGBoost': {
        'model': xgb.XGBClassifier(eval_metric='mlogloss', random_state=RANDOM_STATE),
        'params': {
            'estimator__n_estimators': [50, 100],
            'estimator__max_depth': [3, 5],
            'estimator__learning_rate': [0.1, 0.05]
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=RANDOM_STATE),
        'params': {
            'estimator__n_estimators': [50, 100],
            'estimator__learning_rate': [0.5, 1.0]
        }
    },
    'LightGBM': {
        'model': lgb.LGBMClassifier(random_state=RANDOM_STATE),
        'params': {
            'estimator__n_estimators': [50, 100],
            'estimator__num_leaves': [31, 50],
            'estimator__learning_rate': [0.1, 0.05]
        }
    }
}

imputation_strategies = [
    ('Media', SimpleImputer(strategy='mean')),
    ('Mediana (Robusta)', SimpleImputer(strategy='median'))
]

scaling_strategies = [
    ('StandardScaler (No Manejo)', StandardScaler()),
    ('RobustScaler (Robusto a Outliers)', RobustScaler())
]

sampling_strategies = [
    ('RandomUnderSampler', RandomUnderSampler(random_state=RANDOM_STATE)),
    ('RandomOverSampler', RandomOverSampler(random_state=RANDOM_STATE)),
    ('SMOTE', SMOTE(random_state=RANDOM_STATE))
]

results = []
iteration_count = 0
print("-" * 120)

# --- Experimentos ---
for imp_name, imputer in imputation_strategies:
    for scale_name, scaler in scaling_strategies:
        preprocessor_pipe = ImbPipeline([
            ('imputer', imputer),
            ('scaler', scaler)
        ])
        X_train_prep = preprocessor_pipe.fit_transform(X_train_base)
        X_test_prep = preprocessor_pipe.transform(X_test)
        X_train_prep = pd.DataFrame(X_train_prep, columns=X_train_base.columns)
        X_test_prep = pd.DataFrame(X_test_prep, columns=X_test.columns)

        for sampler_name, sampler in sampling_strategies:
            for model_name, config in models.items():
                iteration_count += 1
                print(f"\n[{iteration_count}] Entrenando: {model_name} | Imputación={imp_name} | Escalado={scale_name} | Balanceo={sampler_name}")

                pipeline = ImbPipeline([
                    ('sampler', sampler),
                    ('estimator', config['model'])
                ])

                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=config['params'],
                    scoring='f1_macro',
                    cv=3,
                    n_jobs=N_JOBS,
                    verbose=0
                )

                grid_search.fit(X_train_prep, y_train_base)
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test_prep)

                metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision (Macro)': precision_score(y_test, y_pred, average='macro', zero_division=0),
                    'Recall (Macro)': recall_score(y_test, y_pred, average='macro', zero_division=0),
                    'F1-Score (Macro)': f1_score(y_test, y_pred, average='macro', zero_division=0)
                }

                try:
                    y_pred_proba = best_model.predict_proba(X_test_prep)
                    auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro')
                except Exception:
                    auc_roc = np.nan

                results.append({
                    'ID': iteration_count,
                    'Modelo': model_name,
                    'Imputacion': imp_name,
                    'Escalado_Outliers': scale_name,
                    'Balanceo': sampler_name,
                    'Mejores_Hiperparametros': grid_search.best_params_,
                    **metrics,
                    'AUC-ROC (OVO)': auc_roc
                })
                print(f"  → F1-Macro={metrics['F1-Score (Macro)']:.4f}, AUC-ROC={auc_roc if not np.isnan(auc_roc) else 'NA'}")

# --- Resultados ---
results_df = pd.DataFrame(results).sort_values(by=['F1-Score (Macro)', 'AUC-ROC (OVO)'], ascending=False)
print("\nTop 10 combinaciones:\n", results_df.head(10).round(4).to_markdown(index=False))

best_row = results_df.iloc[0]
print("\n--- Mejor Modelo General (Basado en F1-Score Macro) ---")
print(best_row.to_string())

# --- Reentrenar el mejor modelo ---
best_config = models[best_row['Modelo']]
best_params = best_row['Mejores_Hiperparametros']

imputer_obj = next(im for name, im in imputation_strategies if name == best_row['Imputacion'])
scaler_obj = next(sc for name, sc in scaling_strategies if name == best_row['Escalado_Outliers'])
sampler_obj = next(s for sname, s in sampling_strategies if sname == best_row['Balanceo'])

preprocessor_pipe = ImbPipeline([
    ('imputer', imputer_obj),
    ('scaler', scaler_obj)
])
X_train_prep = preprocessor_pipe.fit_transform(X_train_base)
X_train_prep = pd.DataFrame(X_train_prep, columns=X_train_base.columns)

pipeline_best = ImbPipeline([
    ('sampler', sampler_obj),
    ('estimator', best_config['model'])
])
estimator_params = {k.replace('estimator__', ''): v for k, v in best_params.items()}
pipeline_best.named_steps['estimator'].set_params(**estimator_params)
pipeline_best.fit(X_train_prep, y_train_base)

# --- Guardar modelo y LabelEncoder ---
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "best_model.joblib")
label_path = os.path.join("models", "label_encoder.joblib")

joblib.dump(pipeline_best, model_path)
joblib.dump(label_encoder, label_path)
print(f"\n✅ Modelo guardado en: {model_path}")
print(f"✅ LabelEncoder guardado en: {label_path}")

# --- Importancia de características ---
final_estimator = pipeline_best.named_steps['estimator']
if hasattr(final_estimator, "feature_importances_"):
    fi_df = pd.DataFrame({
        'feature': X_train_base.columns,
        'importance': final_estimator.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 20 features más importantes:\n", fi_df.head(20).to_markdown(index=False))
else:
    print("El modelo no expone 'feature_importances_'; omitiendo análisis de importancia.")

print("\n🎯 Script finalizado correctamente. Archivos listos en carpeta 'models/'.")
