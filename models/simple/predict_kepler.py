import joblib
import numpy as np
import pandas as pd
import os
import json

# --- Configuración ---
MODEL_PATH = os.path.join("models", "best_model.joblib")
ENCODER_PATH = os.path.join("models", "label_encoder.joblib")

# --- Cargar modelos ---
print("Cargando modelo y codificador...")
pipeline = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
print("Modelos cargados correctamente.\n")

# --- Orden esperado de columnas ---
# Este orden debe coincidir EXACTAMENTE con las columnas numéricas del dataset Kepler
# (excluyendo koi_disposition y las columnas koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec)

FEATURE_COLUMNS = [
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
    'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol',
    'koi_model_snr', 'koi_tce_plnt_num', 'koi_tce_delivname',
    'koi_steff', 'koi_slogg', 'koi_srad', 'ra', 'dec',
    'koi_kepmag', 'koi_smass', 'koi_smet', 'koi_tce_impact'
]
# Ajustar esta lista si el dataset tiene diferente orden o columnas adicionales.

# --- Función principal de inferencia ---
def predict_kepler(input_values):
    """
    input_values: lista o array con los valores de entrada en el mismo orden que FEATURE_COLUMNS
    Devuelve: dict con clase predicha y probabilidades por clase
    """
    if len(input_values) != len(FEATURE_COLUMNS):
        raise ValueError(f"Se esperaban {len(FEATURE_COLUMNS)} valores, pero se recibieron {len(input_values)}.")

    # Crear DataFrame con una sola fila
    input_df = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)

    # Predicción
    predicted_class_encoded = pipeline.predict(input_df)[0]
    predicted_class_label = label_encoder.inverse_transform([predicted_class_encoded])[0]

    # Probabilidades
    if hasattr(pipeline.named_steps['estimator'], "predict_proba"):
        probs = pipeline.predict_proba(input_df)[0]
        class_probs = {
            label_encoder.inverse_transform([i])[0]: float(probs[i]) for i in range(len(probs))
        }
    else:
        class_probs = {}

    return {
        "predicted_class": predicted_class_label,
        "class_probabilities": class_probs
    }

# --- (Ejemplo) Para de uso local ---
if __name__ == "__main__":
    # Testeando un input en el mismo orden que FEATURE_COLUMNS
    example_input = [
        12.3, 134.5, 0.02, 3.4, 1500, 1.1, 500, 200,
        10, 1, 1, 5700, 4.5, 1.0, 291.2, 45.3, 14.2, 1.02, 0.1, 0.02
    ]

    print("Valores de entrada:", example_input)
    result = predict_kepler(example_input)
    print("\nResultado de predicción:")
    print(json.dumps(result, indent=4))
