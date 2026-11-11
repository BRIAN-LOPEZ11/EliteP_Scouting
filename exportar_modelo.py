import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Solo guarda las gráficas, no abre ventanas
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, classification_report
)
from src.database.connection import MySQLConnector

# Carpeta de salida para gráficas y tablas
DIR_PROCESADOS = os.path.join('data', 'processed')


def main():
    print("=================================================================")
    print("  FASE 1: ENTRENAMIENTO Y PERSISTENCIA DEL MODELO (DEPLOYMENT)")
    print("=================================================================\n")

    os.makedirs(DIR_PROCESADOS, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # -------------------------------------------------------------
    # INGESTA DESDE MYSQL
    # -------------------------------------------------------------
    db = MySQLConnector()
    query = """
        SELECT
            p.player_id,
            p.name AS jugador,
            p.position AS posicion,
            p.height_in_cm AS estatura_cm,
            TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS edad,
            p.market_value_in_eur AS valor_mercado
        FROM players p
        WHERE p.market_value_in_eur IS NOT NULL
          AND p.date_of_birth IS NOT NULL;
    """
    print("[1] Extrayendo datos desde MySQL...")
    df = db.extract_data(query)

    # -------------------------------------------------------------
    # PREPROCESAMIENTO E IMPUTACIÓN
    # -------------------------------------------------------------
    df['estatura_cm'] = pd.to_numeric(df['estatura_cm'], errors='coerce')
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['valor_mercado'] = pd.to_numeric(df['valor_mercado'], errors='coerce')

    df['posicion'] = df['posicion'].fillna('Sin Posicion')

    # Guardar las medianas de imputación para reutilizar en producción
    medianas_estatura = df.groupby('posicion')['estatura_cm'].median().to_dict()
    mediana_global_estatura = df['estatura_cm'].median()
    mediana_global_edad = df['edad'].median()

    df['estatura_cm'] = df.groupby('posicion')['estatura_cm'].transform(lambda x: x.fillna(x.median()))
    df['estatura_cm'] = df['estatura_cm'].fillna(mediana_global_estatura)
    df['edad'] = df['edad'].fillna(mediana_global_edad)

    # Target binario: Élite = Top 25% (Percentil 75) del valor de mercado
    umbral_elite = df['valor_mercado'].quantile(0.75)
    df['es_elite'] = (df['valor_mercado'] >= umbral_elite).astype(int)

    # One-Hot Encoding y matriz de variables
    df_dummies = pd.get_dummies(df[['posicion']], drop_first=True, dtype=int)
    X = pd.concat([df[['edad', 'estatura_cm']], df_dummies], axis=1)
    y = df['es_elite']

    # División 80/20 estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[OK] Registros: {X.shape[0]} | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")

    # -------------------------------------------------------------
    # OPTIMIZACIÓN CON GRIDSEARCHCV
    # -------------------------------------------------------------
    print("[2] Optimizando Random Forest con GridSearchCV (5-Fold CV)...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'min_samples_leaf': [5, 10, 20]
    }

    grid = GridSearchCV(
        estimator=RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"    Mejores hiperparámetros: {grid.best_params_}")
    print(f"    Mejor AUC en validación cruzada: {grid.best_score_:.4f}\n")

    mejor_modelo = grid.best_estimator_
    y_pred = mejor_modelo.predict(X_test)
    y_prob_rf = mejor_modelo.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    print(f"[3] AUC en Test: {auc_rf:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Estándar', 'Élite']))

    # -------------------------------------------------------------
    # DIAGNÓSTICO DE OVERFITTING: ÁRBOL SIN RESTRICCIÓN vs CON PODA
    # -------------------------------------------------------------
    print("[4] Diagnóstico de Overfitting (Train vs Test)...")
    arbol_libre = DecisionTreeClassifier(random_state=42)
    arbol_libre.fit(X_train, y_train)

    arbol_podado = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
    arbol_podado.fit(X_train, y_train)

    filas = []
    for nombre, modelo in [('Árbol sin poda', arbol_libre), ('Árbol con poda (max_depth=4)', arbol_podado)]:
        acc_train = accuracy_score(y_train, modelo.predict(X_train))
        acc_test = accuracy_score(y_test, modelo.predict(X_test))
        filas.append({
            'Modelo': nombre,
            'Accuracy Train': round(acc_train, 4),
            'Accuracy Test': round(acc_test, 4),
            'Brecha (Train - Test)': round(acc_train - acc_test, 4)
        })
    tabla_overfitting = pd.DataFrame(filas)
    print(tabla_overfitting.to_string(index=False))
    
    # -------------------------------------------------------------
    # DIAGNÓSTICO DE MODELOS TRIVIALES
    # -------------------------------------------------------------
    print("\n[4.1] Diagnóstico de predicciones por modelo...")

    for nombre, modelo in [
        ("Árbol sin poda", arbol_libre),
        ("Árbol podado", arbol_podado),
        ("Random Forest", mejor_modelo)
    ]:
        predicciones = modelo.predict(X_test)

        print(f"\nModelo: {nombre}")
        print("Distribución de predicciones:")
        print(
            pd.Series(predicciones)
            .value_counts()
            .reindex([0, 1], fill_value=0)
            .rename(index={0: "Estándar", 1: "Élite"})
        )

        print("Matriz de confusión:")
        print(confusion_matrix(y_test, predicciones))




    # Gráfico 1: Accuracy Train vs Test
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x_pos = np.arange(len(tabla_overfitting))
    ancho = 0.35
    b1 = ax.bar(x_pos - ancho / 2, tabla_overfitting['Accuracy Train'], ancho,
                label='Train', color='#2980b9', edgecolor='black')
    b2 = ax.bar(x_pos + ancho / 2, tabla_overfitting['Accuracy Test'], ancho,
                label='Test', color='#e67e22', edgecolor='black')
    ax.bar_label(b1, fmt='%.3f', padding=3)
    ax.bar_label(b2, fmt='%.3f', padding=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tabla_overfitting['Modelo'])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Diagnóstico de Overfitting: Train vs Test', fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PROCESADOS, 'overfitting_train_vs_test.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # CURVAS ROC COMPARATIVAS: ÁRBOL ÚNICO vs RANDOM FOREST
    # -------------------------------------------------------------
    y_prob_tree = arbol_podado.predict_proba(X_test)[:, 1]
    auc_tree = roc_auc_score(y_test, y_prob_tree)
    print(f"\n[5] ROC-AUC -> Árbol único: {auc_tree:.4f} | Random Forest: {auc_rf:.4f}")

    fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_tree, tpr_tree, color='#e67e22', linestyle='--', linewidth=2,
            label=f'Árbol Único (AUC = {auc_tree:.3f})')
    ax.plot(fpr_rf, tpr_rf, color='#27ae60', linewidth=2.5,
            label=f'Random Forest (AUC = {auc_rf:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Clasificador Aleatorio (AUC = 0.500)')
    ax.set_title('Comparativa de Curvas ROC', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR - Recall)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PROCESADOS, 'curvas_roc_comparativas.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # IMPORTANCIA DE VARIABLES (GINI)
    # -------------------------------------------------------------
    importancias = pd.Series(mejor_modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n[6] Top variables más relevantes (Gini):")
    print(importancias.head(8))

    top = importancias.head(10).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color='#2980b9', edgecolor='black', alpha=0.85)
    ax.set_title('Importancia de Variables (Random Forest)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importancia Relativa (Gini)')
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PROCESADOS, 'importancia_variables.png'), dpi=300)
    plt.close()

    print(f"\n[OK] Gráficas y tablas guardadas en '{DIR_PROCESADOS}/'")
    
    

    # -------------------------------------------------------------
    # ENTRENAMIENTO FINAL (mejores hiperparámetros, todos los datos)
    # -------------------------------------------------------------
    rf_model = RandomForestClassifier(
        **grid.best_params_, class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_model.fit(X, y)

    # -------------------------------------------------------------
    # PAQUETE DE DESPLIEGUE
    # -------------------------------------------------------------
    paquete_despliegue = {
        'modelo': rf_model,
        'columnas_X': X.columns.tolist(),
        'medianas_estatura': medianas_estatura,
        'mediana_global_estatura': mediana_global_estatura,
        'mediana_global_edad': mediana_global_edad,
        'umbral_elite_eur': umbral_elite
    }

    ruta_modelo = 'models/modelo_scouting_rf.joblib'
    joblib.dump(paquete_despliegue, ruta_modelo)
    print(f"Paquete de modelo exportado exitosamente en: '{ruta_modelo}'")


if __name__ == "__main__":
    main()