from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

# -------------------------------------------------------------
# CARGA DEL PAQUETE DE DESPLIEGUE (generado por exportar_modelo.py)
# -------------------------------------------------------------
RUTA_MODELO = 'models/modelo_scouting_rf.joblib'
paquete = joblib.load(RUTA_MODELO)

modelo = paquete['modelo']
columnas_X = paquete['columnas_X']
medianas_estatura = paquete['medianas_estatura']
mediana_global_estatura = paquete['mediana_global_estatura']
umbral_elite_eur = paquete['umbral_elite_eur']

# Las posiciones válidas son las que el modelo vio en el entrenamiento
POSICIONES_VALIDAS = sorted(medianas_estatura.keys())

app = FastAPI(title="API Scouting - Random Forest", version="1.0")


# -------------------------------------------------------------
# VALIDACIÓN DE ENTRADA (Pydantic)
# -------------------------------------------------------------
class JugadorEntrada(BaseModel):
    posicion: str
    edad: int = Field(ge=18, le=45, description="Edad en años (18 a 45)")
    estatura_cm: Optional[float] = Field(default=None, gt=150, lt=210,
                                         description="Estatura en cm (opcional, entre 150 y 210 exclusivo)")

    @field_validator('posicion')
    @classmethod
    def validar_posicion(cls, v: str) -> str:
        v = v.strip()
        if v not in POSICIONES_VALIDAS:
            raise ValueError(f"Posición inválida. Opciones: {POSICIONES_VALIDAS}")
        return v


# -------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------
@app.get("/posiciones")
def listar_posiciones():
    return {"posiciones": POSICIONES_VALIDAS}


@app.post("/predict")
def predict(datos: JugadorEntrada):
    # Imputación en producción: misma lógica del entrenamiento
    estatura = datos.estatura_cm
    if estatura is None:
        estatura = medianas_estatura.get(datos.posicion)
        if estatura is None or pd.isna(estatura):
            estatura = mediana_global_estatura

    # Construir la fila con el mismo orden de columnas del entrenamiento
    fila = {col: 0 for col in columnas_X}
    fila['edad'] = datos.edad
    fila['estatura_cm'] = float(estatura)
    col_dummy = f"posicion_{datos.posicion}"
    if col_dummy in fila:  # la categoría base (drop_first) queda en 0
        fila[col_dummy] = 1

    X = pd.DataFrame([fila], columns=columnas_X)
    prediccion = int(modelo.predict(X)[0])
    prob_elite = float(modelo.predict_proba(X)[0][1])

    return {
        "prediccion": prediccion,
        "etiqueta": "Élite" if prediccion == 1 else "Estándar",
        "probabilidad_elite": round(prob_elite * 100, 2),
        "estatura_usada_cm": round(float(estatura), 1),
        "umbral_elite_eur": float(umbral_elite_eur),
    }