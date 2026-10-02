"""
OcupaPerú — Núcleo (sin interfaz)

Aquí vive todo lo que NO es ventana: carga de datos y modelos, predicción,
pronóstico y la base de datos SQLite de consultas. La ventana (ventana.py)
solo llama a estas funciones.

Los modelos NUNCA se entrenan aquí: solo se cargan los ya guardados en models/.
"""
import sqlite3
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

BASE = Path(__file__).parent
DB = str(BASE / "consultas.db")

MESES = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
         7: "Julio", 8: "Agosto", 9: "Setiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}

# ──────────────────────────────────────────────────────────────
# CARGA DE DATOS Y MODELOS
# ──────────────────────────────────────────────────────────────
df = pd.read_csv(BASE / "ocupaperu_analitico.csv", parse_dates=["FECHA"])
serie = pd.read_csv(BASE / "ocupaperu_serie.csv", parse_dates=["FECHA"])
res_clf = pd.read_csv(BASE / "models" / "resultados_clasificacion.csv")
res_reg = pd.read_csv(BASE / "models" / "resultados_regresion.csv")
shap_imp = pd.read_csv(BASE / "models" / "shap_importancia.csv")
cm = np.load(BASE / "models" / "matriz_confusion.npy")
pron = pd.read_csv(BASE / "models" / "pronostico_6meses.csv", index_col=0, parse_dates=True)
res_arr = pd.read_csv(BASE / "models" / "resultados_serie_ARRIBOS.csv")
res_tnoh = pd.read_csv(BASE / "models" / "resultados_serie_TNOH_NACIONAL.csv")
pred_reg = pd.read_csv(BASE / "models" / "predicciones_regresion.csv", parse_dates=["FECHA"])

modelo_clf = joblib.load(BASE / "models" / "modelo_clasificacion.pkl")
modelo_reg = joblib.load(BASE / "models" / "modelo_regresion.pkl")
FEATURES = joblib.load(BASE / "models" / "features.pkl")
UMBRAL = joblib.load(BASE / "models" / "umbral.pkl")

DEPTOS = sorted(df.DEPARTAMENTO.unique())
CLASES = sorted(df.CLASE.unique())
CATEGS = sorted(df.CATEGORIA.unique())


# ──────────────────────────────────────────────────────────────
# BASE DE DATOS (SQLite) — consultas guardadas
# ──────────────────────────────────────────────────────────────
def _ejecutar(sql, params=()):
    con = sqlite3.connect(DB)
    cur = con.execute(sql, params)          # consultas parametrizadas (?)
    filas = cur.fetchall()
    con.commit()
    con.close()
    return filas


def init_db():
    _ejecutar("""CREATE TABLE IF NOT EXISTS consultas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        departamento TEXT, clase TEXT, categoria TEXT,
        mes INTEGER, anio INTEGER,
        prediccion TEXT, probabilidad REAL, tnoh_estimado REAL,
        timestamp TEXT)""")


def _ahora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def crear_consulta(dep, clase, cat, mes, anio, r):
    _ejecutar("""INSERT INTO consultas
        (departamento, clase, categoria, mes, anio, prediccion, probabilidad, tnoh_estimado, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?)""",
              (dep, clase, cat, mes, anio, r["prediccion"],
               round(r["probabilidad"], 4), round(r["tnoh_estimado"], 2), _ahora()))


def actualizar_consulta(id_, dep, clase, cat, mes, anio, r):
    _ejecutar("""UPDATE consultas SET departamento=?, clase=?, categoria=?, mes=?, anio=?,
                 prediccion=?, probabilidad=?, tnoh_estimado=?, timestamp=? WHERE id=?""",
              (dep, clase, cat, mes, anio, r["prediccion"],
               round(r["probabilidad"], 4), round(r["tnoh_estimado"], 2), _ahora(), id_))


def eliminar_consulta(id_):
    _ejecutar("DELETE FROM consultas WHERE id=?", (id_,))


def listar_consultas():
    return _ejecutar("""SELECT id, departamento, clase, categoria, mes, anio,
                        prediccion, probabilidad, tnoh_estimado, timestamp
                        FROM consultas ORDER BY id DESC""")


init_db()


# ──────────────────────────────────────────────────────────────
# PREDICCIÓN (clasificación + regresión)
# ──────────────────────────────────────────────────────────────
def construir_features(departamento, clase, categoria, mes, anio):
    """Arma las columnas EN EL MISMO ORDEN que vio el modelo al entrenar.

    Los rezagos salen del historial real de ese perfil: son datos PASADOS,
    conocidos antes del mes que se predice. Eso es lo que los hace legítimos.
    """
    h = df[(df.DEPARTAMENTO == departamento) & (df.CLASE == clase) &
           (df.CATEGORIA == categoria)].sort_values("FECHA")
    if h.empty:
        return None, "No hay historial para esa combinación (ese tipo de hospedaje no existe en ese departamento)."

    ult = h.iloc[-1]
    mismo_mes = h[h.MES == mes]

    f = {
        "MES": mes,
        "MES_SIN": np.sin(2 * np.pi * mes / 12),
        "MES_COS": np.cos(2 * np.pi * mes / 12),
        "TRIMESTRE": (mes - 1) // 3 + 1,
        "ES_FIESTAS_PATRIAS": int(mes == 7),
        "ES_SEMANA_SANTA": int(mes in (3, 4)),
        "ES_FIN_DE_ANIO": int(mes in (12, 1)),
        "ES_TEMPORADA_SECA": int(mes in (5, 6, 7, 8, 9)),
        "ES_COVID": 0,
        "ES_POST_COVID": 1,
        "T": (anio - 2015) * 12 + mes,
        "ANIO": anio,
        # rezagos: del historial del perfil
        "TNOH_LAG1": ult.PORCENTAJE_TNOH,
        "TNOH_LAG3": h.PORCENTAJE_TNOH.iloc[-3] if len(h) >= 3 else ult.PORCENTAJE_TNOH,
        "TNOH_LAG12": mismo_mes.PORCENTAJE_TNOH.iloc[-1] if len(mismo_mes) else h.PORCENTAJE_TNOH.mean(),
        "TNOH_MM3": h.PORCENTAJE_TNOH.tail(3).mean(),
        "TNOH_MM12": h.PORCENTAJE_TNOH.tail(12).mean(),
        # estructura: lo último conocido del perfil
        "NUMERO_ESTABLECIMIENTOS": ult.NUMERO_ESTABLECIMIENTOS,
        "NUMERO_HABITACIONES": ult.NUMERO_HABITACIONES,
        "HAB_POR_ESTAB": ult.HAB_POR_ESTAB,
        "PLAZAS_POR_HAB": ult.PLAZAS_POR_HAB,
        "LOG_ESTAB": ult.LOG_ESTAB,
        "LOG_HAB": ult.LOG_HAB,
        "N_ESTRELLAS": ult.N_ESTRELLAS,
        # perfil histórico del departamento
        "DEP_PCT_EXT_HIST": ult.DEP_PCT_EXT_HIST,
        "DEP_TNOH_HIST": ult.DEP_TNOH_HIST,
        "DEP_ESTACIONALIDAD": ult.DEP_ESTACIONALIDAD,
    }
    f["DELTA_YOY"] = f["TNOH_LAG1"] - f["TNOH_LAG12"]

    x = pd.DataFrame([f])
    for col in FEATURES:                       # one-hot en el orden exacto del entrenamiento
        if col not in x.columns:
            x[col] = 0
    for pref, val in [("DEPARTAMENTO_", departamento), ("CLASE_", clase), ("CATEGORIA_", categoria)]:
        if pref + val in FEATURES:
            x[pref + val] = 1
    return x[FEATURES], None


def predecir(departamento, clase, categoria, mes, anio):
    """Devuelve (resultado, error). resultado = {prediccion, probabilidad, tnoh_estimado}."""
    x, err = construir_features(departamento, clase, categoria, mes, anio)
    if err:
        return None, err
    proba = float(modelo_clf.predict_proba(x)[0, 1])
    return {
        "prediccion": "ALTA OCUPACIÓN" if proba >= UMBRAL else "Ocupación normal",
        "probabilidad": proba,
        "tnoh_estimado": float(modelo_reg.predict(x)[0]),
    }, None


# ──────────────────────────────────────────────────────────────
# PRONÓSTICO
# ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def pronosticar(var, metodo, hasta):
    """Pronostica desde el último mes con datos hasta `hasta` (AAAA-MM-01), con el mismo método
    ganador de entrenar.py y los mismos datos (desde 2022, para dejar fuera el COVID)."""
    y = serie.set_index("FECHA")[var].asfreq("MS")
    y = y[y.index >= "2022-01-01"]
    h = (pd.Timestamp(hasta).year - y.index[-1].year) * 12 + pd.Timestamp(hasta).month - y.index[-1].month
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "SARIMA" in metodo:
            f = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False).forecast(h)
        elif "Holt" in metodo:
            f = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12).fit().forecast(h)
        else:
            f = pd.Series([y[-3:].mean()] * h,
                          index=pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS"))
    return f


def pronostico_6m(var):
    """Pronóstico de `var` (ARRIBOS o TNOH_NACIONAL) para los próximos 6 meses contados desde HOY.

    Los datos llegan hasta jun-2025: los meses entre esa fecha y hoy también se pronostican
    para poder llegar hasta el presente.
    """
    res = res_arr if var == "ARRIBOS" else res_tnoh
    inicio = pd.Timestamp.today().normalize().replace(day=1)
    fin = inicio + pd.DateOffset(months=5)
    if fin <= serie.FECHA.iloc[-1]:                       # por si los datos fueran más recientes
        inicio, fin = pron.index[0], pron.index[-1]
        fc = pron[var]
    else:
        fc = pronosticar(var, res.iloc[0].Modelo, fin.strftime("%Y-%m-01"))
    return {"fc": fc, "ult6": fc.loc[inicio:fin],
            "metodo": res.iloc[0].Modelo, "mape": float(res.iloc[0]["MAPE_%"])}
