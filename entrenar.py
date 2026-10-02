"""
OcupaPerú — Entrenamiento de modelos
Genera la carpeta models/ EN TU MÁQUINA, con TUS versiones de librerías.
Así nunca hay conflicto de versiones al cargar los .pkl.

Uso:  python entrenar.py     (tarda ~2 minutos)
"""
import os, time, warnings
import joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score, precision_score,
    recall_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error)
from xgboost import XGBClassifier, XGBRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import shap

os.makedirs("models", exist_ok=True)
print("=" * 62)
print("OcupaPerú — entrenando modelos con TUS versiones de librerías")
import sklearn, xgboost
print(f"sklearn {sklearn.__version__} | xgboost {xgboost.__version__}")
print("=" * 62)

# ─── DATOS ───
df = pd.read_csv("ocupaperu_analitico.csv", parse_dates=["FECHA"])
CAT = ["DEPARTAMENTO", "CLASE", "CATEGORIA"]
X = pd.get_dummies(df.drop(columns=["Y_ALTA_OCUPACION", "PORCENTAJE_TNOH", "FECHA", "NO_REPORTO"]),
                   columns=CAT)
yc, yr = df.Y_ALTA_OCUPACION, df.PORCENTAJE_TNOH
tr, te = df.ANIO <= 2023, df.ANIO >= 2024          # split TEMPORAL, no aleatorio
Xtr, Xte = X[tr], X[te]
print(f"\nTRAIN {tr.sum():,} | TEST {te.sum():,} | features {X.shape[1]}")

# ─── REGRESIÓN (5 modelos) ───
print("\n[1/4] Regresión del % de ocupación...")
regs = {
    "Regresión Lineal": Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
    "Árbol de Decisión": DecisionTreeRegressor(max_depth=12, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1),
    "MLP": Pipeline([("sc", StandardScaler()),
                     ("m", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))]),
}
filas = []
for n, m in regs.items():
    t = time.time(); m.fit(Xtr, yr[tr]); p = m.predict(Xte); dt = time.time() - t
    filas.append({"Modelo": n, "RMSE": np.sqrt(mean_squared_error(yr[te], p)),
                  "MAE": mean_absolute_error(yr[te], p), "R2": r2_score(yr[te], p), "Tiempo_s": dt})
REG = pd.DataFrame(filas).sort_values("RMSE")
REG.to_csv("models/resultados_regresion.csv", index=False)
print(REG.round(3).to_string(index=False))

# Predicciones del test -> alimentan los gráficos de regresión del dashboard
pred_reg = pd.DataFrame({"real": yr[te].values})
for n, m in regs.items():
    pred_reg[n] = m.predict(Xte)
pred_reg["DEPARTAMENTO"] = df.loc[te, "DEPARTAMENTO"].values
pred_reg["MES"] = df.loc[te, "MES"].values
pred_reg["FECHA"] = df.loc[te, "FECHA"].values
pred_reg.to_csv("models/predicciones_regresion.csv", index=False)

# SELECCIÓN — la tabla del enunciado pide comparar RF regresor vs XGBoost.
# Empatan en RMSE (6.405 vs 6.406), pero el RF pesa ~343 MB y tarda 65 s;
# XGBoost pesa <1 MB y tarda 1.2 s. Mismo resultado, 340x más liviano.
mejor_reg = "XGBoost"
joblib.dump(regs[mejor_reg], "models/modelo_regresion.pkl")
print(f"→ Desplegado: {mejor_reg} ({os.path.getsize('models/modelo_regresion.pkl')/1e6:.2f} MB)")
print(f"   RF empata en RMSE pero pesa ~343 MB -> descartado por coste, no por precisión.")
print(f"   HALLAZGO HONESTO: la Regresión Lineal queda a "
      f"{REG[REG.Modelo=='Regresión Lineal'].RMSE.iloc[0] - REG.RMSE.min():.3f} de RMSE del mejor.")
print(f"   El modelo complejo casi no aporta precisión — hay que decirlo en el reporte.")

# ─── CLASIFICACIÓN (5 modelos) ───
print("\n[2/4] Clasificación de alta ocupación...")
ratio = (yc[tr] == 0).sum() / (yc[tr] == 1).sum()
clfs = {
    "Regresión Logística": Pipeline([("sc", StandardScaler()),
                                     ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    "Árbol de Decisión": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                             scale_pos_weight=ratio, random_state=42, n_jobs=-1, eval_metric="logloss"),
    "MLP": Pipeline([("sc", StandardScaler()),
                     ("m", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))]),
}
filas, fit = [], {}
for n, m in clfs.items():
    t = time.time(); m.fit(Xtr, yc[tr]); dt = time.time() - t
    p, pp = m.predict(Xte), m.predict_proba(Xte)[:, 1]
    fit[n] = m
    filas.append({"Modelo": n, "Accuracy": accuracy_score(yc[te], p),
                  "Precisión": precision_score(yc[te], p), "Recall": recall_score(yc[te], p),
                  "F1": f1_score(yc[te], p), "ROC_AUC": roc_auc_score(yc[te], pp), "Tiempo_s": dt})
CLF = pd.DataFrame(filas).sort_values("F1", ascending=False)
CLF.to_csv("models/resultados_clasificacion.csv", index=False)
print(CLF.round(3).to_string(index=False))

ganador = CLF.iloc[0].Modelo
modelo = fit[ganador]
print(f"→ Ganador: {ganador}")
joblib.dump(modelo, "models/modelo_clasificacion.pkl")
joblib.dump(list(X.columns), "models/features.pkl")
joblib.dump(0.45, "models/umbral.pkl")
np.save("models/matriz_confusion.npy", confusion_matrix(yc[te], modelo.predict(Xte)))

# ─── SHAP ───
print("\n[3/4] SHAP...")
base = modelo.named_steps["m"] if hasattr(modelo, "named_steps") else modelo
expl = shap.TreeExplainer(base)
muestra = Xte.sample(min(500, len(Xte)), random_state=42)
sv = expl.shap_values(muestra)
if isinstance(sv, list): sv = sv[1]
if sv.ndim == 3: sv = sv[:, :, 1]
imp = pd.DataFrame({"feature": X.columns, "shap_medio": np.abs(sv).mean(0)}) \
        .sort_values("shap_medio", ascending=False)
imp.to_csv("models/shap_importancia.csv", index=False)
np.save("models/shap_values.npy", sv)
muestra.to_csv("models/shap_muestra.csv", index=False)
joblib.dump(expl, "models/explainer.pkl")
print(imp.head(6).round(4).to_string(index=False))

# ─── SERIES ───
print("\n[4/4] Pronóstico...")
serie = pd.read_csv("ocupaperu_serie.csv", parse_dates=["FECHA"]).set_index("FECHA").asfreq("MS")
futuros = {}
for VAR in ["ARRIBOS", "TNOH_NACIONAL"]:
    y = serie[VAR]
    y_pc = y[y.index >= "2022-01-01"]                 # corte post-COVID
    train, test = y_pc[:-12], y_pc[-12:]
    pron = {
        "Media móvil (3m)": pd.Series([train[-3:].mean()] * 12, index=test.index),
        "Holt-Winters": ExponentialSmoothing(train, trend="add", seasonal="add",
                                             seasonal_periods=12).fit().forecast(12),
        "SARIMA(1,1,1)(1,1,1,12)": SARIMAX(train, order=(1, 1, 1),
                                           seasonal_order=(1, 1, 1, 12)).fit(disp=False).forecast(12),
    }
    tab = pd.DataFrame([{"Modelo": n,
                         "MAPE_%": mean_absolute_percentage_error(test, f) * 100,
                         "RMSE": np.sqrt(mean_squared_error(test, f))}
                        for n, f in pron.items()]).sort_values("MAPE_%")
    tab.to_csv(f"models/resultados_serie_{VAR}.csv", index=False)
    mejor = tab.iloc[0].Modelo
    print(f"  {VAR}: gana {mejor} (MAPE {tab.iloc[0]['MAPE_%']:.2f}%)")

    if "SARIMA" in mejor:
        f = SARIMAX(y_pc, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False).forecast(6)
    elif "Holt" in mejor:
        f = ExponentialSmoothing(y_pc, trend="add", seasonal="add", seasonal_periods=12).fit().forecast(6)
    else:
        f = pd.Series([y_pc[-3:].mean()] * 6,
                      index=pd.date_range(y_pc.index[-1] + pd.offsets.MonthBegin(), periods=6, freq="MS"))
    futuros[VAR] = f
pd.DataFrame(futuros).to_csv("models/pronostico_6meses.csv")
