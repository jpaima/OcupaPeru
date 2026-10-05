# OcupaPerú

Programa de escritorio (ventana con pestañas y gráficos interactivos) que predice y pronostica la **ocupación hotelera en los 25 departamentos del Perú**
con datos abiertos de MINCETUR (2015 – 2025).

**Fuente:** MINCETUR (DGIETA) — [Indicadores de Ocupabilidad](https://www.datosabiertos.gob.pe/dataset/indicadores-de-ocupabilidad-ministerio-de-comercio-exterior-y-turismo-mincetur) (datosabiertos.gob.pe), 11 archivos, 2015 – junio 2025. Licencia Open Data Commons Attribution: uso libre citando la fuente.
61,890 filas crudas → 23,702 filas listas para analizar.

## ¿Qué responde?

1. ¿Cómo se espera que evolucionen los arribos a hospedajes y la ocupación en los próximos 6 meses?
2. ¿Qué tan bien predicen los modelos y qué datos pesan más en la predicción?
3. Para un departamento, tipo de hospedaje y mes concreto: ¿será un mes de **alta ocupación**? ¿Con qué probabilidad y qué % de ocupación se estima?

## Cómo correrlo

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)

pip install -r requirements.txt
python entrenar.py                # entrena los modelos y genera models/  (~2 min, una sola vez)
python OcupaPeru.py                 # abre la ventana del programa
```

`entrenar.py` guarda los modelos con las versiones de librerías de **tu** equipo. Si usas modelos
entrenados en otra máquina puede salir `XGBoostError: input stream corrupted`. Entrenando local no ocurre.

## Qué contiene el programa

| Pestaña | Para qué sirve |
|---|---|
| Pronóstico | Un solo gráfico: serie histórica y proyección de arribos u ocupación para los próximos 6 meses (contados desde el mes actual) |
| Modelos | Compara 5 modelos, muestra la matriz de confusión y qué datos pesan más en la predicción |
| Mis consultas | Elige departamento, tipo de hospedaje, categoría, mes y año: el modelo predice si será alta ocupación, con su probabilidad y el % estimado. Las consultas se guardan y se pueden editar o eliminar (base de datos SQLite) |

## Estructura

```
├── OcupaPeru.py                    # Ventana del programa (Tkinter + Matplotlib)
├── nucleo.py                       # Datos, modelos, predicción, pronóstico y base SQLite
├── entrenar.py                     # Entrena los modelos -> genera models/
├── requirements.txt
├── ocupaperu_analitico.csv         # 23,702 filas x 31 variables
├── ocupaperu_serie.csv             # 126 meses (ocupación + arribos)
├── models/                         # modelos y resultados generados por entrenar.py
├── notebooks/                      # preparación y modelado paso a paso (versión Jupyter y versión Colab)
└── consultas.db                    # base de las consultas guardadas (se crea sola)
```

El programa **nunca entrena**: solo carga los modelos ya guardados y predice.

## Con qué está hecho

| Pieza | Para qué |
|---|---|
| **Tkinter** (incluido en Python) | La ventana, las pestañas, los botones, las listas y las tablas |
| **Matplotlib** | Los gráficos dentro de la ventana, con zoom, desplazamiento y valores al pasar el mouse |
| **SQLite** (incluido en Python) | Las consultas guardadas (`consultas.db`) |
| **pandas / scikit-learn / XGBoost / statsmodels** | Datos, modelos entrenados y pronóstico |

El pronóstico se calcula en un hilo aparte para que la ventana no se congele.

## Resultados principales

| Tarea | Mejor modelo | Resultado |
|---|---|---|
| Clasificar "alta ocupación" | XGBoost | F1 = 0.803 · ROC-AUC = 0.953 |
| Estimar el % de ocupación | XGBoost | Error típico (RMSE) = 6.4 puntos · R² = 0.824 |
| Pronosticar arribos | SARIMA | Error de 4.5 % (MAPE) |
| Pronosticar ocupación | Holt-Winters | Error de 3.1 % (MAPE) |

## Decisiones técnicas importantes

1. **Los archivos cambiaron de formato.** En 2024 MINCETUR renombró todas las columnas. Unirlos sin
   normalizar habría dejado columnas medio vacías sin avisar.
2. **Se evitó el "data leakage".** Algunas columnas (como la ocupación de camas, `PORCENTAJE_TNOC`) casi
   reconstruyen la respuesta. Al excluirlas, el ROC-AUC bajó de ~0.99 a 0.953: el 0.99 era engañoso.
3. **Validación en el tiempo, no al azar.** Se entrena con años anteriores y se prueba con 2024–2025,
   como ocurriría en la vida real. Un split aleatorio filtraría información del futuro.
4. **Umbral de 0.45, no 0.5.** No anticipar un mes lleno es más caro que prever de más, así que el
   modelo se inclina a detectar la mayor cantidad posible de meses de alta ocupación.
5. **El COVID no se ve igual en todos los indicadores.** La ocupación no cayó a 0 (los hoteles cerrados
   dejan de reportar), pero los arribos sí se desplomaron (−91 % en abril de 2020). Por eso los
   pronósticos usan datos desde 2022.
