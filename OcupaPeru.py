"""
OcupaPerú — Programa de escritorio (ventana)

Uso:   python ventana.py

Hecho con:
  · Tkinter (viene con Python)   -> la ventana, pestañas, botones, listas y tablas
  · Matplotlib                   -> los gráficos, dentro de la ventana (con zoom y desplazamiento)
  · SQLite (viene con Python)    -> las consultas guardadas
Toda la lógica (datos, modelos, predicción, pronóstico) está en nucleo.py.
"""
import queue
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.dates as mdates
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

print("Cargando datos y modelos…")
import nucleo as nu

GRANATE, ORO, AZUL, VERDE = "#8B1A1A", "#C8A951", "#2E4057", "#5B8C5A"
FONDO = "#F6F4EF"


def tabla(padre, columnas, filas, anchos=None, alto=6, barra=False):
    """Crea una tabla (Treeview) con barra de desplazamiento y la llena con `filas`."""
    marco = ttk.Frame(padre)
    tv = ttk.Treeview(marco, columns=columnas, show="headings", height=alto, selectmode="browse")
    for i, c in enumerate(columnas):
        tv.heading(c, text=c)
        tv.column(c, width=(anchos[i] if anchos else 110), anchor="center")
    if barra:
        sb = ttk.Scrollbar(marco, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
    tv.pack(side="left", fill="both", expand=True)
    for f in filas:
        tv.insert("", "end", values=f)
    return marco, tv


def grafico(padre, figsize, con_barra=True):
    """Inserta un gráfico de Matplotlib dentro de la ventana."""
    fig = Figure(figsize=figsize, dpi=100, layout="constrained")   # se reacomoda solo al cambiar el tamaño
    lienzo = FigureCanvasTkAgg(fig, master=padre)
    if con_barra:
        barra = NavigationToolbar2Tk(lienzo, padre, pack_toolbar=False)
        barra.update()
        barra.pack(side="bottom", fill="x")
    lienzo.get_tk_widget().pack(side="top", fill="both", expand=True)
    return fig, lienzo


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OcupaPerú — Predicción de ocupación hotelera")
        self.geometry(f"1150x{min(840, self.winfo_screenheight() - 90)}+30+10")
        self.minsize(960, 640)
        self.configure(bg=FONDO)
        self.cola = queue.Queue()          # el pronóstico se calcula en otro hilo y avisa por aquí

        self._estilos()
        self._cabecera()
        self.pestanas = pestañas = ttk.Notebook(self)
        pestañas.pack(fill="both", expand=True, padx=10, pady=(6, 10))
        self.t_pron, self.t_mod, self.t_crud = ttk.Frame(pestañas), ttk.Frame(pestañas), ttk.Frame(pestañas)
        pestañas.add(self.t_pron, text="  Pronóstico  ")
        pestañas.add(self.t_mod, text="  Modelos  ")
        pestañas.add(self.t_crud, text="  Mis consultas  ")

        self._pronostico()
        self._modelos()
        self._consultas()

        self.after(100, self._revisar_cola)
        self.actualizar_pronostico()

    # ───────────────────────── apariencia ─────────────────────────
    def _estilos(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame", background=FONDO)
        s.configure("TLabel", background=FONDO, font=("Segoe UI", 10))
        s.configure("TNotebook", background=FONDO)
        s.configure("TNotebook.Tab", font=("Segoe UI", 11, "bold"), padding=(10, 5))
        s.map("TNotebook.Tab", background=[("selected", "white")])
        s.configure("TRadiobutton", background=FONDO, font=("Segoe UI", 10))
        s.configure("Treeview", rowheight=24, font=("Segoe UI", 10))
        s.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        s.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), foreground="white", background=GRANATE)
        s.map("Accent.TButton", background=[("active", "#A62323")])
        s.configure("Titulo.TLabel", font=("Segoe UI", 13, "bold"))
        s.configure("Nota.TLabel", font=("Segoe UI", 9), foreground="#555555")

    def _cabecera(self):
        barra = tk.Frame(self, bg=GRANATE)
        barra.pack(fill="x")
        tk.Label(barra, text="OcupaPerú", bg=GRANATE, fg="white",
                 font=("Segoe UI", 20, "bold")).pack(anchor="w", padx=14, pady=(8, 0))
        tk.Label(barra, text="Predicción de la ocupación hotelera en los 25 departamentos del Perú · datos abiertos de "
                             "MINCETUR 2015–2025 (Indicadores de Ocupabilidad, DGIETA-MINCETUR · licencia Open Data "
                             "Commons Attribution)", bg=GRANATE, fg="#F3DADA", font=("Segoe UI", 9),
                 wraplength=1100, justify="left").pack(anchor="w", padx=14, pady=(0, 8))

        kpis = ttk.Frame(self)
        kpis.pack(fill="x", padx=10, pady=(8, 0))
        datos = [("Registros", f"{len(nu.df):,}"),
                 ("Meses de historia", f"{nu.serie.FECHA.nunique()}"),
                 ("F1 · alta ocupación", f"{nu.res_clf.F1.max():.3f}"),
                 ("MAPE · arribos", f"{nu.res_arr['MAPE_%'].min():.2f}%")]
        for i, (tit, val) in enumerate(datos):
            k = tk.Frame(kpis, bg="white", highlightbackground="#DDD", highlightthickness=1)
            k.grid(row=0, column=i, sticky="ew", padx=4)
            kpis.columnconfigure(i, weight=1)
            tk.Label(k, text=tit, bg="white", fg="#666", font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=(6, 0))
            tk.Label(k, text=val, bg="white", fg=AZUL, font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=10, pady=(0, 6))

    # ───────────────────────── PESTAÑA 1: PRONÓSTICO ─────────────────────────
    def _pronostico(self):
        p = self.t_pron
        arriba = ttk.Frame(p)
        arriba.pack(fill="x", padx=12, pady=(10, 0))
        ttk.Label(arriba, text="Pronóstico nacional a 6 meses", style="Titulo.TLabel").pack(anchor="w")
        fila = ttk.Frame(arriba)
        fila.pack(fill="x", pady=4)
        ttk.Label(fila, text="¿Qué quieres pronosticar?").pack(side="left")
        self.var_pron = tk.StringVar(value="ARRIBOS")
        for txt, val in [("Arribos a hospedajes", "ARRIBOS"), ("Ocupación hotelera (TNOH)", "TNOH_NACIONAL")]:
            ttk.Radiobutton(fila, text=txt, value=val, variable=self.var_pron,
                            command=self.actualizar_pronostico).pack(side="left", padx=12)
        self.estado = ttk.Label(fila, text="", foreground=GRANATE)
        self.estado.pack(side="right")

        # primero lo de abajo (nota y tabla), para que siempre se vean; el gráfico ocupa lo que sobra
        self.nota_p = ttk.Label(p, text="", style="Nota.TLabel", wraplength=1080, justify="left")
        self.nota_p.pack(side="bottom", anchor="w", padx=12, pady=6)
        marco, self.tv_pron = tabla(p, ("Mes", "Valor"), [], anchos=[220, 220], alto=6)
        marco.pack(side="bottom", fill="x", padx=12, pady=(6, 0))

        zona = ttk.Frame(p)
        zona.pack(side="top", fill="both", expand=True, padx=12)
        self.fig_p, self.lienzo_p = grafico(zona, (9.5, 3.6))
        self.lienzo_p.mpl_connect("motion_notify_event", self._hover)
        self._puntos = None

    def actualizar_pronostico(self):
        var = self.var_pron.get()
        self.estado.config(text="Calculando pronóstico…")
        self.config(cursor="watch")

        def trabajo():                      # se ejecuta en otro hilo para que la ventana no se congele
            try:
                self.cola.put((var, nu.pronostico_6m(var)))
            except Exception as e:
                self.cola.put((var, e))
        threading.Thread(target=trabajo, daemon=True).start()

    def _revisar_cola(self):
        while not self.cola.empty():
            var, res = self.cola.get()
            self.config(cursor="")
            self.estado.config(text="")
            if isinstance(res, Exception):
                messagebox.showerror("Pronóstico", f"No se pudo calcular el pronóstico:\n{res}")
            elif var == self.var_pron.get():           # ignora respuestas de una opción ya cambiada
                self._pintar_pronostico(var, res)
        self.after(100, self._revisar_cola)

    def _pintar_pronostico(self, var, res):
        es_arr = var == "ARRIBOS"
        nombre = "Arribos a hospedajes" if es_arr else "Ocupación hotelera (TNOH, %)"
        serie, fc, ult6 = nu.serie, res["fc"], res["ult6"]

        self.fig_p.clear()
        ax = self.fig_p.add_subplot(111)
        ax.axvspan(np.datetime64("2020-03-01"), np.datetime64("2021-12-01"), color="grey", alpha=.15)
        ax.text(np.datetime64("2020-09-01"), 0.97, "COVID", transform=ax.get_xaxis_transform(),
                ha="center", va="top", color="#666", fontsize=9)
        ax.plot(serie.FECHA, serie[var], color=AZUL, lw=1.6, label="Histórico")
        ult_f, ult_v = serie.FECHA.iloc[-1], serie[var].iloc[-1]       # une el histórico con el pronóstico
        ax.plot([ult_f] + list(fc.index), [ult_v] + list(fc.values), "--", color=GRANATE, lw=2, label="Pronóstico")
        ax.plot(ult6.index, ult6.values, "-o", color=GRANATE, lw=3, ms=7, label="Próximos 6 meses")
        ax.set_title(f"{nombre}: histórico y pronóstico", fontsize=12, fontweight="bold")
        ax.grid(alpha=.25)
        ax.legend(loc="lower right", fontsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(
            (lambda v, _: f"{v / 1e6:.1f} M") if es_arr else (lambda v, _: f"{v:.0f} %")))

        # tooltip al pasar el mouse
        self.ax_p = ax
        self.annot = ax.annotate("", xy=(0, 0), xytext=(14, 14), textcoords="offset points",
                                 bbox=dict(boxstyle="round", fc="white", ec=GRANATE), fontsize=9, visible=False)
        xs = list(serie.FECHA) + list(fc.index)
        ys = list(serie[var]) + list(fc.values)
        self._puntos = (mdates.date2num(xs), np.array(ys, dtype=float), es_arr)
        self.lienzo_p.draw_idle()

        for fila in self.tv_pron.get_children():
            self.tv_pron.delete(fila)
        self.tv_pron.heading("Valor", text=nombre)
        for d, v in zip(ult6.index, ult6.values):
            self.tv_pron.insert("", "end", values=(f"{nu.MESES[d.month]} {d.year}",
                                                   f"{v:,.0f}" if es_arr else f"{v:.1f} %"))
        u = serie.FECHA.iloc[-1]
        self.nota_p.config(text=(
            f"Método: {res['metodo'].split('(')[0]} · error promedio de {res['mape']:.1f}% al probarlo con los últimos "
            f"12 meses con datos. Los datos llegan hasta {nu.MESES[u.month].lower()} de {u.year}; después el método "
            "extiende el patrón histórico, así que cuanto más lejos, menos precisa es la estimación. "
            "Pasa el mouse sobre la línea para ver cada valor; usa la barra de abajo para acercar o mover el gráfico."))

    def _hover(self, evento):
        if self._puntos is None or evento.inaxes is not getattr(self, "ax_p", None) or evento.xdata is None:
            return
        xs, ys, es_arr = self._puntos
        i = int(np.argmin(abs(xs - evento.xdata)))
        if abs(xs[i] - evento.xdata) > 20:                 # más de ~20 días de distancia: no mostrar
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self.lienzo_p.draw_idle()
            return
        fecha = mdates.num2date(xs[i])
        valor = f"{ys[i]:,.0f}" if es_arr else f"{ys[i]:.1f} %"
        self.annot.xy = (xs[i], ys[i])
        self.annot.set_text(f"{nu.MESES[fecha.month]} {fecha.year}\n{valor}")
        self.annot.set_visible(True)
        self.lienzo_p.draw_idle()

    # ───────────────────────── PESTAÑA 2: MODELOS ─────────────────────────
    def _modelos(self):
        sub = ttk.Notebook(self.t_mod)
        sub.pack(fill="both", expand=True, padx=8, pady=8)
        t_c, t_r, t_i = ttk.Frame(sub), ttk.Frame(sub), ttk.Frame(sub)
        sub.add(t_c, text="  Clasificación (alta ocupación)  ")
        sub.add(t_r, text="  Regresión (% de ocupación)  ")
        sub.add(t_i, text="  ¿Qué datos pesan más?  ")
        self._tab_clasificacion(t_c)
        self._tab_regresion(t_r)
        self._tab_importancia(t_i)

    def _tab_clasificacion(self, p):
        r = nu.res_clf.round(3)
        marco, _ = tabla(p, list(r.columns), r.values.tolist(), alto=len(r))
        marco.pack(fill="x", padx=10, pady=(10, 4))
        g = r.iloc[0]
        tk.Label(p, text=f"Ganador: {g.Modelo} — F1 = {g.F1:.3f} · ROC-AUC = {g.ROC_AUC:.3f}", bg="#E3F1E3",
                 fg="#245024", font=("Segoe UI", 10, "bold"), anchor="w", padx=10, pady=6).pack(fill="x", padx=10)

        cuerpo = ttk.Frame(p)
        cuerpo.pack(fill="both", expand=True, padx=10, pady=8)
        txt_lbl = ttk.Label(cuerpo, wraplength=400, justify="left")
        txt_lbl.pack(side="right", anchor="n", padx=16)
        izq = ttk.Frame(cuerpo)
        izq.pack(side="left", fill="both", expand=True)
        fig, lienzo = grafico(izq, (4.6, 3.6), con_barra=False)
        ax = fig.add_subplot(111)
        cm = nu.cm
        ax.imshow(cm, cmap="Reds")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=14,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Predicho: Normal", "Predicho: Alta"])
        ax.set_yticks([0, 1], ["Real: Normal", "Real: Alta"])
        ax.set_title("Matriz de confusión", fontweight="bold")
        lienzo.draw()

        tn, fp, fn, tp = cm.ravel()
        txt = (f"Verdaderos negativos (TN):  {tn:,}\nFalsos positivos (FP):  {fp:,}\n"
               f"Falsos negativos (FN):  {fn:,}\nVerdaderos positivos (TP):  {tp:,}\n\n"
               "¿Qué error cuesta más? El FN: el modelo dijo \"normal\", llegó lleno, el hotel no contrató → "
               "servicio colapsado y daño reputacional. El FP solo cuesta un mes de planilla.\n\n"
               f"Por eso el umbral es {nu.UMBRAL}, no 0.5: se sacrifica precisión para ganar recall en la clase "
               "\"alta ocupación\".")
        txt_lbl.config(text=txt)

    def _tab_regresion(self, p):
        r = nu.res_reg.round(3)
        marco, _ = tabla(p, list(r.columns), r.values.tolist(), alto=len(r))
        marco.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(p, bg="#FFF4D6", fg="#6B4E00", justify="left", anchor="w", padx=10, pady=6, wraplength=1050,
                 font=("Segoe UI", 10),
                 text="Hallazgo honesto: Random Forest y XGBoost dan el MISMO RMSE (6.405 vs 6.406), y la Regresión "
                      "Lineal queda a 0.02. El modelo complejo casi no aporta precisión. Se eligió XGBoost por coste: "
                      "RF pesaba 343 MB y tardaba 65 s; XGBoost pesa <1 MB y tarda 1.2 s.").pack(fill="x", padx=10)

        fila = ttk.Frame(p)
        fila.pack(fill="x", padx=10, pady=6)
        ttk.Label(fila, text="Modelo a graficar:").pack(side="left")
        self.modelos_reg = [c for c in nu.pred_reg.columns if c not in ("real", "DEPARTAMENTO", "MES", "FECHA")]
        self.cb_reg = ttk.Combobox(fila, values=self.modelos_reg, state="readonly", width=22)
        self.cb_reg.set("XGBoost" if "XGBoost" in self.modelos_reg else self.modelos_reg[0])
        self.cb_reg.pack(side="left", padx=8)
        self.cb_reg.bind("<<ComboboxSelected>>", lambda e: self._pintar_regresion())
        self.m_r2, self.m_rmse, self.m_err = tk.StringVar(), tk.StringVar(), tk.StringVar()
        for var in (self.m_r2, self.m_rmse, self.m_err):
            ttk.Label(fila, textvariable=var, font=("Segoe UI", 10, "bold")).pack(side="left", padx=14)

        zona = ttk.Frame(p)
        zona.pack(fill="both", expand=True, padx=10)
        self.fig_r, self.lienzo_r = grafico(zona, (7, 3.6))
        ttk.Label(p, style="Nota.TLabel", wraplength=1050,
                  text="Cuanto más cerca estén los puntos de la línea punteada, mejor predice el modelo. "
                       "Se queda algo corto en los meses de ocupación más alta.").pack(anchor="w", padx=10, pady=4)
        self._pintar_regresion()

    def _pintar_regresion(self):
        m = self.cb_reg.get()
        yv, yp = nu.pred_reg["real"], nu.pred_reg[m]
        resid = yv - yp
        r2 = 1 - (resid ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        self.m_r2.set(f"R² = {r2:.3f}")
        self.m_rmse.set(f"RMSE = {np.sqrt((resid ** 2).mean()):.2f} pts")
        self.m_err.set(f"Error medio = {resid.mean():+.2f} pts")
        self.fig_r.clear()
        ax = self.fig_r.add_subplot(111)
        ax.scatter(yv, yp, s=8, alpha=.35, color=AZUL)
        lim = [0, float(max(yv.max(), yp.max())) * 1.02]
        ax.plot(lim, lim, "--", color=GRANATE, label="Predicción perfecta")
        ax.set_xlabel("TNOH real (%)")
        ax.set_ylabel("TNOH predicho (%)")
        ax.set_title(f"Real vs. predicho — {m} (R² = {r2:.3f})", fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(alpha=.25)
        self.lienzo_r.draw_idle()

    def _tab_importancia(self, p):
        ttk.Label(p, text="¿Qué datos pesan más en la predicción?", style="Titulo.TLabel").pack(anchor="w", padx=10, pady=(10, 0))
        cuerpo = ttk.Frame(p)
        cuerpo.pack(fill="both", expand=True, padx=10, pady=6)
        txt_lbl = ttk.Label(cuerpo, wraplength=380, justify="left")
        txt_lbl.pack(side="right", anchor="n", padx=16)
        izq = ttk.Frame(cuerpo)
        izq.pack(side="left", fill="both", expand=True)
        fig, lienzo = grafico(izq, (6.5, 4.6), con_barra=False)
        top = nu.shap_imp.head(15).sort_values("shap_medio")
        ax = fig.add_subplot(111)
        ax.barh(top.feature, top.shap_medio, color=GRANATE)
        ax.set_title("Importancia global (|SHAP| medio)", fontweight="bold")
        ax.grid(axis="x", alpha=.25)
        lienzo.draw()
        txt_lbl.config(text=(
            "TNOH_LAG1 domina. ¿Es leakage?\n\n"
            "No. Es el TNOH del mes pasado — un dato conocido antes de predecir.\n\n"
            "El leakage era PORCENTAJE_TNOC (r ≈ 0.93 con el target), que es del mismo mes. "
            "Esa se excluyó, junto con pernoctaciones, arribos y empleo.\n\n"
            "Al sacarlas, el ROC-AUC bajó de ~0.99 a 0.953. Y ese es el punto: el 0.99 era engañoso."))

    # ───────────────────────── PESTAÑA 3: MIS CONSULTAS ─────────────────────────
    def _consultas(self):
        p = self.t_crud
        ttk.Label(p, text="Nueva consulta", style="Titulo.TLabel").pack(anchor="w", padx=12, pady=(10, 4))

        form = ttk.Frame(p)
        form.pack(fill="x", padx=12)
        self.f_dep = self._combo(form, 0, "Departamento", nu.DEPTOS, "LIMA", 20)
        self.f_cla = self._combo(form, 1, "Clase", nu.CLASES, "HOTEL", 18)
        self.f_cat = self._combo(form, 2, "Categoría", nu.CATEGS, "3 ESTRELLAS", 18)
        self.f_mes = self._combo(form, 3, "Mes", list(nu.MESES.values()), "Julio", 12)
        ttk.Label(form, text="Año").grid(row=0, column=4, sticky="w", padx=6)
        self.f_anio = tk.IntVar(value=2025)
        ttk.Spinbox(form, from_=2025, to=2030, textvariable=self.f_anio, width=8).grid(row=1, column=4, padx=6, sticky="w")

        botones = ttk.Frame(p)
        botones.pack(fill="x", padx=12, pady=8)
        ttk.Button(botones, text="Predecir y guardar", style="Accent.TButton",
                   command=self.guardar).pack(side="left")
        self.btn_act = ttk.Button(botones, text="Actualizar seleccionada", command=self.actualizar, state="disabled")
        self.btn_act.pack(side="left", padx=8)
        self.btn_del = ttk.Button(botones, text="Eliminar seleccionada", command=self.eliminar, state="disabled")
        self.btn_del.pack(side="left")
        ttk.Button(botones, text="Limpiar selección", command=self._deseleccionar).pack(side="left", padx=8)

        self.resultado = tk.StringVar()
        self.lbl_res = tk.Label(p, textvariable=self.resultado, bg="#E3F1E3", fg="#245024", anchor="w",
                                font=("Segoe UI", 10, "bold"), padx=10, pady=6, wraplength=1050, justify="left")
        self.lbl_res.pack(fill="x", padx=12)

        ttk.Label(p, text="Consultas guardadas", style="Titulo.TLabel").pack(anchor="w", padx=12, pady=(10, 2))
        cols = ("ID", "Departamento", "Clase", "Categoría", "Mes", "Año", "Predicción", "Probabilidad",
                "TNOH est.", "Fecha")
        marco, self.tv = tabla(p, cols, [], anchos=[45, 120, 110, 120, 90, 60, 130, 110, 85, 150], alto=10, barra=True)
        marco.pack(fill="both", expand=True, padx=12)
        self.tv.bind("<<TreeviewSelect>>", self._al_seleccionar)
        ttk.Label(p, style="Nota.TLabel", wraplength=1050,
                  text="Para editar o eliminar: haz clic en una fila de la tabla (sus datos pasan al formulario), "
                       "cámbialos y pulsa \"Actualizar seleccionada\". Las consultas se guardan en consultas.db "
                       "(SQLite local) con fecha y hora automáticas.").pack(anchor="w", padx=12, pady=6)
        self.refrescar()

    def _combo(self, padre, col, titulo, valores, defecto, ancho):
        ttk.Label(padre, text=titulo).grid(row=0, column=col, sticky="w", padx=6)
        cb = ttk.Combobox(padre, values=valores, state="readonly", width=ancho)
        cb.set(defecto if defecto in valores else valores[0])
        cb.grid(row=1, column=col, padx=6, sticky="w")
        return cb

    def _leer_formulario(self):
        mes = {v: k for k, v in nu.MESES.items()}[self.f_mes.get()]
        try:
            anio = int(self.f_anio.get())
        except tk.TclError:
            messagebox.showwarning("Año", "Escribe un año válido (2025–2030).")
            return None
        if not 2025 <= anio <= 2030:
            messagebox.showwarning("Año", "El año debe estar entre 2025 y 2030.")
            return None
        return self.f_dep.get(), self.f_cla.get(), self.f_cat.get(), mes, anio

    def _mostrar(self, texto, ok=True):
        self.lbl_res.config(bg="#E3F1E3" if ok else "#F8DADA", fg="#245024" if ok else "#7A1010")
        self.resultado.set(texto)

    def guardar(self):
        datos = self._leer_formulario()
        if datos is None:
            return
        r, err = nu.predecir(*datos)
        if err:
            self._mostrar(err, ok=False)
            return
        nu.crear_consulta(*datos, r)
        self._mostrar(f"{r['prediccion']} — probabilidad {r['probabilidad']:.1%} · "
                      f"TNOH estimado {r['tnoh_estimado']:.1f}% · consulta guardada")
        self.refrescar()

    def actualizar(self):
        sel = self.tv.selection()
        datos = self._leer_formulario()
        if not sel or datos is None:
            return
        id_ = int(self.tv.item(sel[0])["values"][0])
        r, err = nu.predecir(*datos)
        if err:
            self._mostrar(err, ok=False)
            return
        nu.actualizar_consulta(id_, *datos, r)
        self._mostrar(f"Consulta #{id_} actualizada → {r['prediccion']} ({r['probabilidad']:.1%})")
        self.refrescar()

    def eliminar(self):
        sel = self.tv.selection()
        if not sel:
            return
        id_ = int(self.tv.item(sel[0])["values"][0])
        if messagebox.askyesno("Eliminar consulta", f"¿Eliminar la consulta #{id_}?"):
            nu.eliminar_consulta(id_)
            self._mostrar(f"Consulta #{id_} eliminada")
            self.refrescar()

    def refrescar(self):
        for fila in self.tv.get_children():
            self.tv.delete(fila)
        for (id_, dep, cla, cat, mes, anio, pred, prob, tnoh, ts) in nu.listar_consultas():
            self.tv.insert("", "end", values=(id_, dep, cla, cat, nu.MESES[mes], anio, pred,
                                              f"{prob * 100:.1f}%", f"{tnoh:.1f}%", ts))
        self.btn_act.config(state="disabled")
        self.btn_del.config(state="disabled")

    def _al_seleccionar(self, _):
        sel = self.tv.selection()
        if not sel:
            return
        v = self.tv.item(sel[0])["values"]
        self.f_dep.set(v[1]); self.f_cla.set(v[2]); self.f_cat.set(v[3]); self.f_mes.set(v[4])
        self.f_anio.set(int(v[5]))
        self.btn_act.config(state="normal")
        self.btn_del.config(state="normal")

    def _deseleccionar(self):
        self.tv.selection_remove(self.tv.selection())
        self.btn_act.config(state="disabled")
        self.btn_del.config(state="disabled")


if __name__ == "__main__":
    if sys.platform == "win32":                      # nitidez en pantallas con escala > 100 %
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    App().mainloop()
