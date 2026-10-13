import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Integración de Riemann",
    page_icon="∫",
    layout="wide"
)

st.title("∫ Integración numérica")
st.subheader("Método de Riemann por puntos medios")

st.write(
    "Calcula una integral definida mediante rectángulos "
    "y observa su representación gráfica."
)

st.markdown("### 1. Define la integral")

funcion = st.text_input(
    "Función f(x)",
    value="x**2",
    help="Usa ** para potencias. Ejemplo: x**2 + 2*x + 1"
)

col1, col2, col3 = st.columns(3)

with col1:
    a = st.number_input("Límite inferior (a)", value=0.0)

with col2:
    b = st.number_input("Límite superior (b)", value=2.0)

with col3:
    n = st.number_input(
        "Número de subintervalos (n)",
        min_value=1,
        max_value=1000,
        value=4,
        step=1
    )

if st.button("Calcular integral", type="primary"):

    try:
        if a == b:
            st.warning("Los límites de integración son iguales.")
        else:
            # Evaluar la función en un entorno restringido
            def f(x):
                return eval(
                    funcion,
                    {"__builtins__": {}},
                    {"x": x, "np": np}
                )

            n = int(n)
            dx = (b - a) / n

            # Puntos medios de cada subintervalo
            puntos = a + (np.arange(n) + 0.5) * dx
            alturas = np.asarray(f(puntos), dtype=float)

            if alturas.ndim == 0:
                alturas = np.full(n, float(alturas))

            if alturas.shape != puntos.shape:
                st.error("La función no produjo un valor válido por subintervalo.")
                st.stop()

            if not np.all(np.isfinite(alturas)):
                st.error("La función produjo valores no finitos.")
                st.stop()

            # Suma de Riemann por puntos medios
            integral = dx * np.sum(alturas)

            st.markdown("### 2. Resultado")

            st.metric(
                label="Integral aproximada",
                value=f"{integral:.10f}"
            )

            st.latex(
                r"\int_a^b f(x)\,dx"
                r"\approx \Delta x\sum_{i=1}^{n}f(x_i^*)"
            )

            st.write(f"**Ancho de cada subintervalo:** Δx = {dx:.6f}")
            st.write(f"**Número de rectángulos:** {n}")

            # Gráfica de la función y los rectángulos
            x_grafica = np.linspace(a, b, 500)
            y_grafica = np.asarray(f(x_grafica), dtype=float)

            if y_grafica.ndim == 0:
                y_grafica = np.full_like(x_grafica, float(y_grafica))

            if y_grafica.shape != x_grafica.shape:
                st.error("No se pudo representar la función en la gráfica.")
                st.stop()

            if not np.all(np.isfinite(y_grafica)):
                st.error("La función no está definida en toda la región de la gráfica.")
                st.stop()

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.plot(
                x_grafica,
                y_grafica,
                color="royalblue",
                label="f(x)"
            )

            ax.bar(
                puntos,
                alturas,
                width=abs(dx),
                align="center",
                alpha=0.35,
                edgecolor="black",
                color="orange",
                label="Rectángulos de Riemann"
            )

            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.set_title("Suma de Riemann por puntos medios")
            ax.grid(True, alpha=0.25)
            ax.legend()

            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "La suma de Riemann es una aproximación numérica; "
                "su precisión depende de la función y del número de subintervalos."
            )

    except Exception as error:
        st.error(
            "No se pudo evaluar la función. Revisa su escritura. "
            "Ejemplo: x**2 + 2*x. "
            f"Detalle: {error}"
        )