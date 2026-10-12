import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from math import comb

st.set_page_config(page_title="Simulateurs de calcul numérique")
st.title("Simulateurs de calcul numérique")

F = {
    "sin x sur [0 ; π]": (np.sin, 0, np.pi, 2.0, (-0.1, 1.2)),
    "x² sur [0 ; 1]": (lambda x: x**2, 0, 1, 1/3, (-0.1, 1.1)),
    "√x sur [0 ; 1]": (np.sqrt, 0, 1, 2/3, (-0.1, 1.1)),
    "eˣ sur [0 ; 1]": (np.exp, 0, 1, np.e - 1, (0, 2.9)),
}

def trap(f, a, b, n):
    y = f(np.linspace(a, b, n + 1))
    return (b - a) / n * (y.sum() - (y[0] + y[-1]) / 2)

def simp(f, a, b, n):
    y = f(np.linspace(a, b, n + 1))
    return (b - a) / (3 * n) * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())

tab1, tab2 = st.tabs(["Intégration", "Splines vs Runge"])

with tab1:
    nom = st.selectbox("Fonction", list(F))
    meth = st.radio("Méthode", ["Simpson", "Trapèzes"], horizontal=True)
    n = st.slider("n (nombre de tranches)", 2, 64, 4, step=2)
    f, a, b, I, yr = F[nom]
    v = (simp if meth == "Simpson" else trap)(f, a, b, n)
    c1, c2, c3 = st.columns(3)
    c1.metric("Aire approchée", f"{v:.6f}")
    c2.metric("Valeur exacte", f"{I:.6f}")
    c3.metric("Erreur", f"{abs(v - I):.2e}")
    xs = np.linspace(a, b, 400)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(xs, f(xs), "k", lw=2)
    X = np.linspace(a, b, n + 1)
    h = (b - a) / n
    if meth == "Trapèzes":
        ax.fill_between(X, f(X), alpha=0.3, color="orange")
    else:
        t = np.linspace(0, 2, 21)
        for i in range(0, n, 2):
            y0, y1, y2 = f(X[i]), f(X[i + 1]), f(X[i + 2])
            ax.fill_between(X[i] + t * h, y0 + t * (y1 - y0) + t * (t - 1) / 2 * (y2 - 2 * y1 + y0), alpha=0.3, color="orange")
    ax.set_ylim(*yr)
    st.pyplot(fig)
    st.table([{"n": k, "Erreur trapèzes": f"{abs(trap(f, a, b, k) - I):.2e}", "Erreur Simpson": f"{abs(simp(f, a, b, k) - I):.2e}"} for k in (2, 4, 8, 16, 32)])
    st.caption("À tester : avec sin x, double n (erreur ÷4 pour les trapèzes, ÷16 pour Simpson). Avec √x, Simpson perd son avantage.")

with tab2:
    k = st.slider("Nombre de nœuds", 3, 21, 9)
    R = lambda x: 1 / (1 + 25 * x**2)
    dR = lambda x: -50 * x / (1 + 25 * x**2) ** 2
    xs = np.linspace(-1, 1, k); ys = R(xs); h = 2 / (k - 1)
    w = np.array([(-1) ** j * comb(k - 1, j) for j in range(k)], float)

    def lag(x):
        d = x - xs
        if np.any(np.abs(d) < 1e-12):
            return ys[np.argmin(np.abs(d))]
        t = w / d
        return (t * ys).sum() / t.sum()

    m = [dR(-1.0)]
    for i in range(k - 1):
        m.append(-m[i] + 2 * (ys[i + 1] - ys[i]) / h)

    def sp(x):
        i = min(k - 2, max(0, int(np.floor((x + 1) / h))))
        t = x - xs[i]
        return ys[i] + m[i] * t + (m[i + 1] - m[i]) / (2 * h) * t * t

    G = np.linspace(-1, 1, 400)
    L = np.array([lag(x) for x in G]); S = np.array([sp(x) for x in G])
    c1, c2 = st.columns(2)
    c1.metric("Erreur max, polynôme", f"{np.abs(L - R(G)).max():.3f}")
    c2.metric("Erreur max, spline", f"{np.abs(S - R(G)).max():.3f}")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(G, R(G), "k", lw=2, label="f(x) = 1/(1+25x²)")
    ax.plot(G, L, "r", label="Polynôme de Lagrange")
    ax.plot(G, S, color="orange", label="Spline quadratique")
    ax.plot(xs, ys, "ko", ms=4)
    ax.set_ylim(-0.6, 1.4); ax.legend(fontsize=8)
    st.pyplot(fig)
    st.caption("Monte à 15 nœuds : le polynôme oscille aux bords (phénomène de Runge), la spline reste stable.")
