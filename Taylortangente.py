"""
Série de Taylor (Maclaurin) de y = tan(x)

    tan(x) = x + x^3/3 + 2x^5/15 + 17x^7/315 + 62x^9/2835 + ...

Termo geral (com números de Bernoulli B_2n):
    tan(x) = Σ_{n>=1} (-1)^(n-1) * 2^(2n) * (2^(2n) - 1) * B_2n / (2n)! * x^(2n-1)

Raio de convergência: |x| < π/2  (onde tan(x) tem suas assíntotas)
"""
from fractions import Fraction
from math import comb, factorial, pi
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# 1) Coeficientes exatos
# ---------------------------------------------------------------
def bernoulli(n_max):
    """Números de Bernoulli B_0..B_n_max como frações exatas."""
    B = [Fraction(0)] * (n_max + 1)
    B[0] = Fraction(1)
    for m in range(1, n_max + 1):
        B[m] = -sum(comb(m + 1, k) * B[k] for k in range(m)) / (m + 1)
    return B


def coeficientes_tan(n_termos):
    """Lista de (expoente, coeficiente) dos n_termos primeiros termos não nulos."""
    B = bernoulli(2 * n_termos)
    coefs = []
    for n in range(1, n_termos + 1):
        c = (Fraction((-1) ** (n - 1) * 4 ** n * (4 ** n - 1)) * B[2 * n]
             / factorial(2 * n))
        coefs.append((2 * n - 1, c))
    return coefs


def tan_taylor(x, n_termos):
    """Aproximação de tan(x) usando n_termos termos não nulos da série."""
    x = np.asarray(x, dtype=float)
    return sum(float(c) * x ** k for k, c in coeficientes_tan(n_termos))


def polinomio_texto(n_termos):
    partes = []
    for k, c in coeficientes_tan(n_termos):
        num, den = c.numerator, c.denominator
        termo = f"x^{k}" if k > 1 else "x"
        if den == 1:
            partes.append(termo if num == 1 else f"{num}{termo}")
        else:
            partes.append(f"{'' if num == 1 else num}{termo}/{den}")
    return " + ".join(partes) + " + ..."


# ---------------------------------------------------------------
# 2) Resultados numéricos
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("tan(x) ~", polinomio_texto(6))
    print("\nCoeficientes:")
    for k, c in coeficientes_tan(8):
        print(f"  x^{k:<2}: {str(c):>18}  = {float(c):.10f}")

    print("\nConvergência:")
    for x0 in (0.5, 1.0, 1.4, 1.6):
        print(f"\n  x = {x0}  (tan = {np.tan(x0):.8f})")
        for n in (1, 2, 3, 5, 10, 20):
            aprox = tan_taylor(x0, n)
            print(f"    {n:>2} termos: {aprox:>16.8f}   erro = {abs(aprox - np.tan(x0)):.2e}")

    # Conferência com SymPy (se instalado)
    try:
        import sympy as sp
        X = sp.symbols("x")
        print("\nSymPy:", sp.series(sp.tan(X), X, 0, 12))
    except ImportError:
        pass

    # -----------------------------------------------------------
    # 3) Gráficos
    # -----------------------------------------------------------
    ordens = [1, 2, 3, 5, 8]
    cores = plt.cm.viridis(np.linspace(0.15, 0.85, len(ordens)))
    x = np.linspace(-pi / 2 + 0.02, pi / 2 - 0.02, 800)

    # Gráfico 1: tan(x) e polinômios de Taylor
    plt.figure(figsize=(10, 6))
    plt.plot(x, np.tan(x), "k", lw=3, label="tan(x)")
    for n, cor in zip(ordens, cores):
        plt.plot(x, tan_taylor(x, n), color=cor, lw=2,
                 label=f"{n} termo(s)  (grau {2*n-1})")
    for a in (-pi / 2, pi / 2):
        plt.axvline(a, ls="--", color="gray")
    plt.ylim(-6, 6)
    plt.title("tan(x) e seus polinômios de Taylor")
    plt.xlabel("x"); plt.ylabel("y"); plt.grid(alpha=.3); plt.legend()
    plt.savefig("grafico_aproximacoes.png", dpi=150, bbox_inches="tight")

    # Gráfico 2: erro absoluto (escala log)
    plt.figure(figsize=(10, 6))
    xp = np.linspace(0.01, pi / 2 - 0.02, 600)
    for n, cor in zip(ordens, cores):
        plt.semilogy(xp, np.abs(tan_taylor(xp, n) - np.tan(xp)), color=cor,
                     lw=2, label=f"{n} termo(s)")
    plt.axvline(pi / 2, ls="--", color="gray")
    plt.title("Erro absoluto |P_n(x) - tan(x)|")
    plt.xlabel("x"); plt.ylabel("erro"); plt.grid(alpha=.3, which="both"); plt.legend()
    plt.savefig("grafico_erro.png", dpi=150, bbox_inches="tight")

    # Gráfico 3: convergência dentro e fora do raio π/2
    plt.figure(figsize=(10, 6))
    ns = np.arange(1, 26)
    for x0, cor in [(1.0, "tab:blue"), (1.5, "tab:orange"), (1.65, "tab:red")]:
        erros = [abs(tan_taylor(x0, n) - np.tan(x0)) for n in ns]
        plt.semilogy(ns, erros, "o-", color=cor, label=f"x = {x0}")
    plt.title("Erro x número de termos (raio de convergência = π/2 ≈ 1,571)")
    plt.xlabel("número de termos"); plt.ylabel("erro"); plt.grid(alpha=.3, which="both")
    plt.legend()
    plt.savefig("grafico_convergencia.png", dpi=150, bbox_inches="tight")

    plt.show()