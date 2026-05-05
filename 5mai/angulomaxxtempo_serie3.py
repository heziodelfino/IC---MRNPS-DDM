import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

CAMINHO = Path(__file__).with_name("serie_3.csv")

COLUNA_TEMPO = "tempo_s"
COLUNA_ANGULO = "angulo_rad"

TEMPO_INICIO = None
TEMPO_FIM = None
PONTOS_MINIMOS = 10

# False: detecta máximos positivos de angulo_rad.
# True: detecta máximos de |angulo_rad|, isto é, amplitude máxima em módulo.
USAR_MODULO_DO_ANGULO = False

MOSTRAR_GRAFICO = False  # mude para True se quiser abrir a janela do gráfico
PASTA_SAIDA = Path(__file__).parent / "figuras_serie_3"


def carregar_dados(caminho=CAMINHO):
    """Carrega a Série 3 exportada do Capstone e padroniza as colunas numéricas."""
    df = pd.read_csv(caminho)

    colunas_necessarias = [COLUNA_TEMPO, COLUNA_ANGULO]
    faltando = [col for col in colunas_necessarias if col not in df.columns]

    if faltando:
        raise ValueError(
            f"Colunas não encontradas no CSV: {faltando}\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )

    for col in colunas_necessarias:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=colunas_necessarias)
    df = df.sort_values(COLUNA_TEMPO).reset_index(drop=True)

    return df


def recortar_intervalo(df):
    """Aplica a máscara temporal, usando os limites reais quando None."""
    tempo_min = df[COLUNA_TEMPO].min()
    tempo_max = df[COLUNA_TEMPO].max()

    inicio = tempo_min if TEMPO_INICIO is None else TEMPO_INICIO
    fim = tempo_max if TEMPO_FIM is None else TEMPO_FIM

    mask = (df[COLUNA_TEMPO] >= inicio) & (df[COLUNA_TEMPO] <= fim)
    df_plot = df.loc[mask].copy()

    if len(df_plot) < PONTOS_MINIMOS:
        raise RuntimeError(f"Poucos pontos no intervalo {inicio}s–{fim}s.")

    return df_plot, inicio, fim


def estimar_distancia_picos(tempo, sinal):
    """
    Estima o período dominante via FFT para definir a distância mínima
    entre máximos locais no find_peaks.
    """
    dt = np.median(np.diff(tempo))

    if dt <= 0 or np.isnan(dt):
        dt = 1.0

    n = len(tempo)

    sinal_centralizado = sinal - np.mean(sinal)

    yf = np.fft.rfft(sinal_centralizado)
    xf = np.fft.rfftfreq(n, dt)

    if len(xf) > 1:
        idx = np.argmax(np.abs(yf)[1:]) + 1
        f0 = xf[idx]
        periodo_estimado = 1.0 / f0 if f0 > 0 else None
    else:
        periodo_estimado = None

    if periodo_estimado is not None:
        distance_samples = max(3, int(0.6 * periodo_estimado / dt))
        print(f"Período estimado via FFT: {periodo_estimado:.3f} s.")
        print(f"Distância mínima entre picos: {distance_samples} amostras.")
    else:
        distance_samples = 10
        print(f"Não foi possível estimar o período. Usando {distance_samples} amostras.")

    return distance_samples


def detectar_angulos_maximos(tempo, angulo, distance_samples):
    """Detecta máximos locais do ângulo, ou de seu módulo, se configurado."""
    if USAR_MODULO_DO_ANGULO:
        sinal_para_picos = np.abs(angulo)
    else:
        sinal_para_picos = angulo

    amplitude_range = np.max(sinal_para_picos) - np.min(sinal_para_picos)
    prominence = max(amplitude_range * 0.05, 1e-3)

    indices_picos, _ = find_peaks(
        sinal_para_picos,
        distance=distance_samples,
        prominence=prominence
    )

    tempo_picos = tempo[indices_picos]
    angulo_max = sinal_para_picos[indices_picos]

    print(f"-- {len(tempo_picos)} máximos de ângulo detectados.")

    return indices_picos, tempo_picos, angulo_max


def finalizar_figura(nome_arquivo):
    """Salva a figura e, opcionalmente, mostra na tela."""
    PASTA_SAIDA.mkdir(exist_ok=True)

    caminho_figura = PASTA_SAIDA / nome_arquivo
    plt.tight_layout()
    plt.savefig(caminho_figura, dpi=300)
    print(f"Figura salva em: {caminho_figura}")

    if MOSTRAR_GRAFICO:
        plt.show()
    else:
        plt.close()


def main():
    df = carregar_dados()
    df_plot, inicio, fim = recortar_intervalo(df)

    tempo = df_plot[COLUNA_TEMPO].to_numpy()
    angulo = df_plot[COLUNA_ANGULO].to_numpy()

    sinal_fft = np.abs(angulo) if USAR_MODULO_DO_ANGULO else angulo
    distance_samples = estimar_distancia_picos(tempo, sinal_fft)

    indices_picos, tempo_picos, angulo_max = detectar_angulos_maximos(
        tempo,
        angulo,
        distance_samples
    )

    plt.figure(figsize=(10, 6))

    if USAR_MODULO_DO_ANGULO:
        plt.plot(tempo, np.abs(angulo), label="|Ângulo|", alpha=0.45)
        titulo = f"Ângulo máximo em módulo vs tempo -- Série 3 ({inicio:.2f}s a {fim:.2f}s)"
        ylabel = "|Ângulo máximo| (rad)"
        nome_arquivo = "angulo_max_modulo_vs_tempo_serie3.png"
    else:
        plt.plot(tempo, angulo, label="Ângulo", alpha=0.45)
        titulo = f"Ângulo máximo vs tempo — Série 3 ({inicio:.2f}s a {fim:.2f}s)"
        ylabel = "Ângulo máximo (rad)"
        nome_arquivo = "angulo_max_vs_tempo_serie3.png"

    # Máximos detectados
    plt.plot(
        tempo_picos,
        angulo_max,
        "o-",
        label=f"Ângulos máximos detectados (N={len(tempo_picos)})",
        markersize=5,
    )

    plt.title(titulo, fontsize=16)
    plt.xlabel("Tempo (s)", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    finalizar_figura(nome_arquivo)


if __name__ == "__main__":
    main()
