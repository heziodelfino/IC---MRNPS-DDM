import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

CAMINHO = Path(__file__).with_name("serie_3.csv")

COLUNA_TEMPO = "tempo_s"
COLUNA_ANGULO = "angulo_rad"

# A Série 3 extraída vai aproximadamente de 0.05 s até 244.80 s.
# usar None para pegar automaticamente o início/fim disponível.
TEMPO_INICIO = None
TEMPO_FIM = None
PONTOS_MINIMOS = 10

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

    plt.figure(figsize=(10, 6))

    plt.plot(tempo, angulo, label="Ângulo", alpha=0.8)

    plt.title(f"Ângulo vs tempo — Série 3 ({inicio:.2f}s a {fim:.2f}s)", fontsize=16)
    plt.xlabel("Tempo (s)", fontsize=14)
    plt.ylabel("Ângulo (rad)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    finalizar_figura("angulo_vs_tempo_serie3.png")


if __name__ == "__main__":
    main()
