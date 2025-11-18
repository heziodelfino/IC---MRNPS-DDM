import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

# --- PARÂMETROS DO ARQUIVO E COLUNAS ---
caminho = r"C:\Users\Hézio\Downloads\pendulo pasco-longo(in).csv"
COLUNA_TEMPO = 'Tempo (s)'
COLUNA_VELOCIDADE_ANGULAR = 'Velocidade angular (rad/s)'

# --- PARÂMETROS DA MÁSCARA ---
TEMPO_INICIO = 300
TEMPO_FIM = 1500
PONTOS_MINIMOS = 10

try:
    # ---------------------------------------------------------------------------
    # 1. Carrega e 'limpa' os dados do arquivo fornecido em 'caminho'
    # ---------------------------------------------------------------------------
    df = pd.read_csv(caminho, sep=';') 

    for col in [COLUNA_TEMPO, COLUNA_VELOCIDADE_ANGULAR]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(',', '.', regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remover NaNs após a conversão para evitar problemas na FFT
    df.dropna(subset=[COLUNA_TEMPO, COLUNA_VELOCIDADE_ANGULAR], inplace=True)

    # ---------------------------------------------------------------------------
    # 2. Aplicar a Máscara de Tempo
    # ---------------------------------------------------------------------------
    mask = (df[COLUNA_TEMPO] >= TEMPO_INICIO) & (df[COLUNA_TEMPO] <= TEMPO_FIM)
    df_plot = df[mask].copy()

    if len(df_plot) < PONTOS_MINIMOS:
        raise RuntimeError(f"Poucos pontos no intervalo {TEMPO_INICIO}s–{TEMPO_FIM}s.")

    # Definir as variáveis para o cálculo (equivalente a t_win e signal)
    t_win = df_plot[COLUNA_TEMPO].values
    signal = df_plot[COLUNA_VELOCIDADE_ANGULAR].values
    
    # ---------------------------------------------------------------------------
    # 3. Estima período dominante por FFT (para escolher 'distance' em find_peaks)
    # ---------------------------------------------------------------------------
    dt = np.median(np.diff(t_win))
    if dt <= 0 or np.isnan(dt):
        dt = 1.0

    N = len(t_win)
    # A FFT deve ser aplicada no sinal com a média removida (se não estiver centrado)
    yf = np.fft.rfft(signal - np.mean(signal)) 
    xf = np.fft.rfftfreq(N, dt)
    
    if len(xf) > 1:
        # Pega o índice do maior pico, excluindo o primeiro elemento (frequência zero/DC)
        idx = np.argmax(np.abs(yf)[1:]) + 1
        f0 = xf[idx]
        period_est = 1.0 / f0 if f0 > 0 else None
    else:
        period_est = None

    if period_est is not None:
        # A distância deve ser de aproximadamente 60% do período em número de amostras
        distance_samples = max(3, int(0.6 * period_est / dt))
        print(f"Periodo estimado (FFT): {period_est:.3f} s. Distância de pico: {distance_samples} amostras.")
    else:
        distance_samples = 10
        print(f"Não foi possível estimar o período. Usando distância padrão: {distance_samples} amostras.")


    # ---------------------------------------------------------------------------
    # 4. Detecta apenas picos positivos
    # ---------------------------------------------------------------------------
    # 'Prominence' ajuda a ignorar ruídos. Usamos 5% da faixa total da amplitude.
    amplitude_range = np.max(signal) - np.min(signal)
    prominence = max(amplitude_range * 0.05, 1e-3)
    
    # Detecta picos
    peaks_pos, properties = find_peaks(signal, distance=distance_samples, prominence=prominence)

    # Coordenadas dos picos
    t_peaks = t_win[peaks_pos]
    vel_peaks = signal[peaks_pos] 
    
    print(f"✅ {len(t_peaks)} picos positivos detectados.")

    # ---------------------------------------------------------------------------
    # 5. Plotagem do gráfico com destaque dos picos
    # ---------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Plotar o sinal original (filtrado)
    plt.plot(t_win, signal, label='Velocidade Angular', color='blue', alpha=0.7)
    
    # Plotar os picos detectados
    plt.plot(t_peaks, vel_peaks, 'o', label=f'Picos Detectados (N={len(t_peaks)})', color='red', markersize=6)
    plt.legend(fontsize = 20)

    
    plt.title(f'(Intervalo: {TEMPO_INICIO}s a {TEMPO_FIM}s)', fontsize=20)
    plt.xlabel('Tempo (s)', fontsize=20)
    plt.ylabel('Velocidade Angular (rad/s)', fontsize=20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.show() 
    
except FileNotFoundError:
    print(f"❌ Erro: O arquivo não foi encontrado no caminho: {caminho}")
except RuntimeError as e:
    print(f"❌ Erro na validação: {e}")
except Exception as e:
    print(f"❌ Ocorreu um erro ao processar ou plotar os dados: {e}")