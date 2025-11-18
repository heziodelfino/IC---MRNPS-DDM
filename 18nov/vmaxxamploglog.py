import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

# --- PARÂMETROS DO ARQUIVO E COLUNAS ---
caminho = r"C:\Users\Hézio\Downloads\pendulo pasco-longo(in).csv"
COLUNA_TEMPO = 'Tempo (s)'
COLUNA_VELOCIDADE_ANGULAR = 'Velocidade angular (rad/s)'
COLUNA_ÂNGULO = 'Ângulo (rad)'

# --- PARÂMETROS DA MÁSCARA ---
TEMPO_INICIO = 0
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
        # Pega o índice do maior pico, excluindo o primeiro elemento (f_{0}/DC)
        idx = np.argmax(np.abs(yf)[1:]) + 1
        f0 = xf[idx]
        period_est = 1.0 / f0 if f0 > 0 else None
    else:
        period_est = None

    if period_est is not None:
        distance_samples = max(3, int(0.6 * period_est / dt))
        print(f"Periodo estimado (FFT): {period_est:.3f} s. Distância de pico: {distance_samples} amostras.")
    else:
        distance_samples = 10
        print(f"Não foi possível estimar o período. Usando distância padrão: {distance_samples} amostras.")


    # ---------------------------------------------------------------------------
    # 4a. Detecta apenas picos positivos
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

    # --- 4b. Detectar picos de amplitude (Ângulo) ---

    # converter coluna de ângulo
    df_plot[COLUNA_ÂNGULO] = (
        df_plot[COLUNA_ÂNGULO].astype(str).str.strip().str.replace(',', '.', regex=False)
    )
    df_plot[COLUNA_ÂNGULO] = pd.to_numeric(df_plot[COLUNA_ÂNGULO], errors='coerce')

    angle = df_plot[COLUNA_ÂNGULO].values

    # usar a MESMA distância entre picos para sincronizar com a velocidade
    peaks_ang, _ = find_peaks(angle, distance=distance_samples)

    # pegar amplitudes nos mesmos instantes
    amp_peaks = angle[peaks_ang]

    # converter rad para graus
    amp_peaks_deg = np.degrees(amp_peaks)

    # alinhar quantidades de picos
    min_len = min(len(vel_peaks), len(amp_peaks_deg))
    vel_peaks = vel_peaks[:min_len]
    amp_peaks_deg = amp_peaks_deg[:min_len]

    # ---------------------------------------------------------------------------
    #   4c. Máscara para plot das amplitudes (Amp<-35°) e (Amp>-35°)
    # ---------------------------------------------------------------------------     
    mask_amp = (amp_peaks_deg > 35) & (amp_peaks_deg < 45)
    amp_filtrado = amp_peaks_deg[mask_amp]
    vel_filtrada = vel_peaks[mask_amp]

    cut = -35

    mask_esq = amp_peaks_deg < cut     # lado esquerdo do gráfico
    mask_dir = amp_peaks_deg >= cut    # lado direito do gráfico

    amp_esq = amp_peaks_deg[mask_esq]
    vel_esq = vel_peaks[mask_esq]

    amp_dir = amp_peaks_deg[mask_dir]
    vel_dir = vel_peaks[mask_dir]

    #1° plot
    plt.figure(figsize=(10,6))
    plt.plot(amp_esq, vel_esq, 'o-', markersize=4)

    plt.title('Velocidade máxima vs Amplitude (amp < -35°)')
    plt.xlabel('Amplitude (graus)')
    plt.ylabel('Velocidade máxima (rad/s)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    #2° plot
    plt.figure(figsize=(10,6))
    plt.plot(amp_dir, vel_dir, 'o-', markersize=4)

    plt.title('Velocidade máxima vs Amplitude (amp ≥ -35°)')
    plt.xlabel('Amplitude (graus)')
    plt.ylabel('Velocidade máxima (rad/s)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # ---------------------------------------------------------------------------
    # 5. Plotagem do gráfico com destaque dos picos
    # ---------------------------------------------------------------------------
        # --- 6. Gráfico velocidade_max vs amplitude ---
    plt.figure(figsize=(10,6))
    plt.plot(amp_peaks_deg, vel_peaks, 'o-', markersize=6)

    plt.title('Velocidade máxima vs Amplitude')
    plt.xlabel('Amplitude (graus)')
    plt.ylabel('Velocidade máxima (rad/s)')
    plt.grid(True, linestyle='--', alpha=0.7)
    

    plt.show()

    # --- 6. Gráfico velocidade_max vs amplitude ---
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(np.abs(amp_peaks_deg), np.abs(vel_peaks), 'o-', markersize=6)

    ax.set_aspect('auto')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.75)
    ax.grid(True, which='minor', linestyle='-', linewidth=0.25, alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title('Velocidade máxima vs |Amplitude| (log-log)')
    ax.set_xlabel('|Amplitude| (graus)')
    ax.set_ylabel('Velocidade máxima (rad/s)')

    plt.show()


    plt.show()

    
except FileNotFoundError:
    print(f"❌ Erro: O arquivo não foi encontrado no caminho: {caminho}")
except RuntimeError as e:
    print(f"❌ Erro na validação: {e}")
except Exception as e:
    print(f"❌ Ocorreu um erro ao processar ou plotar os dados: {e}")


