from manim import *
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# -------- CONFIG --------
DATA_FILE = "pendulo_subcritico.npz"
TCUT = 6.60

GAMMA1 = 0.05
GAMMA2 = 0.0125

RUN_TIME_TOTAL = 18
DOWNSAMPLE = 1  # aumenta p/ acelerar (ex: 2, 3, 5)

# -------- utils --------
def load_dataset_npz(filename=DATA_FILE):
    file = BASE_DIR / filename
    data = np.load(file)

    t = data["t"]
    x = data["x_clean"]

    # robusto: não assume dt perfeito
    v = np.gradient(x, t)

    return t, x, v

def nice_step(span, target_ticks=6):
    span = float(abs(span))
    if span == 0:
        return 1.0
    raw = span / target_ticks
    exp = 10 ** np.floor(np.log10(raw))
    frac = raw / exp
    if frac < 1.5:
        nice = 1
    elif frac < 3.5:
        nice = 2
    elif frac < 7.5:
        nice = 5
    else:
        nice = 10
    return nice * exp

def padded_range(arr, pad_frac=0.10):
    a_min = float(np.nanmin(arr))
    a_max = float(np.nanmax(arr))
    span = a_max - a_min
    if span <= 0:
        span = 1.0
    pad = pad_frac * span
    return a_min - pad, a_max + pad

class PhaseSpacePendulumTwoRegimes(Scene):
    def construct(self):
        t, x, v = load_dataset_npz(DATA_FILE)

        # downsample (performance)
        t = t[::DOWNSAMPLE]
        x = x[::DOWNSAMPLE]
        v = v[::DOWNSAMPLE]

        # índice do corte
        idx_cut = int(np.searchsorted(t, TCUT))
        idx_cut = int(np.clip(idx_cut, 2, len(t) - 2))

        # ranges automáticos
        x_min, x_max = padded_range(x, 0.10)
        v_min, v_max = padded_range(v, 0.10)

        x_step = nice_step(x_max - x_min, target_ticks=6)
        v_step = nice_step(v_max - v_min, target_ticks=6)

        axes = Axes(
            x_range=[x_min, x_max, x_step],
            y_range=[v_min, v_max, v_step],
            x_length=6,
            y_length=6,
            axis_config={"include_numbers": True, "font_size": 20},
            tips=False,
        ).to_edge(LEFT, buff=1)

        x_label = axes.get_x_axis_label(MathTex("x"), edge=RIGHT)
        y_label = axes.get_y_axis_label(MathTex(r"\dot{x}"), edge=UP)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # pontos no espaço de fase
        points = [axes.c2p(xi, vi) for xi, vi in zip(x, v)]
        n = len(points)

        # --- traço 1 (até o corte)
        trace1 = VMobject()
        trace1.set_stroke(width=2)
        trace1.set_opacity(0.6)
        trace1.set_color("#77b5b6")  # azul
        trace1.set_points_as_corners(points[:2])

        # --- traço 2 (depois do corte)
        trace2 = VMobject()
        trace2.set_stroke(width=2)
        trace2.set_opacity(0.6)
        trace2.set_color("#9671bd")  # magenta
        trace2.set_points_as_corners(points[idx_cut:idx_cut+2])

        # ponto móvel
        mover = Dot(points[0], radius=0.06, color="#77b5b6")

        # marca o ponto de troca
        cut_dot = Dot(points[idx_cut], radius=0.07, color=YELLOW)

        # label do gamma (troca no corte)
        gamma_lbl1 = MathTex(rf"\gamma={GAMMA1}").scale(0.7).to_corner(UR)
        gamma_lbl2 = MathTex(rf"\gamma={GAMMA2}").scale(0.7).to_corner(UR)

        self.add(trace1, mover, gamma_lbl1)

        # tempos proporcionais ao tamanho do trecho
        rt1 = RUN_TIME_TOTAL * (idx_cut / (n - 1))
        rt2 = RUN_TIME_TOTAL - rt1

        # -------- SEGMENTO 1 --------
        def update_trace1(mob, alpha):
            idx = max(2, int(alpha * (idx_cut - 1)))
            mob.set_points_as_corners(points[:idx])

        def update_mover1(mob, alpha):
            idx = int(alpha * (idx_cut - 1))
            mob.move_to(points[idx])

        self.play(
            UpdateFromAlphaFunc(trace1, update_trace1),
            UpdateFromAlphaFunc(mover, update_mover1),
            run_time=rt1,
            rate_func=linear
        )

        # evento do corte: marca e troca label/cor
        self.play(FadeIn(cut_dot), Transform(gamma_lbl1, gamma_lbl2), run_time=0.6)
        mover.set_color("#9671bd")
        self.add(trace2)  # começa a aparecer o traço 2

        # -------- SEGMENTO 2 --------
        def update_trace2(mob, alpha):
            idx = idx_cut + max(2, int(alpha * ((n - 1) - idx_cut)))
            mob.set_points_as_corners(points[idx_cut:idx])

        def update_mover2(mob, alpha):
            idx = idx_cut + int(alpha * ((n - 1) - idx_cut))
            mob.move_to(points[idx])

        self.play(
            UpdateFromAlphaFunc(trace2, update_trace2),
            UpdateFromAlphaFunc(mover, update_mover2),
            run_time=rt2,
            rate_func=linear
        )

        # marca início e fim
        start_dot = Dot(points[0], radius=0.07, color=WHITE)
        end_dot = Dot(points[-1], radius=0.07, color=WHITE)

        self.play(FadeIn(start_dot), FadeIn(end_dot))
        self.wait(2)
