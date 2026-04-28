from manim import *
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_dataset(regime="subcritico"):
    """
    Carrega pendulo_subcritico.npz, pendulo_critico.npz ou pendulo_super.npz.
    O arquivo precisa ter as chaves:
      - t
      - x_clean

    Aqui x_clean esta sendo tratado como coordenada angular theta(t).
    """
    file = BASE_DIR / f"pendulo_{regime}.npz"

    if not file.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file}")

    data = np.load(file)

    t = data["t"]
    x = data["x_clean"]

    dt = t[1] - t[0]
    v = np.gradient(x, dt)

    return t, x, v


def calculate_energies(theta, omega, m=1.0, L=1.0, g=9.81):
    """
    Energia potencial e cinética de um pêndulo simples.

    U = m g L (1 - cos(theta))
    K = 1/2 m (L omega)^2
    """
    U = m * g * L * (1 - np.cos(theta))
    K = 0.5 * m * (L * omega) ** 2
    E = K + U
    return K, U, E


class PhaseSpacePendulum(Scene):
    def construct(self):
        # ======================================================
        # 1. Dados do pêndulo
        # ======================================================
        regime = "subcritico"
        t, theta, omega = load_dataset(regime)

        K, U, E = calculate_energies(theta, omega)

        eps = 1e-12
        E_safe = np.maximum(E, eps)
        E0 = max(float(E[0]), eps)

        # Barras superiores: partilha instantânea K <-> U
        K_frac = K / E_safe
        U_frac = U / E_safe

        # Barra inferior: energia total restante
        E_frac = np.clip(E / E0, 0, 1)

        total_frames = len(t)

        def get_index(alpha):
            idx = int(alpha * (total_frames - 1))
            return int(np.clip(idx, 0, total_frames - 1))

        def conversion_text(i):
            if i == 0:
                return "início"

            dK = K[i] - K[i - 1]
            dU = U[i] - U[i - 1]

            if dK > 0 and dU < 0:
                return "U → K"
            elif dU > 0 and dK < 0:
                return "K → U"
            else:
                return "dissipação"

        # ======================================================
        # 2. Painel esquerdo: espaço de fase
        # ======================================================
        x_min, x_max = -0.25, 0.25
        v_min, v_max = -0.6, 0.6

        axes = Axes(
            x_range=[x_min, x_max, 0.1],
            y_range=[v_min, v_max, 0.2],
            x_length=5.8,
            y_length=5.8,
            axis_config={"include_numbers": True, "font_size": 18},
            tips=False,
        ).to_edge(LEFT, buff=0.7)

        title_phase = Text("Espaço de fase", font_size=26).next_to(axes, UP, buff=0.25)
        x_label = axes.get_x_axis_label(MathTex(r"\theta"), edge=RIGHT)
        y_label = axes.get_y_axis_label(MathTex(r"\dot{\theta}"), edge=UP)

        points = [axes.c2p(th, om) for th, om in zip(theta, omega)]

        # ======================================================
        # 3. Painel direito: barras de energia
        # ======================================================
        panel_title = Text("Conversão de energia", font_size=28)
        panel_title.to_edge(RIGHT, buff=1.0).shift(UP * 2.65)

        max_bar_width = 3.25
        bar_height = 0.38
        left_anchor = np.array([2.05, 0.0, 0.0])

        yK = 1.45
        yU = 0.45
        yE = -0.80

        def make_bar(width_frac, y, color):
            width = max(0.001, max_bar_width * float(width_frac))
            bar = Rectangle(
                width=width,
                height=bar_height,
                stroke_width=1,
                stroke_color=WHITE,
                fill_color=color,
                fill_opacity=0.88,
            )
            bar.move_to(left_anchor + RIGHT * (width / 2) + UP * y)
            return bar

        def make_bar_frame(y):
            frame = Rectangle(
                width=max_bar_width,
                height=bar_height,
                stroke_width=1.5,
                stroke_color=GRAY_B,
                fill_opacity=0,
            )
            frame.move_to(left_anchor + RIGHT * (max_bar_width / 2) + UP * y)
            return frame

        k_frame = make_bar_frame(yK)
        u_frame = make_bar_frame(yU)
        e_frame = make_bar_frame(yE)

        k_label = MathTex(r"K/E(t)", font_size=32).next_to(k_frame, LEFT, buff=0.25)
        u_label = MathTex(r"U/E(t)", font_size=32).next_to(u_frame, LEFT, buff=0.25)
        e_label = MathTex(r"E/E_0", font_size=32).next_to(e_frame, LEFT, buff=0.25)

        # ======================================================
        # 4. Entrada estática dos elementos
        # ======================================================
        self.play(
            Create(axes),
            Write(title_phase),
            Write(x_label),
            Write(y_label),
            run_time=1.5,
        )

        self.play(
            Write(panel_title),
            Create(k_frame),
            Create(u_frame),
            Create(e_frame),
            FadeIn(k_label),
            FadeIn(u_label),
            FadeIn(e_label),
            run_time=1.8,
        )

        # ======================================================
        # 5. Objetos animados sincronizados
        # ======================================================
        alpha_tracker = ValueTracker(0)

        trace = always_redraw(
            lambda: VMobject(color=BLUE)
            .set_stroke(width=2.5, opacity=0.65)
            .set_points_as_corners(
                points[: max(2, get_index(alpha_tracker.get_value()))]
            )
        )

        mover = always_redraw(
            lambda: Dot(
                points[get_index(alpha_tracker.get_value())],
                radius=0.065,
                color=YELLOW,
            )
        )

        k_bar = always_redraw(
            lambda: make_bar(
                K_frac[get_index(alpha_tracker.get_value())],
                yK,
                BLUE,
            )
        )

        u_bar = always_redraw(
            lambda: make_bar(
                U_frac[get_index(alpha_tracker.get_value())],
                yU,
                ORANGE,
            )
        )

        e_bar = always_redraw(
            lambda: make_bar(
                E_frac[get_index(alpha_tracker.get_value())],
                yE,
                GREEN,
            )
        )

        k_percent = always_redraw(
            lambda: VGroup(
                DecimalNumber(
                    100 * K_frac[get_index(alpha_tracker.get_value())],
                    num_decimal_places=1,
                    font_size=26,
                ),
                Text("%", font_size=23),
            )
            .arrange(RIGHT, buff=0.04)
            .next_to(k_frame, RIGHT, buff=0.2)
        )

        u_percent = always_redraw(
            lambda: VGroup(
                DecimalNumber(
                    100 * U_frac[get_index(alpha_tracker.get_value())],
                    num_decimal_places=1,
                    font_size=26,
                ),
                Text("%", font_size=23),
            )
            .arrange(RIGHT, buff=0.04)
            .next_to(u_frame, RIGHT, buff=0.2)
        )

        e_percent = always_redraw(
            lambda: VGroup(
                DecimalNumber(
                    100 * E_frac[get_index(alpha_tracker.get_value())],
                    num_decimal_places=1,
                    font_size=26,
                ),
                Text("%", font_size=23),
            )
            .arrange(RIGHT, buff=0.04)
            .next_to(e_frame, RIGHT, buff=0.2)
        )

        flow_text = always_redraw(
            lambda: Text(
                f"Conversão dominante: {conversion_text(get_index(alpha_tracker.get_value()))}",
                font_size=23,
            ).next_to(panel_title, DOWN, buff=0.35)
        )

        self.add(
            trace,
            mover,
            k_bar,
            u_bar,
            e_bar,
            k_percent,
            u_percent,
            e_percent,
            flow_text,
        )

        # Aqui acontece a animação de verdade:
        # o alpha vai de 0 até 1, e todos os always_redraw atualizam junto.
        self.play(
            alpha_tracker.animate.set_value(1),
            run_time=18,
            rate_func=linear,
        )

        # ======================================================
        # 6. Marca início e fim no espaço de fase
        # ======================================================
        start_dot = Dot(points[0], radius=0.07, color=GREEN)
        end_dot = Dot(points[-1], radius=0.07, color=RED)

        start_label = Text("início", font_size=20, color=GREEN).next_to(start_dot, UP, buff=0.12)
        end_label = Text("fim", font_size=20, color=RED).next_to(end_dot, DOWN, buff=0.12)

        self.play(
            FadeIn(start_dot),
            FadeIn(end_dot),
            FadeIn(start_label),
            FadeIn(end_label),
            run_time=1.2,
        )

        self.wait(2)
