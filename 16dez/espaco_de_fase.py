from manim import *
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def load_dataset(regime="critico"):
    file = BASE_DIR / f"pendulo_{regime}.npz"

    data = np.load(file)

    t = data["t"]
    x = data["x_clean"]
    dt = t[1] - t[0]

    v = np.gradient(x, dt)
    return t, x, v


class PhaseSpacePendulum(Scene):
    def construct(self):
        # escolha do regime
        t, x, v = load_dataset("critico")  # "critico" ou "super"

        # limites do espaço de fase
        x_min, x_max = -0.25, 0.25
        v_min, v_max = -0.6, 0.6

        axes = Axes(
            x_range=[x_min, x_max, 0.1],
            y_range=[v_min, v_max, 0.2],
            x_length=6,
            y_length=6,
            axis_config={"include_numbers": True, "font_size": 20},
            tips=False,
        ).to_edge(LEFT, buff=1)

        x_label = axes.get_x_axis_label(MathTex("x"), edge=RIGHT)
        y_label = axes.get_y_axis_label(MathTex(r"\dot{x}"), edge=UP)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # converter dados para coordenadas do manim
        points = [axes.c2p(xi, vi) for xi, vi in zip(x, v)]

        # traço
        trace = VMobject()
        trace.set_stroke(width=2)
        trace.set_opacity(0.5)
        trace.set_points_as_corners(points[:2])

        # ponto móvel
        mover = Dot(points[0], radius=0.06)

        self.add(trace, mover)

        total_frames = len(points)

        def update_trace(mob, alpha):
            idx = max(2, int(alpha * (total_frames - 1)))
            mob.set_points_as_corners(points[:idx])

        def update_mover(mob, alpha):
            idx = int(alpha * (total_frames - 1))
            mob.move_to(points[idx])

        self.play(
            UpdateFromAlphaFunc(trace, update_trace),
            UpdateFromAlphaFunc(mover, update_mover),
            run_time=18,
            rate_func=linear
        )

        # marca início e fim
        start_dot = Dot(points[0], radius=0.07)
        end_dot = Dot(points[-1], radius=0.07)

        self.play(FadeIn(start_dot), FadeIn(end_dot))
        self.wait(2)
