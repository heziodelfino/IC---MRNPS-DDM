from manim import *
import numpy as np

def load_dataset(label="forte"):
    data = np.load("pendulum_synthetic.npz")
    t = data[f"{label}_t"]
    theta = data[f"{label}_th"]
    omega = data[f"{label}_om"]
    return t, theta, omega

class PhaseSpacePendulum(Scene):
    def construct(self):
        # carregar dataset (mude "medio" para "fraco" ou "forte")
        t, theta, omega = load_dataset("forte")

        # domínio
        x_min, x_max = -2.5, 2.5   # theta
        y_min, y_max = -4.0, 4.0   # omega

        axes = Axes(
            x_range=[x_min, x_max, 1],
            y_range=[y_min, y_max, 2],
            x_length=6,
            y_length=6,
            axis_config={"include_numbers": True, "font_size": 18},
            tips=False,
        ).to_edge(LEFT, buff=1)

        x_label = axes.get_x_axis_label(MathTex(r"\theta"), edge=RIGHT, direction=RIGHT)
        y_label = axes.get_y_axis_label(MathTex(r"\dot{\theta}"), edge=UP, direction=UP)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # converte os dados para coordenadas do manim
        points = [axes.c2p(th, om) for th, om in zip(theta, omega)]

        # traço inicial (vazio)
        trace = VMobject()
        trace.set_points_as_corners([points[0], points[1]])
        trace.set_stroke(width=2)
        trace.set_opacity(0.4)

        # ponto móvel
        mover = Dot(point=points[0], radius=0.06)
        mover_label = MathTex("t=0").next_to(mover, UR).scale(0.6)

        self.add(trace, mover, mover_label)

        # desenha o traço
        total_frames = len(points)
        def update_trace(mob, alpha):
            idx = int(alpha * (total_frames - 1))
            if idx < 2:
                mob.set_points_as_corners(points[:2])
            else:
                mob.set_points_as_corners(points[:idx])
        def update_mover(mob, alpha):
            idx = int(alpha * (total_frames - 1))
            mob.move_to(points[idx])

        # usar two animations em paralelo: atualização contínua
        self.play(
            UpdateFromAlphaFunc(trace, lambda m, a: update_trace(m, a)),
            UpdateFromAlphaFunc(mover, lambda m, a: update_mover(m, a)),
            run_time=18, rate_func=linear
        )

        start_dot = Dot(points[0], radius=0.07)
        end_dot = Dot(points[-1], radius=0.07)
        end_label = MathTex("t_{fim}").next_to(end_dot, UR).scale(0.6)
        self.play(FadeIn(start_dot), FadeIn(end_dot), Write(end_label))
        self.wait(2)
