from manim import *
import numpy as np

from manim import *
import numpy as np

def load_dataset(which="clean", ic="default", path="pendulo_subcritico.npz"):
    """
    which: "noisy" ou "clean"
    ic:
      - "default": começa no início do arquivo
      - "vmax_at_theta0": redefine t=0 no ponto onde theta ~ 0 e |omega| ~ vmax
    """
    data = np.load(path)

    t = data["t"].astype(float)
    theta = data["x_noisy" if which == "noisy" else "x_clean"].astype(float)

    omega = np.gradient(theta, t)

    if ic == "vmax_at_theta0":
        omega_abs = np.abs(omega)
        omega_max = omega_abs.max() + 1e-12
        theta_scale = np.max(np.abs(theta)) + 1e-12

        score = (np.abs(theta) / theta_scale) + (1.0 - omega_abs / omega_max)
        idx0 = int(np.argmin(score))

        t = t[idx0:] - t[idx0]
        theta = theta[idx0:]
        omega = omega[idx0:]

    
        if omega[0] < 0:
            theta = -theta
            omega = -omega

        theta = theta - theta[0]

    return t, theta, omega


class PhaseSpacePendulum_invCI(Scene):
    def construct(self):
        t, theta, omega = load_dataset(which="clean", ic="vmax_at_theta0", path="pendulo_subcritico.npz")

        stride = 1 
        t = t[::stride]
        theta = theta[::stride]
        omega = omega[::stride]

        pad_x = 0.10 * (theta.max() - theta.min() + 1e-9)
        pad_y = 0.10 * (omega.max() - omega.min() + 1e-9)
        x_min, x_max = theta.min() - pad_x, theta.max() + pad_x
        y_min, y_max = omega.min() - pad_y, omega.max() + pad_y

        axes = Axes(
            x_range=[x_min, x_max, (x_max - x_min) / 5],
            y_range=[y_min, y_max, (y_max - y_min) / 5],
            x_length=6,
            y_length=6,
            axis_config={"include_numbers": True, "font_size": 18},
            tips=False,
        ).to_edge(LEFT, buff=1)

        x_label = axes.get_x_axis_label(MathTex(r"\theta"), edge=RIGHT, direction=RIGHT)
        y_label = axes.get_y_axis_label(MathTex(r"\dot{\theta}"), edge=UP, direction=UP)

        self.play(Create(axes), Write(x_label), Write(y_label))

        points = [axes.c2p(th, om) for th, om in zip(theta, omega)]
        total = len(points)

        
        k = ValueTracker(0)

        trace = VMobject()
        trace.set_stroke(width=2, opacity=0.4)
        trace.set_points_as_corners(points[:2])

        def trace_updater(mob):
            idx = int(k.get_value())
            idx = max(2, min(idx, total - 1))
            mob.set_points_as_corners(points[:idx+1])

        trace.add_updater(lambda m: trace_updater(m))

        mover = Dot(points[0], radius=0.06)
        mover.add_updater(lambda m: m.move_to(points[int(k.get_value())]))

        time_num = DecimalNumber(0, num_decimal_places=2).scale(0.6)
        time_tex = VGroup(MathTex("t="), time_num, MathTex(" s")).arrange(RIGHT, buff=0.06)
        time_tex.add_updater(lambda m: m.next_to(mover, UR))
        time_num.add_updater(lambda m: m.set_value(t[int(k.get_value())]))

        self.add(trace, mover, time_tex)

        self.play(
            k.animate.set_value(total - 1),
            run_time=18,
            rate_func=linear
        )

        start_dot = Dot(points[0], radius=0.07)
        end_dot = Dot(points[-1], radius=0.07)
        end_label = MathTex("t_{fim}").next_to(end_dot, UR).scale(0.6)
        self.play(FadeIn(start_dot), FadeIn(end_dot), Write(end_label))
        self.wait(2)

        # limpa updaters (boa prática)
        trace.clear_updaters()
        mover.clear_updaters()
        time_tex.clear_updaters()
        time_num.clear_updaters()

class PhaseSpacePendulum_invCI_Normalizado(Scene):
    def construct(self):
        t, theta, omega = load_dataset(which="clean", ic="vmax_at_theta0", path="pendulo_subcritico.npz")
        omega0 = float(np.load("pendulo_subcritico.npz")["omega0"])
        omega = omega / omega0

        stride = 1  
        t = t[::stride]
        theta = theta[::stride]
        omega = omega[::stride]

        pad_x = 0.10 * (theta.max() - theta.min() + 1e-9)
        pad_y = 0.10 * (omega.max() - omega.min() + 1e-9)
        x_min, x_max = theta.min() - pad_x, theta.max() + pad_x
        y_min, y_max = omega.min() - pad_y, omega.max() + pad_y

        axes = Axes(
            x_range=[x_min, x_max, (x_max - x_min) / 5],
            y_range=[y_min, y_max, (y_max - y_min) / 5],
            x_length=6,
            y_length=6,
            axis_config={"include_numbers": True, "font_size": 18},
            tips=False,
        ).to_edge(LEFT, buff=1)

        x_label = axes.get_x_axis_label(MathTex(r"\theta"), edge=RIGHT, direction=RIGHT)
        y_label = axes.get_y_axis_label(MathTex(r"\dot{\theta}/\omega_0"), edge=UP, direction=UP)

        self.play(Create(axes), Write(x_label), Write(y_label))

        points = [axes.c2p(th, om) for th, om in zip(theta, omega)]
        total = len(points)
        if total < 2:
            raise ValueError("Dataset curto demais para animar (precisa de pelo menos 2 pontos).")

        k = ValueTracker(0)

        trace = VMobject()
        trace.set_stroke(width=2, opacity=0.4)
        trace.set_points_as_corners(points[:2])

        def trace_updater(mob):
            idx = int(k.get_value())
            idx = max(2, min(idx, total - 1))
            mob.set_points_as_corners(points[:idx + 1])

        trace.add_updater(trace_updater)

        mover = Dot(points[0], radius=0.06)

        def mover_updater(mob):
            idx = int(k.get_value())
            idx = max(0, min(idx, total - 1))
            mob.move_to(points[idx])

        mover.add_updater(mover_updater)

        time_num = DecimalNumber(0, num_decimal_places=2).scale(0.6)
        time_tex = VGroup(MathTex("t="), time_num, MathTex(r"\,s")).arrange(RIGHT, buff=0.06)

        def time_tex_updater(mob):
            mob.next_to(mover, UR)

        def time_num_updater(mob):
            idx = int(k.get_value())
            idx = max(0, min(idx, total - 1))
            mob.set_value(t[idx])

        time_tex.add_updater(time_tex_updater)
        time_num.add_updater(time_num_updater)

        self.add(trace, mover, time_tex)

        self.play(
            k.animate.set_value(total - 1),
            run_time=18,
            rate_func=linear
        )

        start_dot = Dot(points[0], radius=0.07)
        end_dot = Dot(points[-1], radius=0.07)
        end_label = MathTex("t_{fim}").next_to(end_dot, UR).scale(0.6)
        self.play(FadeIn(start_dot), FadeIn(end_dot), Write(end_label))
        self.wait(2)

        trace.clear_updaters()
        mover.clear_updaters()
        time_tex.clear_updaters()
        time_num.clear_updaters()

