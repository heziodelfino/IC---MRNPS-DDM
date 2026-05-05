from manim import *
import numpy as np

config.background_color = WHITE


class PhaseSpaceDamping(Scene):
    def construct(self):
        omega0 = 1.0
        x0 = 2.2
        v0 = 0.0
        t_max = 18
        n_points = 900

        cases = [
            {
                "title": "Weak damping",
                "caption": "(a)",
                "zeta": 0.12,
                "colors": ["#beee62", "#70ae6e"],
            },
            {
                "title": "Moderate damping",
                "caption": "(b)",
                "zeta": 0.50,
                "colors": ["#378d94", "#77b5b6"],
            },
            {
                "title": "Strong damping",
                "caption": "(c)",
                "zeta": 1.40,
                "colors": ["#9671bd", "#6a408d"],
            },
        ]

        title = Text(
            "Phase space: damped oscillator",
            color=BLACK,
            font_size=34
        ).to_edge(UP)

        equation = MathTex(
            r"\ddot{x} + 2\zeta\omega_0\dot{x} + \omega_0^2x = 0",
            color=BLACK,
            font_size=30
        ).next_to(title, DOWN, buff=0.15)

        self.play(Write(title), FadeIn(equation, shift=DOWN))

        panels = VGroup()

        for case in cases:
            panel = self.create_panel(
                title=case["title"],
                caption=case["caption"],
                zeta=case["zeta"],
                colors=case["colors"],
                omega0=omega0,
                x0=x0,
                v0=v0,
                t_max=t_max,
                n_points=n_points,
            )
            panels.add(panel)

        panels.arrange(RIGHT, buff=0.55)
        panels.next_to(equation, DOWN, buff=0.4)

        self.play(FadeIn(panels))

        progress = ValueTracker(0)

        animated_curves = VGroup()
        animated_dots = VGroup()

        for panel in panels:
            axes = panel.axes
            phase_points = panel.phase_points
            colors = panel.colors

            # The curve is created after the panels are positioned.
            full_curve = VMobject()
            full_curve.set_points_smoothly([
                axes.c2p(x, v) for x, v in phase_points
            ])
            full_curve.set_stroke(width=4)
            full_curve.set_color_by_gradient(*colors)

            animated_curve = always_redraw(
                lambda full_curve=full_curve, progress=progress:
                full_curve.copy().pointwise_become_partial(
                    full_curve,
                    0,
                    progress.get_value()
                )
            )

            moving_dot = always_redraw(
                lambda phase_points=phase_points, axes=axes, progress=progress, colors=colors:
                Dot(
                    axes.c2p(
                        *phase_points[
                            min(
                                int(progress.get_value() * (len(phase_points) - 1)),
                                len(phase_points) - 1
                            )
                        ]
                    ),
                    radius=0.055,
                    color=colors[1],
                )
            )

            animated_curves.add(animated_curve)
            animated_dots.add(moving_dot)

        self.add(animated_curves, animated_dots)

        self.play(
            progress.animate.set_value(1),
            run_time=10,
            rate_func=linear
        )

        self.wait(2)

    def create_panel(
        self,
        title,
        caption,
        zeta,
        colors,
        omega0,
        x0,
        v0,
        t_max,
        n_points,
    ):
        axes = Axes(
            x_range=[-2.6, 2.6, 1],
            y_range=[-2.6, 2.6, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={
                "color": BLACK,
                "stroke_width": 2,
                "include_tip": False,
            },
            tips=False,
        )

        x_label = MathTex("x", color=BLACK, font_size=24)
        v_label = MathTex(r"\dot{x}", color=BLACK, font_size=24)

        x_label.next_to(axes.x_axis.get_end(), RIGHT, buff=0.08)
        v_label.next_to(axes.y_axis.get_end(), UP, buff=0.08)

        panel_title = Text(
            title,
            color=BLACK,
            font_size=24
        ).next_to(axes, UP, buff=0.25)

        zeta_label = MathTex(
            rf"\zeta = {zeta}",
            color=BLACK,
            font_size=24
        ).next_to(axes, DOWN, buff=0.22)

        caption_label = Text(
            caption,
            color=BLACK,
            font_size=26
        ).next_to(zeta_label, DOWN, buff=0.18)

        phase_points = self.calculate_phase_space(
            zeta=zeta,
            omega0=omega0,
            x0=x0,
            v0=v0,
            t_max=t_max,
            n_points=n_points,
        )

        panel = VGroup(
            axes,
            x_label,
            v_label,
            panel_title,
            zeta_label,
            caption_label,
        )

        panel.axes = axes
        panel.phase_points = phase_points
        panel.colors = colors

        return panel

    def calculate_phase_space(
        self,
        zeta,
        omega0,
        x0,
        v0,
        t_max,
        n_points,
    ):
        ts = np.linspace(0, t_max, n_points)
        dt = ts[1] - ts[0]

        state = np.array([x0, v0], dtype=float)
        points = []

        def f(state):
            x, v = state

            dxdt = v
            dvdt = -2 * zeta * omega0 * v - omega0**2 * x

            return np.array([dxdt, dvdt])

        for _ in ts:
            points.append((state[0], state[1]))

            k1 = f(state)
            k2 = f(state + 0.5 * dt * k1)
            k3 = f(state + 0.5 * dt * k2)
            k4 = f(state + dt * k3)

            state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        return points