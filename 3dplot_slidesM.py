from manim import *
from manim_slides import ThreeDSlide

class ThreeDExample(ThreeDSlide):
    def construct(self):
        axes = ThreeDAxes()
        circle = Circle(radius=3, color=BLUE)
        dot = Dot(color=RED)

        self.add(axes)

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        self.play(GrowFromCenter(circle))
        self.begin_ambient_camera_rotation(rate=75 * DEGREES / 4)

        self.next_slide()

        self.next_slide(loop=True)
        self.play(MoveAlongPath(dot, circle), run_time=4, rate_func=linear)
        self.next_slide()

        self.stop_ambient_camera_rotation()
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)

        self.play(dot.animate.move_to(ORIGIN))
        self.next_slide()

        self.play(dot.animate.move_to(RIGHT * 3))
        self.next_slide()

        self.next_slide(loop=True)
        self.play(MoveAlongPath(dot, circle), run_time=2, rate_func=linear)
        self.next_slide()

        self.play(dot.animate.move_to(ORIGIN))
