import argparse
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

font, font_scale, thickness = cv.FONT_HERSHEY_SIMPLEX, 0.25, 1

min_h, max_h = 0, 179
min_s, max_s = 104, 255
min_v, max_v = 156, 255


def process_frame(frame):
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask = cv.inRange(frame_hsv, (min_h, min_s, min_v), (max_h, max_s, max_v))
    mask = cv.dilate(mask, (3, 3), iterations=3)
    mask = cv.erode(mask, (3, 3), iterations=3)

    contours, _ = cv.findContours(mask, cv.RETR_TREE,
                                  cv.CHAIN_APPROX_SIMPLE)

    rect = cv.boundingRect(contours[0]) if len(contours) > 0 else None

    return mask, rect


class ParticlesFilter:

    def __init__(self, N, world_shape, particles_size, sigma):
        self.N = N
        self.w, self.h = world_shape
        self.pw, self.ph = particles_size

        self.particles = []
        self.estimations = []
        self.sigma = sigma
        self.v = np.zeros((2,))

    def initiation(self, ):
        self.particles = [Particle.spawn_random(
            self.w, self.h, self.pw, self.ph, 1 / self.N
        ) for _ in range(self.N)]

    def draw(self, frame, color=(255, 255, 0),
             estimation_color=(255, 0, 255), text=''):

        for particle in self.particles:
            particle.draw(frame, color, thickness=1)

        cv.putText(frame, text, (10, 10), font, font_scale,
                   color, thickness, cv.LINE_AA, False)

        for particle in self.estimations[-1:]:
            particle.draw(frame, estimation_color, thickness=2)

    def evaluation(self, frame):
        weights = np.zeros((self.N,))
        W = np.sum(frame)

        for i, particle in enumerate(self.particles):
            x, y, w, h = particle.X
            window = frame[y:y + h, x:x + w]

            weight = np.sum(window) / (W + 1e-15)
            weights[i] = weight

        weights = weights / (np.sum(weights) + 1e-15)

        for i, particle in enumerate(self.particles):
            particle.w = weights[i]

    def estimation(self):
        p: Particle = sorted(self.particles, reverse=True)[0].copy()
        p_last: Particle = self.estimations[-1] if len(self.estimations) > 0 else p

        self.estimations.append(p)
        self.v = p - p_last

    def selection(self, ):
        new_particles = []

        weights = [p.w for p in self.particles]
        cumsum_weights = np.cumsum(weights)

        for _ in range(len(weights)):
            csw_greater = cumsum_weights > np.random.uniform()

            particle = self.particles[np.argmax(csw_greater)]
            particle.w = 1 / self.N
            new_particles.append(particle)

        self.particles = new_particles

    def diffusion(self):
        for i in range(self.N):
            p = self.particles[i]
            x, y, w, h = p.X

            w_p = self.w * self.sigma
            h_p = self.h * self.sigma

            x_noise = np.random.randint(-w_p, w_p)
            y_noise = np.random.randint(-h_p, h_p)

            x = max(min(x + x_noise, self.w), 0)
            y = max(min(y + y_noise, self.h), 0)

            p.X = np.array([x, y, w, h])

            self.particles[i] = p.copy()

    def motion_modeling(self):
        for p in self.particles:
            p.apply_motion(self.v)


@dataclass(repr=True, order=True)
class Particle:
    i = 0

    name: str = field(init=False, compare=False)
    X: np.array = field(init=True, compare=False)
    w: np.array = field(init=True, compare=True)

    def __post_init__(self, ):
        self.name = f'P{self.i}'
        self.update()

    def copy(self):
        return Particle(self.X, self.w)

    def __sub__(self, p) -> np.array:
        return self.X - p.X

    def apply_motion(self, v):
        self.X[:2] += v[:2]

    def draw(self, canvas, color=(0, 0, 255), thickness=2):
        x, y, w, h = self.X
        cv.rectangle(canvas, (x, y), (x + w, y + h),
                     color, thickness, cv.LINE_AA)
        cv.putText(canvas, self.name, (x, y - 5), font, font_scale,
                   color, 1, cv.LINE_AA, False)

    @classmethod
    def update(cls): cls.i += 1

    @classmethod
    def restart(cls): cls.i = 0

    @staticmethod
    def spawn_random(max_x, max_y, p_w, p_h, weight):
        X = [np.random.randint(0, max_x),
             np.random.randint(0, max_y), p_w, p_h]
        return Particle(np.array(X), weight)


def execute_particle_filter(sequence_path):

    vc = cv.VideoCapture(sequence_path)

    fs = int(vc.get(cv.CAP_PROP_FRAME_COUNT))
    ww = int(vc.get(cv.CAP_PROP_FRAME_WIDTH))
    wh = int(vc.get(cv.CAP_PROP_FRAME_HEIGHT))

    w_size = (ww, wh)

    # Ajustar la cantidad de partículas
    p_amount = 30

    # Ajustar el tamaño de la partícula
    p_s = 40
    p_size = (p_s, p_s)

    Particle.restart()
    particles_filter = ParticlesFilter(p_amount, w_size,
                                       p_size, sigma=0.09)
    particles_filter.initiation()

    while vc.isOpened():
        ret, frame = vc.read()

        if ret:

            mask, rect = process_frame(frame)

            if rect is not None:
                mask_aux = mask / 255.
                # Evaluation
                particles_filter.evaluation(mask_aux)
                # Estimation
                particles_filter.estimation()
                # Selection
                particles_filter.selection()
                # Diffusion
                particles_filter.diffusion()
                # Motion Modeling
                particles_filter.motion_modeling()

            mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            draw_in = mask_rgb

            particles_filter.draw(draw_in)

            cv.imshow('ParticlesFilter', draw_in)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    vc.release()
    cv.destroyAllWindows()


def action(args):
    execute_particle_filter(args.images + '/%1d.jpg')


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Kalman Filter")

    # Add parser images.
    parser.add_argument(
        "--images", required=True,
        help="Images folder"
    )

    # Set the parser function.
    parser.set_defaults(func=action)
    # Parse the arguments.
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    parse_arguments()