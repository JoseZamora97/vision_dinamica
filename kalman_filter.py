import argparse
import cv2 as cv
import numpy as np


def center(x, y, w, h):
    return x + w // 2, y + h // 2


def plot_centers(frame, centers, color, sth):
    for i in range(len(centers)):
        if i != 0: frame = cv.line(frame, centers[i], centers[i - 1], color, sth)
    return frame


def plot_v(frame, vx, vy, origin, color=(0, 255, 255)):
    font, font_scale, thickness = cv.FONT_HERSHEY_SIMPLEX, 1, 2
    text = f"Vx: {vx}\nVy: {vy}"

    y, dy = origin[1], 50
    for i, line in enumerate(text.split('\n')):
        frame = cv.putText(frame, line, (origin[0], y), font, font_scale,
                           color, thickness, cv.LINE_AA, False)
        y = y + dy

    return frame


class Kalman:
    def __init__(self, A, H, X_ini, P_ini, R, Q):
        self.A = A
        self.H = H
        self.X = X_ini
        self.P = P_ini
        self.R = R
        self.Q = Q

    def predict(self, ):
        self.X = self.A @ self.X
        self.P = self.A @ (self.P @ self.A.T) + self.Q

    def correct(self, Z):
        H_R_HT = self.H @ (self.P @ self.H.T) + self.R
        K = self.P @ (self.H.T @ np.linalg.inv(H_R_HT))
        self.P = self.P - K @ (self.H @ self.P)
        self.X = self.X + K @ (Z - self.H @ self.X)


def execute_kalman_filter(sequence_path, player_data, r=0.5, q=10e-4):

    vc = cv.VideoCapture(sequence_path)
    fs = int(vc.get(cv.CAP_PROP_FRAME_COUNT))

    # Declare the Kalman filter params
    A = np.eye(4) + np.diag((1, 1), k=2)
    H = np.eye(2, 4)
    R = np.eye(2) * r
    Q = np.eye(4) * q
    X_ini = np.zeros((4, 1))
    P_ini = np.eye(4)

    # Crate the Kalman object with params
    kalman = Kalman(A, H, X_ini, P_ini, R, Q)

    # Create lists to plot the center (x, y) coords
    centers_or, centers_km = [], []
    color_or, color_km = (255, 0, 0), (0, 255, 255)

    i = 0
    while vc.isOpened():
        # Get current measurement
        x0, y0, w, h = player_data[i]
        # Prediction
        kalman.predict()
        # Update Z with measurement
        Z = [[x0], [y0]]
        # Correction
        kalman.correct(np.array(Z))
        # Get filtered measurement
        x, y, vx, vy = kalman.X
        # Read the frame
        ret, frame = vc.read()

        if ret:
            # Draw the vx, vy texts
            frame = plot_v(frame, vx, vy, (50, 970))

            # Draw the centers
            frame = plot_centers(frame, centers_or, color_or, 2)
            frame = plot_centers(frame, centers_km, color_km, 2)

            # Draw bounding boxes
            frame = cv.rectangle(frame, (x0, y0), (x0 + w, y0 + h), color_or, 2)
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), color_km, 3)

            # Update frame
            cv.imshow('KalmanFilter', frame)

            # Append new centers
            centers_or.append(center(x0, y0, w, h))
            centers_km.append(center(x, y, w, h))

            i += 1

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    vc.release()
    cv.destroyAllWindows()


def action(args):
    data = np.loadtxt(args.detections, delimiter=',', dtype=np.int32)
    execute_kalman_filter(args.images + '/%04d.jpg', data, args.r, args.q)


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Kalman Filter")

    # Add parser images.
    parser.add_argument(
        "--images", required=True,
        help="Images folder"
    )
    # Add parser images.
    parser.add_argument(
        "--detections", required=True,
        help="Detections file (.csv)"
    )
    # Add parser images.
    parser.add_argument(
        "--r", required=True, type=float,
        help="R value"
    )
    # Add parser images.
    parser.add_argument(
        "--q", required=True, type=float,
        help="Q value"
    )

    # Set the parser function.
    parser.set_defaults(func=action)
    # Parse the arguments.
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    parse_arguments()
