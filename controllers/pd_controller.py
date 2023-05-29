import numpy as np
from .controller import Controller


class PDDecentralizedController(Controller):
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def calculate_control(self, q, q_dot, q_d, q_d_dot, q_d_ddot):
        Kd = np.array([[17, 0], [0, 17]])
        Kp = np.array([[27, 0], [0, 27]])

        u = Kd @ (q_d_dot - q_dot) - Kp @ (q_d - q)

        return u
