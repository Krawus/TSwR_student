import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp, m3=0.1, r3=0.05):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1**2 + self.l1**2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2**2 + self.l2**2)
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2.0 / 5 * self.m3 * self.r3**2

    def M(self, x):
        q1, q2, q1_dot, q2_dot = x

        d1 = self.l1 / 2
        d2 = self.l2 / 2
        alpha = (
            self.m1 * d1**2
            + self.I_1
            + self.m2 * (self.l1**2 + d2**2)
            + self.I_2
            + self.m3 * (self.l1**2 + self.l2**2)
            + self.I_3
        )
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        gamma = self.m2 * d2**2 + self.I_2 + self.m3 * self.l2**2 + self.I_3
        M_11 = alpha + 2 * beta * np.cos(q2)
        M_12 = gamma + beta * np.cos(q2)
        M_21 = gamma + beta * np.cos(q2)
        M_22 = gamma

        return np.array([[M_11, M_12], [M_21, M_22]])

    def C(self, x):
        q1, q2, q1_dot, q2_dot = x

        d1 = self.l1 / 2
        d2 = self.l2 / 2
        alpha = (
            self.m1 * d1**2
            + self.I_1
            + self.m2 * (self.l1**2 + d2**2)
            + self.I_2
            + self.m3 * (self.l1**2 + self.l2**2)
            + self.I_3
        )
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        gamma = self.m2 * d2**2 + self.I_2 + self.m3 * self.l2**2 + self.I_3

        C_11 = -beta * np.sin(q2) * q2_dot
        C_12 = -beta * np.sin(q2) * (q1_dot + q2_dot)
        C_21 = beta * np.sin(q2) * q1_dot
        C_22 = 0

        return np.array([[C_11, C_12], [C_21, C_22]])
