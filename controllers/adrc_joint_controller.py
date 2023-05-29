import numpy as np
from observers.eso import ESO
from .controller import Controller
from models.manipulator_model import ManiuplatorModel



class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd

        A = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
        B = np.array([[0],
                      [b],
                      [0]])
        L = np.array([[3*p],
                      [3*p**2],
                      [p**3]])
        W = np.array([[1, 0, 0]])
        self.eso = ESO(A, B, W, L, q0, Tp)


    def set_b(self, b):
        self.b = b
        self.B =np.array([[0],
                          [self.b],
                          [0]],dtype=np.float32)
        self.eso.set_B(self.B)   
        
    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot, step):
        
        q = x[0]

        z_hat = self.eso.get_state()
        x_hat = z_hat[0]

        x_hat_dot = z_hat[1]

        f = z_hat[2]

        e = q_d - q
        e_dot = q_d_dot - x_hat_dot

        v = q_d_ddot + self.kd * e_dot + self.kp * e
        u = (v - f) / self.b
        self.eso.update(q, u)

        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1**2 + self.l1**2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2**2 + self.l2**2)
        self.m3 = 0.1
        self.r3 = 0.05
        self.I_3 = 2.0 / 5 * self.m3 * self.r3**2

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
        M_11 = alpha + 2 * beta * np.cos(x_hat)
        M_12 = gamma + beta * np.cos(x_hat)
        M_21 = gamma + beta * np.cos(x_hat)
        M_22 = gamma

        M = np.array([[M_11, M_12], [M_21, M_22]])

        M_inv = np.linalg.inv(M)

        self.set_b(M_inv[step, step])
        
        return u
