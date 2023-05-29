import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):

        Kd = np.array([[17, 0], [0, 17]])
        Kp = np.array([[27, 0], [0, 27]])

        q1, q2, q1_dot, q2_dot = x
        q = np.array([q1, q2])
        q_dot = np.array([q1_dot, q2_dot])

        q = q.reshape(2, 1)
        q_dot = q_dot.reshape(2, 1)

        q_r = q_r.reshape(2, 1)
        q_r_dot = q_r_dot.reshape(2, 1)
        q_r_ddot = q_r_ddot.reshape(2, 1)
 
        v = q_r_ddot + Kd @ (q_r_dot - q_dot) + Kp @ (q_r - q)
       
        tau = self.model.M(x) @ v + self.model.C(x) @ q_dot

        return tau
