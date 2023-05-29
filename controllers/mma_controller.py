import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel
from manipulators.planar_2dof import PlanarManipulator2DOF


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        model1 = ManiuplatorModel(Tp, 0.1, 0.05)
        model2 = ManiuplatorModel(Tp, 0.01, 0.01)
        model3 = ManiuplatorModel(Tp, 1.0, 0.3)
        self.models = [model1, model2, model3]
        self.i = 0
        self.u = np.array([[0], [0]])

    def choose_model(self, x):
        prevError = np.inf
        q = x[:2]
        q_dot = x[2:]

        for modelIndex, model in enumerate(self.models):
            y = model.M(x) @ self.u + model.C(x) @ q_dot
            error = np.sum(abs(q - y[:2]))
            if error < prevError:
                prevError = error
                self.i = modelIndex

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]

        Kd = np.array([[17, 0], [0, 17]])
        Kp = np.array([[27, 0], [0, 65]])

        v = q_r_ddot + Kd @ (q_r_dot - q_dot) + Kp @ (q_r - q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]

        self.u = u
        
        return u
