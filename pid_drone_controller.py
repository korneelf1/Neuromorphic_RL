from environments import SimpleDrone

env = SimpleDrone(optic_flow=True)

class PID:
    def __init__(self,P,I,D):
        self.P = P
        self.I = I
        self.D = D

        self.buffer = []

    def act(self,obs):
        action = self.P*obs + self.I*integr + self.D*der

        return action