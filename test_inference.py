import numpy as np
from sklearn.processing import normalize
from autonomous_vehicle import AutonomousVehicle
class TestInference:

    def __init__(self,model,sim):

        # importing agents information
        self.sim = sim
        self.agents = AutonomousVehicle
        #self.curr_state = AutonomousVehicle.state  # cumulative #TODO: import this!
        self.goal = sim.goal  # CHECK THIS
        #self.traj = AutonomousVehicle.planned_trajectory_set  # TODO: check if this is right!
        self.T = 1  # one step look ahead/ Time Horizon

        "dummy data"
        self.curr_state = [0, 0, 0, 0] #sx, sy, vx, vy
        self.actions = [-2, -0.5, 0, 2] #accelerations (m/s^2)
        
    def baseline_inference(self):

        pass