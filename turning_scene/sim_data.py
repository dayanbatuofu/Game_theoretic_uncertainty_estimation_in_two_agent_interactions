import pickle

class Sim_Data():

    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.states = [[]] * self.num_agents #actual states
        self.orientations = [[]] * self.num_agents
        self.speed = [[]] * self.num_agents
        self.FOV = [[]] * self.num_agents


    def record(self, agents):
        for i in range(self.num_agents):
            self.states[i] = agents[i].states
            self.orientations[i] = agents[i].orientation
            self.speed[i] = agents[i].states
            self.FOV[i] = agents[i].FOV