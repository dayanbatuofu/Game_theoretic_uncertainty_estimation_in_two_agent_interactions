import matplotlib.pyplot as plt
from inference_model import InferenceModel
from autonomous_vehicle import AutonomousVehicle
import numpy as np

class Plot:
    def __init__(self):
        #TODO: import state information, fill in the args
        self.states = AutonomousVehicle()
        self.p_state = InferenceModel.baseline_inference().state_probabilities_infer()

    def contour_plot(self):
        """
        Plots state information as (x, y) = (1D pos, speed) and z = state probability
        :return:
        """
        # TODO: multiple plots for different thetas
        #Pseudo code
        """
        x = self.states[pos]
        y = self.states[speed]
        z = self.p_state
        fig,(ax1, ax2, ax3, ...) = plt.subplots(nrows = 2) #plot separately for different thetas
        
        #------
        #plot 1
        #------
        ax1.contour(x, y, z, levels = 10, linewidths = 1,colors = 'k' )
        #TODO: modify the params
        cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")
        fig.colorbar(cntr1, ax=ax1)
        ax1.plot(x, y, 'ko', ms=3)
        ax1.set(xlim=(-2, 2), ylim=(-2, 2))
        ax1.set_title('state probability distribution with theta 1')
        """
        pass