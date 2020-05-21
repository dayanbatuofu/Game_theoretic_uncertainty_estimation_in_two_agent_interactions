from enum import IntEnum

#Discrete set of possible thetas (aggressiveness/intent)
class Thetas(IntEnum):
    AGGRESSIVE = 0
    NONAGRESSIVE = 1

def getThetaVal(theta):
    """
    Maps discrete intent space to numerical value 

    TODO: actual values in math--> implement
    """
    if theta == Thetas.AGGRESSIVE:
        return 1
    if theta == Thetas.NONAGRESSIVE:
        return 1


#Discrete set of possible actions
class Actions(IntEnum):
    HI_ACC = 0
    LO_ACC = 1
    NO_ACC = 2
    LO_BRAKE = 3
    HI_BRAKE = 4

def getActionVal(act):
    """
    Maps discrete action space to actual numerical value
    """
    
    if act == Actions.HI_ACC:
        return 10
    elif act == Actions.LO_ACC:
        return 1
    elif act == Actions.NO_ACC:
        return 0
    elif act == Actions.LO_BRAKE:
        return .1
    elif act == Actions.HI_BRAKE:
        return .01


#Discrete set of possible lambdas 
lambdas = {0.01, 0.1, 1, 10, 100}


