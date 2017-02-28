from uuid import uuid4
import numpy as np

def rando_name(type = "short"):
    name = str(uuid4())
    if type == "short":
        return name.split("-")[-1]
    else:
        return name.replace("-", "")

def r_squared(Y_hat, Y):
    res = np.sum(np.square(np.subtract(Y.values, Y_hat.values))) #np.sum(np.square(Y.values - Y_hat.values))
    tot = np.sum(np.square(np.subtract(Y.values, np.mean(Y.values))))
    #print res, tot
    return 1. - ( res / tot )
