import math,numpy as np

def angle_delta(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi
