"""Finite simulation-time interpolation for existing kinematic gimbal axes.

This does not add actions or physical joints. A policy still requests exactly
its original endpoint; the live adapter renders the intermediate orientations.
"""
import math


def move_angles(start, target, *, dt, write, tick, rate_deg_s=10., max_delta_deg=5.):
    values = (*start, *target, dt, rate_deg_s, max_delta_deg)
    if len(start) != 2 or len(target) != 2 or not all(math.isfinite(float(v)) for v in values):
        raise ValueError('finite_two_axis_angles_required')
    if dt <= 0 or rate_deg_s <= 0 or max_delta_deg <= 0:
        raise ValueError('positive_motion_parameters_required')
    delta = max(abs(b-a) for a,b in zip(start,target))
    if delta > max_delta_deg + 1e-8:
        raise ValueError('gimbal_command_exceeds_original_step_limit')
    frames = max(1, math.ceil(delta/(rate_deg_s*dt)))
    trace = []
    for i in range(1,frames+1):
        angles = tuple(a+(b-a)*i/frames for a,b in zip(start,target))
        write(*angles)
        tick()
        trace.append({'frame':i,'elapsed_sim_s':i*dt,'yaw_deg':angles[0],'pitch_deg':angles[1]})
    return trace
