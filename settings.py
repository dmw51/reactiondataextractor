""" This files allows dynamically creating a reference to the main figure and using it across multiple modules"""
import numpy as np

global main_figure
DISTANCE_FN_CHARS = lambda cc: 2.25 * np.max([cc.width, cc.height])

main_figure = []
