#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Dict

import numpy as np
import habitat_sim

try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
except ImportError:
    pass

try:
    import magnum as mn
except ImportError:
    pass

# import quadruped_wrapper
from habitat.tasks.ant_v2.ant_robot import AntV2Robot

#define some colors
red = mn.Color4(1.0, 0.0, 0.0, 1.0)
green = mn.Color4(0.0, 1.0, 0.0, 1.0)
blue = mn.Color4(0.0, 0.0, 1.0, 1.0)
black = mn.Color4(0.0, 0.0, 0.0, 1.0)
yellow = mn.Color4(1.0, 1.0, 0.0, 1.0)
purple = mn.Color4(1.0, 0.0, 1.0, 1.0)
#define some other constants (don't change these)
unit_x = mn.Vector3(1.0, 0.0, 0.0)
unit_y = mn.Vector3(0.0, 1.0, 0.0)
unit_z = mn.Vector3(0.0, 0.0, 1.0)

origin = mn.Vector3(0.0, 0.0, 0.0)

class AntV2SimDebugVisualizer():
    def __init__(self, sim):
        self._sim = sim
        self.dlr = self._sim.get_debug_line_render()

    def draw_axis(self):
        """Draw global XYZ as RGB debug lines."""
        self.dlr.draw_transformed_line(mn.Vector3(), unit_x, red)
        self.dlr.draw_transformed_line(mn.Vector3(), unit_y, green)
        self.dlr.draw_transformed_line(mn.Vector3(), unit_z, blue)

    def lerp_color(self, c1, c2, t):
        """Return the interpolated color at t in range [0,1]."""
        return c1 + (c2-c1)*t

    def draw_vector(self, position, direction):
        """Draw a given vector at some position."""
        #NOTE: this is line between two points, so 2nd entry should be a point
        self.dlr.draw_transformed_line(position, position+direction, black)

    def draw_path(self, points, c1=blue, c2=red):
        """Draw a path of points with a blue->red color gradient."""
        assert len(points) > 1

        #compute cumulative distance along the path at each point 
        point_dists = [0]
        for ix,point in enumerate(points):
            if ix > 0:
                prev = points[ix-1]
                point_dists.append(np.linalg.norm(point-prev)+point_dists[-1])
        
        #draw lines between each pair of points with color LERP from distances
        for ix,point in enumerate(points):
            if ix > 0:
                p1_c = self.lerp_color(c1, c2, point_dists[ix-1]/point_dists[-1])
                p2_c = self.lerp_color(c1, c2, point_dists[ix]/point_dists[-1])
                self.dlr.draw_transformed_line(points[ix-1], point, p1_c, p2_c)