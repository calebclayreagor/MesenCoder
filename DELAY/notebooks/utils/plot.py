import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional

class BubbleChart:
    def __init__(self,
                 area: pd.Series,
                 min_area: Optional[float] = None,
                 random_state: Optional[int] = None,
                 bubble_spacing: float = 0,
                 n_iterations: int = 100):
        
        self.area = area.copy()
        if min_area is not None:
            self.area = self.area.loc[self.area >= min_area]
        if random_state is not None:
            self.area = self.area.sample(
                frac = 1, random_state = random_state)

        # bubbles: x, y, r, area
        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(self.area), 4))
        self.bubbles[:, 2] = np.sqrt(self.area.values / np.pi)
        self.bubbles[:, 3] = self.area.values
        self.maxstep = 2 * self.bubbles[:, 2].max()
        self.maxstep += self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]
        self.com = self.center_of_mass()

        # collapse
        self.collapse(n_iterations = n_iterations)

    def plot(self,
             ax: Axes,
             fs_scale: float,
             fs_min: Optional[float] = None,
             fs_max: Optional[float] = None,
             facecolor: str = 'xkcd:sky blue',
             fontstyle: str = 'normal',
             ) -> None:
        
        for i in range(len(self.bubbles)):
            label_i = self.area.index[i]
            pos_i = self.bubbles[i, :2]
            rad_i = self.bubbles[i, 2]
            fs_i = rad_i * fs_scale / len(label_i)
            if fs_min is not None:
                fs_i = max(fs_i, fs_min)
            if fs_max is not None:
                fs_i = min(fs_i, fs_max)

            ax.add_patch(
                plt.Circle(pos_i, rad_i,
                           color = facecolor))
            
            ax.text(*pos_i, label_i,
                    size = fs_i,
                    fontstyle = fontstyle,
                    ha = 'center',
                    va = 'center')
            
        ax.axis(False)
        ax.relim()
        ax.autoscale_view()

    def collapse(self, n_iterations: int) -> None:
        """Move bubbles to the center of mass"""
        for _ in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):

                # try to move directly towards the center of mass
                rest_bub = np.delete(self.bubbles, i, 0)

                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    colliding = self.collides_with(new_bubble, rest_bub)

                    # calculate direction vector
                    dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                    dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                    # calculate orthogonal vector
                    orth = np.array([dir_vec[1], -dir_vec[0]])

                    # test which direction to go
                    new_point1 = (self.bubbles[i, :2] + orth * self.step_dist)
                    new_point2 = (self.bubbles[i, :2] - orth * self.step_dist)
                    dist1 = self.center_distance(self.com, np.array([new_point1]))
                    dist2 = self.center_distance(self.com, np.array([new_point2]))
                    new_point = new_point1 if dist1 < dist2 else new_point2
                    new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                    if not self.check_collisions(new_bubble, rest_bub):
                        self.bubbles[i, :] = new_bubble
                        self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def center_of_mass(self) -> np.ndarray:
        return np.average(self.bubbles[:, :2], axis = 0,
                          weights = self.bubbles[:, 3])

    def center_distance(self,
                        bubble: np.ndarray,
                        bubbles: np.ndarray
                        ) -> np.ndarray:
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self,
                         bubble: np.ndarray,
                         bubbles: np.ndarray
                         ) -> np.ndarray:
        center_distance = self.center_distance(bubble, bubbles)
        center_distance -= (bubble[2] + bubbles[:, 2])
        center_distance -= self.bubble_spacing
        return center_distance

    def check_collisions(self,
                         bubble: np.ndarray,
                         bubbles: np.ndarray
                         ) -> int:
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self,
                      bubble: np.ndarray,
                      bubbles: np.ndarray
                      ) -> int:
        distance = self.outline_distance(bubble, bubbles)
        return np.argmin(distance)