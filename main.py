from time import time
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

PROGN = 'flowpyfield'


class Utils:
    @staticmethod
    def lerp(a: Union[int, float], b: Union[int, float], t: float) -> float:
        """
        Examples
        --------
        ::

            # Linear interpolate mid point of 0 and 10
            assert Utils.lerp(0, 10, 0.5) == 5
        """
        if not isinstance(t, float):
            raise ValueError(f'Expected t to be float. Received {type(t)}.')
        return (a if t == 0.0
                else (b if t == 1.0
                      else ((1 - t) * a) + (t * b)))


# ----------------------------------------------------------------------------------------------------------------------

def run_quiver_field_simulation():
    global create_xyfield, plot_field, run_self

    def create_xyfield(shape: Tuple[int, int], step: float, pattern='sincos'):
        assert isinstance(step, float)

        steprows, stepcols = int(shape[0] / step), int(shape[1] / step)
        rows = np.arange(-steprows, steprows + 1) * step
        cols = np.arange(-stepcols, stepcols + 1) * step

        match pattern:
            case 'anticlockwise':
                field = [(-c, r) for r in rows for c in cols]
            case 'clockwise':
                field = [(r, c) for r in rows for c in cols]
            case 'sincos':
                field = [(np.sin(c), np.cos(r)) for r in rows for c in cols]
            case 'cossin':
                field = [(np.cos(c), np.sin(r)) for r in rows for c in cols]
            case _:
                raise ValueError('Invalid pattern')

        return rows, cols, field

    def plot_field(field, rows, cols):
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        quiver_props = dict(scale=30, width=0.005, color='y', alpha=0.9, )

        coordinates = [(r, c) for r in rows for c in cols]
        for (r, c), (fr, fc) in zip(coordinates, field):
            ax.quiver(r, c, fr, fc, **quiver_props)

        ax.set_title(f'flow field', color='w')
        ax.set_xlabel('X-axis', color='w')
        ax.set_ylabel('Y-axis', color='w')
        ax.tick_params(axis='both', colors='w')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        plt.gca().axis('equal')

        plt.savefig(f'{PROGN}_quiver_{int(time())}.png')

    def run_self():
        shape = (5, 5)
        x, y, field = create_xyfield(shape=shape, step=0.5, pattern='cossin' or 'sincos')
        plot_field(field, x, y)
        return 0

    return run_self()


# ----------------------------------------------------------------------------------------------------------------------

def run_particle_field_simulation():
    global setup_particle_flow_field, run_self, explore_path

    def setup_particle_flow_field():
        global step_size, width, height, max_width, max_height, step_size_visual, x, y, longest_trajectory_x, longest_trajectory_y
        # Set up parameters for the flow field
        step_size = 10
        width, height = 800, 600
        smooth = 50
        # Set up parameters for visualization
        max_width, max_height = 800, 600
        step_size_visual = 20
        # Initialize starting point for the flow field
        x, y = np.random.uniform(0, width), np.random.uniform(0, height)
        # Create flow field
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'o', color='red')  # Starting point
        # Initialize variables for tracking the particle's trajectory
        longest_trajectory_x, longest_trajectory_y = [], []

    def explore_path(x, y, current_trajectory_x, current_trajectory_y):
        """Function to perform DFS-like iteration"""
        while 0 <= x < width and 0 <= y < height:
            n = np.random.rand()  # Replace this with proper noise function
            x += np.cos(n) * step_size
            y += np.sin(n) * step_size
            current_trajectory_x.append(x)
            current_trajectory_y.append(y)

        # Update longest trajectory if the current path is longer
        global longest_trajectory_x, longest_trajectory_y
        if len(current_trajectory_x) > len(longest_trajectory_x):
            longest_trajectory_x, longest_trajectory_y = current_trajectory_x.copy(), current_trajectory_y.copy()

    def run_self():
        setup_particle_flow_field()
        # Perform DFS-like iteration from the starting point
        explore_path(x, y, [x], [y])
        # Plot the longest trajectory
        plt.plot(longest_trajectory_x, longest_trajectory_y, color='goldenrod')
        plt.plot(x, y, 'o', color='rebeccapurple')  # Ending point

        # Visualize flow field
        for y_visual in range(0, max_height, step_size_visual):
            for x_visual in range(0, max_width, step_size_visual):
                n_visual = np.random.rand()  # Replace with proper noise function
                current_point_visual = np.array([x_visual, y_visual])
                next_point_visual = current_point_visual + np.array(
                    [np.cos(n_visual) * step_size_visual, np.sin(n_visual) * step_size_visual])
                plt.plot([current_point_visual[0], next_point_visual[0]],
                         [current_point_visual[1], next_point_visual[1]],
                         color='gray')

        plt.title("Combined Flow Field and Visualization")
        plt.savefig(f'{PROGN}_path_explorer_{int(time())}.png')
        plt.show()
        return 0

    return run_self()


# ----------------------------------------------------------------------------------------------------------------------

def main():
    plt.style.use('dark_background')
    run_quiver_field_simulation()
    run_particle_field_simulation()
    return 0


if __name__ == '__main__':
    exit(main())

# ----------------------------------------------------------------------------------------------------------------------

"""
from time import time
from typing import Tuple, Union
from PIL import Image, ImageDraw
import numpy as np

def create_xyfield(shape: Tuple[int, int], step: float, pattern='sincos'):
    assert isinstance(step, float)

    steprows, stepcols = int(shape[0] / step), int(shape[1] / step)
    rows = np.arange(-steprows, steprows + 1) * step
    cols = np.arange(-stepcols, stepcols + 1) * step

    match pattern:
        case 'anticlockwise':
            field = [(-c, r) for r in rows for c in cols]
        case 'clockwise':
            field = [(r, c) for r in rows for c in cols]
        case 'sincos':
            field = [(np.sin(c), np.cos(r)) for r in rows for c in cols]
        case 'cossin':
            field = [(np.cos(c), np.sin(r)) for r in rows for c in cols]
        case _:
            raise ValueError('Invalid pattern')

    return rows, cols, field

def plot_field(field, rows, cols):
    img_size = (800, 800)
    img = Image.new("RGB", img_size, color="black")
    draw = ImageDraw.Draw(img)

    quiver_props = dict(scale=30, width=0.005, fill='yellow', arrow='last', )

    coordinates = [(r, c) for r in rows for c in cols]
    for (r, c), (fr, fc) in zip(coordinates, field):
        end_point = (r + fr, c + fc)
        draw.line([(r, c), end_point], fill='yellow', width=1)
        draw.polygon([end_point, (end_point[0] - 0.02 * fr, end_point[1] - 0.02 * fc),
                      (end_point[0] - 0.02 * fc, end_point[1] + 0.02 * fr)],
                     fill='yellow')

    img.save(f'flowplotfield_{int(time())}.png')

def run():
    shape = (5, 5)
    x, y, field = create_xyfield(shape=shape, step=0.5, pattern='cossin' or 'sincos')
    plot_field(field, x, y)
    return 0

"""

"""
#include <math.h> // add complier flag `-lm`
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 10
#define HEIGHT 10

typedef struct {
    float x;
    float y;
} Vector;

void initializeFlowField(Vector field[][WIDTH]) {
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            field[i][j].x = cos((float)j / WIDTH * 2 * M_PI);
            field[i][j].y = sin((float)i / HEIGHT * 2 * M_PI);
        }
    }
}

void printFlowField(Vector field[][WIDTH]) {
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("(%0.2f, %0.2f) ", field[i][j].x, field[i][j].y);
        }
        printf("\n");
    }
}

void moveParticle(Vector *particle, Vector field[][WIDTH]) {
    // Interpolate to get the vector at the particle's position
    float xIndex = (particle->x / WIDTH) * (WIDTH - 1);
    float yIndex = (particle->y / HEIGHT) * (HEIGHT - 1);

    int i = (int)yIndex;
    int j = (int)xIndex;

    float xFraction = xIndex - j;
    float yFraction = yIndex - i;

    Vector topLeft = field[i][j];
    Vector topRight = field[i][j + 1];
    Vector bottomLeft = field[i + 1][j];
    Vector bottomRight = field[i + 1][j + 1];

    Vector topInterp = {topLeft.x + xFraction * (topRight.x - topLeft.x),
                        topLeft.y + xFraction * (topRight.y - topLeft.y)};

    Vector bottomInterp = {
        bottomLeft.x + xFraction * (bottomRight.x - bottomLeft.x),
        bottomLeft.y + xFraction * (bottomRight.y - bottomLeft.y)};

    particle->x += topInterp.x + yFraction * (bottomInterp.x - topInterp.x);
    particle->y += topInterp.y + yFraction * (bottomInterp.y - topInterp.y);
}

int main() {
    Vector flowField[HEIGHT][WIDTH];
    initializeFlowField(flowField);

    printf("Flow Field:\n");
    printFlowField(flowField);

    // Initial position of the particle
    Vector particle = {1.0, 1.0};

    // Simulate particle movement through the flow field
    for (int i = 0; i < 10; i++) {
        moveParticle(&particle, flowField);
        printf("Particle Position: (%0.2f, %0.2f)\n", particle.x, particle.y);
    }

    return 0;
}
"""
