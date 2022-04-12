"""
111901030
Mayank Singla
Coding Assignment 6 - Q5
"""

# %%
# Uncomment below line: This line is required to make the animation work in VSCode (Using `ipympl` as the backend for matplotlib plots)
# %matplotlib ipympl
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# Below lines are to ignore the pylint warning in VSCode
# pylint: disable=abstract-method
# pylint: disable=pointless-string-statement
# pylint: disable=invalid-name


def solveODE(init_r, init_v, t0, T, n):
    """
    Solves the ODE for Three-Body problem and plot their trajectories.
    """

    def getnorm(r1, r2):
        """
        Returns the norm of the vector (r1 - r2). If norm is 0, returns some dummy value
        """
        return max(np.linalg.norm(r2 - r1), 10)

    def double_derivative(r1, r2, r3):
        """
        Returns the double derivative (r₁'') of the system of ODEs. The input formula.
        """
        rdd = ((r2 - r1) / (getnorm(r2, r1) ** 3)) + (
            (r3 - r1) / (getnorm(r3, r1) ** 3)
        )
        return list(rdd)

    def vdp_derivatives(t, y):
        """
        Returns the derivatives of the system of differential equations
        """
        r1x, r1y, r2x, r2y, r3x, r3y, v1x, v1y, v2x, v2y, v3x, v3y = y
        r1 = np.array([r1x, r1y])
        r2 = np.array([r2x, r2y])
        r3 = np.array([r3x, r3y])
        v1 = [v1x, v1y]
        v2 = [v2x, v2y]
        v3 = [v3x, v3y]
        v1d = double_derivative(r1, r2, r3)
        v2d = double_derivative(r2, r3, r1)
        v3d = double_derivative(r3, r1, r2)
        return [*v1, *v2, *v3, *v1d, *v2d, *v3d]

    # Time values for the plot at which we will evaluate points
    t = np.linspace(t0, T, n)

    # Solving the system of ODEs
    sol = solve_ivp(
        fun=vdp_derivatives, t_span=[t0, T], y0=[*init_r, *init_v], t_eval=t
    )

    # Values of points on the curve
    r1x, r1y, r2x, r2y, r3x, r3y, *vs = sol.y

    # Figure and Axes of the plot
    fig = plt.figure()
    ax = fig.add_subplot(aspect="equal")

    # The three bodies
    bob_radius = 0.1
    body1 = ax.add_patch(
        plt.Circle((r1x[0], r1y[0]), bob_radius, fc="r", label="Point1")
    )
    body2 = ax.add_patch(
        plt.Circle((r2x[0], r2y[0]), bob_radius, fc="b", label="Point2")
    )
    body3 = ax.add_patch(
        plt.Circle((r3x[0], r3y[0]), bob_radius, fc="g", label="Point3")
    )

    # Plotting the trajectories
    patches = [body1, body2, body3]

    def init():
        """
        Initialization function for the animation.
        """
        # Title for the plot
        ax.set_title("Three-Body Problem")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")

        # Set the plot limits
        ax.set_xlim(-2, 6)
        ax.set_ylim(-4, 4)

        # Return everything that must be plotted at the start of the animation
        return patches

    def animate(i):
        """
        Update the animation at frame i
        """
        # Update the positions of the circles
        body1.set_center((r1x[i], r1y[i]))
        body2.set_center((r2x[i], r2y[i]))
        body3.set_center((r3x[i], r3y[i]))

        # Return everything that must be updated at each frame
        return patches

    numFrames = len(r1x)  # Number of frames in the animation
    interval = 1  # Interval between frames in milliseconds

    # Setting up the animation
    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=numFrames,
        repeat=True,
        interval=interval,
        blit=True,
    )

    plt.legend()
    plt.show()

    return anim


if __name__ == "__main__":
    # Testing the function
    r10 = [0, 0]
    r20 = [3, 1.73]
    r30 = [3, -1.73]
    v10 = [0, 0]
    v20 = [0, 0]
    v30 = [0, 0]
    ani = solveODE(
        init_r=[*r10, *r20, *r30], init_v=[*v10, *v20, *v30], t0=0, T=400, n=1000
    )
