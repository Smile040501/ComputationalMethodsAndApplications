"""
111901030
Mayank Singla
Coding Assignment 6 - Q3
"""

# %%
# Uncomment below line: This line is required to make the animation work in VSCode (Using `ipympl` as the backend for matplotlib plots)
# %matplotlib ipympl
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Below lines are to ignore the pylint warning in VSCode
# pylint: disable=abstract-method
# pylint: disable=pointless-string-statement
# pylint: disable=invalid-name


def solveODE(ode, theta0, v0, h, t0, T, g, L):
    """
    Uses the forward Euler method to solve the given ODE and show the animation of the solution.
    """

    # Required loop variables
    t = t0
    theta = theta0
    v = v0
    thetas = [theta0]  # List to store theta values

    # Applying the forward Euler method
    while t <= T:
        fn = ode(t, theta, v, g, L)
        theta = theta + h * fn[0]
        v = v + h * fn[1]
        thetas.append(theta)
        t += h

    def get_coords(th):
        """
        Return the (x, y) coordinates of the bob at angle th
        """
        return L * math.sin(th), -L * math.cos(th)

    # Figure and Axes of the plot
    fig = plt.figure()
    ax = fig.add_subplot(aspect="equal")

    # The initial position of the pendulum rod
    x0, y0 = get_coords(theta0)
    (line,) = ax.plot([0, x0], [0, y0], lw=3, c="k")

    # The pendulum bob
    bob_radius = 0.008
    bob = ax.add_patch(plt.Circle(get_coords(theta0), bob_radius, fc="r", zorder=3))

    # The pendulum rod
    patches = [line, bob]

    def init():
        """
        Initialization function for the animation.
        """
        # Title for the plot
        ax.set_title("Simple Gravity Pendulum")

        # Set the plot limits
        ax.set_xlim(-L * 1.5, L * 1.5)
        ax.set_ylim(-L * 1.5, L * 1.5)

        # Return everything that must be plotted at the start of the animation
        return patches

    def animate(i):
        """
        Update the animation at frame i
        """
        # Update the pendulum rod
        x, y = get_coords(thetas[i])
        line.set_data([0, x], [0, y])
        bob.set_center((x, y))

        # Return everything that must be updated
        return patches

    numFrames = len(thetas)  # Number of frames in the animation
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

    plt.show()

    return anim


if __name__ == "__main__":
    # The given ODE
    inp_ode = lambda t, theta, v, g, L: (v, -(g / L) * math.sin(theta))

    # Testing the function
    ani = solveODE(
        ode=inp_ode, theta0=math.pi / 4, v0=0, h=0.001, t0=0, T=10, g=10, L=0.1
    )
