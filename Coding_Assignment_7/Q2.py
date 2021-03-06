"""
111901030
Mayank Singla
Coding Assignment 7 - Q1
"""

# %%
# Uncomment below line: This line is required to make the animation work in VSCode (Using `ipympl` as the backend for matplotlib plots)
# %matplotlib ipympl
from matplotlib import projections
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np


# Below lines are to ignore the pylint warning in VSCode
# pylint: disable=abstract-method
# pylint: disable=pointless-string-statement
# pylint: disable=invalid-name


def get_tridiag_matrix(n, a, b, c):
    """
    Returns a tri-diagonal matrix of size n x n with the given diagonal elements
    """
    return np.eye(n, k=-1) * a + np.eye(n, k=0) * b + np.eye(n, k=1) * c


def solve2DHeatEquation(uT0, uB, f, mu, T, h, ht, xmin, xmax, ymin, ymax, xc, yc):
    """
    Solves the 2D heat equation using the finite difference method
    """
    N = int((xmax - xmin) // h) + 1  # Number of points on the sheet
    xs = np.linspace(xmin, xmax, N + 1)  # points on the x-axis
    ys = np.linspace(ymin, ymax, N + 1)  # points on the y-axis

    Nt = int(T // ht) + 1  # Number of time steps
    ts = np.linspace(0, T, Nt + 1)  # Instances in time

    # Initial conditions
    us = np.array([[uT0(x, y) for y in ys] for x in xs])

    # Initialize the tri-diagonal matrix A
    A = get_tridiag_matrix(N + 1, 1, -2, 1)

    # The values of u generated by the PDE at different times
    result = [us]

    # Solve the PDE for each time step
    for t in ts[1:]:
        # Calculate the values of f at different points
        fMat = np.array([[f(x, y, t, xc, yc) for y in ys] for x in xs])

        # Calculate the derivative of u
        du = ((mu / (h**2)) * ((A @ us) + (us @ A))) + fMat

        # Update the values of u
        us = us + ht * du

        # Update the values at the boundary points
        for i in range(N + 1):
            us[i][0] = uB(t)
            us[i][-1] = uB(t)

        for j in range(N + 1):
            us[0][j] = uB(t)
            us[-1][j] = uB(t)

        result.append(us)

    return result, xs, ys, ts


def plot2DHeatEquationAnimation(
    usVals, xs, ys, xmin, xmax, ymin, ymax, mu, show_graph=False
):
    """
    Plots the animation of the 2D Heat Equation
    """
    # Figure and Axes of the plot
    fig = plt.figure()

    if not show_graph:
        ax = plt.axes()
    else:
        ax = plt.axes(projection="3d")

    # What should be plotted and updated at each animation frame
    patches = []

    # Plot the initial condition
    if not show_graph:
        plt1 = plt.imshow(
            usVals[-1],
            cmap="hot",
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            aspect=xmax,
            animated=True,
        )
        cb = fig.colorbar(plt1)  # Add colorbar
        cb.set_label("Temperature")  # Add label to colorbar
        patches.append(plt1)
    else:
        X, Y = np.meshgrid(xs, ys)
        ax.plot_surface(
            X,
            Y,
            usVals[0],
            cmap="hot",
            linewidth=0,
            antialiased=False,
        )

    def init():
        """
        Initialization function for the animation.
        """
        # Title for the plot
        ax.set_title(
            f"Heat conduction in a sheet\n with boundary [{xmin}, {xmax}]??[{ymin}, {ymax}] and ?? = {mu}"
        )

        if not show_graph:
            # Removing the ticks
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        # Return everything that must be plotted at the start of the animation
        return patches

    def animate(i):
        """
        Update the animation at frame i
        """
        # Update the plot
        if not show_graph:
            plt1.set_array(usVals[i])
        else:
            ax.plot_surface(
                X,
                Y,
                usVals[i],
                cmap="hot",
                linewidth=0,
                antialiased=False,
            )

        # Return everything that must be updated
        return patches

    numFrames = len(usVals)  # Number of frames in the animation
    interval = 1  # Interval between frames in milliseconds

    # Setting up the animation
    anim = FuncAnimation(
        fig,
        func=animate,
        frames=numFrames,
        init_func=init,
        repeat=False,
        interval=interval,
    )

    # Display the animation
    plt.show()

    return anim


if __name__ == "__main__":
    # Required values
    in_uT0 = lambda x, y: 0
    in_uB = lambda t: 0
    in_f = lambda x, y, t, xc, yc: np.exp(-np.sqrt(((x - xc) ** 2) + ((y - yc) ** 2)))
    in_mu = 5 * (10 ** (-5))
    in_T, in_h, in_ht = 2000, 0.01, 0.5
    in_xmin, in_xmax, in_ymin, in_ymax = 0, 1, 0, 1
    in_xc, in_yc = 0.5, 0.5

    # Solving the Heat Equation
    uVals, xVals, yVals, tVals = solve2DHeatEquation(
        uT0=in_uT0,
        uB=in_uB,
        f=in_f,
        mu=in_mu,
        T=in_T,
        h=in_h,
        ht=in_ht,
        xmin=in_xmin,
        xmax=in_xmax,
        ymin=in_ymin,
        ymax=in_ymax,
        xc=in_xc,
        yc=in_yc,
    )

    # Plotting the animation
    ani = plot2DHeatEquationAnimation(
        usVals=uVals,
        xs=xVals,
        ys=yVals,
        xmin=in_xmin,
        xmax=in_xmax,
        ymin=in_ymin,
        ymax=in_ymax,
        mu=in_mu,
        # show_graph=False,
        show_graph=True,
    )
