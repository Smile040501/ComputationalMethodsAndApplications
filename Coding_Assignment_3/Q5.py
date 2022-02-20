"""
111901030
Mayank Singla
Coding Assignment 3 - Q5
"""

# %%
# Uncomment above line: This line is required to make the animation work in VSCode (Using `ipympl` as the backend for matplotlib plots)
# %matplotlib ipympl
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline, Akima1DInterpolator, BarycentricInterpolator
import numpy as np


def animation(evalFunc, funcString, xlims, ylims, numFrames, interval):
    """
    Function to animate different interpolations of the given function.\n
    The return value of it must be captured in a variable so as to render the animation without it being deleted.
    """
    x_min, x_max = xlims
    y_min, y_max = ylims

    # Figure and Axes of the plot
    fig, ax = plt.subplots()

    # Plotting the True Value of the Function
    numTruePoints = 1000
    trueXValues = np.linspace(x_min, x_max, numTruePoints)
    trueYValues = [evalFunc(x) for x in trueXValues]
    plt.plot(trueXValues, trueYValues, c="blue", label="True")

    # Curves to plot
    (cubicSpline,) = plt.plot([], [], c="red", label="Cubic spline")
    (akima,) = plt.plot([], [], c="green", label="Akima")
    (barycentric,) = plt.plot([], [], c="purple", label="Barycentric")

    # Curves to animate
    patches = [cubicSpline, akima, barycentric]

    def init():
        """
        Initialization function for the animation.
        """
        # Initializes the plot
        ax.set_title(f"Different interpolations of {funcString}")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_ylim(y_min, y_max)

        # Return everything that must be plotted at the start of the animation
        return patches

    def animate(frame):
        """
        Animation function for the animation.
        """
        # Updating the title
        ax.set_title(f"Different interpolations of {funcString} for {frame} samples")

        # Genearting random samples of points
        xpts = sorted(list(np.random.rand(frame)))
        ypts = [evalFunc(x) for x in xpts]

        # Updating the curves using different interpolations
        cubicSpline.set_data(trueXValues, CubicSpline(xpts, ypts)(trueXValues))
        akima.set_data(trueXValues, Akima1DInterpolator(xpts, ypts)(trueXValues))
        barycentric.set_data(
            trueXValues, BarycentricInterpolator(xpts, ypts)(trueXValues)
        )

        # Return everything that must be updated
        return patches

    # Setting up the animation
    anim = FuncAnimation(
        fig, animate, init_func=init, frames=numFrames, interval=interval, blit=True
    )

    plt.grid()
    plt.legend(loc="upper left")
    plt.show()

    # Retruning the animation
    return anim


# Sample Test Function
def evalFunction(x):
    """
    Returns the value of tan(x) ⋅ sin(30x) ⋅ eˣ
    """
    return (np.tan(x) * np.sin(30 * x)) * np.exp(x)


if __name__ == "__main__":
    # Animation of the sample function
    anim = animation(
        evalFunc=evalFunction,
        funcString="tan(x)⋅sin(30x)⋅eˣ",
        xlims=(0, 1),
        ylims=(-4, 4),
        numFrames=100,
        interval=500,
    )
