"""
111901030
Mayank Singla
Coding Assignment 4 - Q1
"""

# %%
from math import sin, cos
from numpy import linspace
import matplotlib.pyplot as plt


def sin_x_2(x):
    """
    Returns the value of the function sin(x²) at x
    """
    return sin(x * x)


def sin_x_2_derivative(x):
    """
    Returns the derivative of the function sin(x²) at x
    Derivative is: 2 * x * cos(x²)
    """
    return 2 * x * cos(x * x)


def visualize(func, func_derivative, func_str, h, x_min, x_max):
    """
    Visualize the actual derivative (f'(x)) and forward finite difference approximation of the input function in the input interval
    """

    def forward_finite_difference(x):
        """
        Returns the forward finite difference approximation of the input function at x
        """
        return (func(x + h) - func(x)) / h

    # Number of points to plot
    numPoints = 1000

    # x-points generated uniformly between the interval
    xpts = linspace(x_min, x_max, numPoints)

    # The actual derivative of the input function at x-points
    ypts_actual = [func_derivative(x) for x in xpts]

    # The forward finite difference approximation of the input function at x-points
    ypts_approx = [forward_finite_difference(x) for x in xpts]

    # Plot the curves

    # Giving title and labels to the plot
    plt.title(
        f"Visualization of actual derivative f'(x) and\nforward finite difference approximation δ⁺ of {func_str}"
    )
    plt.xlabel("x")
    plt.ylabel("f'(x) and δ⁺")

    # Plotting the actual derivative
    plt.plot(xpts, ypts_actual, c="r", label="f'(x)")

    # Plotting the forward finite difference approximation
    plt.plot(xpts, ypts_approx, c="b", label="δ⁺")

    # Displaying the curve
    plt.grid()
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    visualize(sin_x_2, sin_x_2_derivative, "sin(x²)", 0.01, 0, 1)
