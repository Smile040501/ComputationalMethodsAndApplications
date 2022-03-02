"""
111901030
Mayank Singla
Coding Assignment 4 - Q2
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
    Visualize the absolute errors of approximation of δ⁺, δ⁻, and δᶜ
    """

    def forward_finite_difference(x):
        """
        Returns the forward finite difference approximation of the input function at x
        """
        return (func(x + h) - func(x)) / h

    def backward_finite_difference(x):
        """
        Returns the backward finite difference approximation of the input function at x
        """
        return (func(x) - func(x - h)) / h

    def centered_finite_difference(x):
        """
        Returns the centered finite difference approximation of the input function at x
        """
        return (func(x + h) - func(x - h)) / (2 * h)

    # Number of points to plot
    numPoints = 1000

    # x-points generated uniformly between the interval
    xpts = linspace(x_min, x_max, numPoints)

    # The forward finite difference approximation of the input function at x-points
    ypts_forward = [
        abs(forward_finite_difference(x) - func_derivative(x)) for x in xpts
    ]

    # The backward finite difference approximation of the input function at x-points
    ypts_backward = [
        abs(backward_finite_difference(x) - func_derivative(x)) for x in xpts
    ]

    # The centered finite difference approximation of the input function at x-points
    ypts_centered = [
        abs(centered_finite_difference(x) - func_derivative(x)) for x in xpts
    ]

    # Plot the curves

    # Giving title and labels to the plot
    plt.title(
        f"Visualization of absolute errors of approximation\nof δ⁺(x), δ⁻(x) and δᶜ(x) of {func_str}"
    )
    plt.xlabel("x")
    plt.ylabel("Absolute error")

    # Plotting the absolute errors of approximation for forward finite difference approximation
    plt.plot(xpts, ypts_forward, c="r", label="|δ⁺ - f'(x)|")

    # Plotting the absolute errors of approximation for backward finite difference approximation
    plt.plot(xpts, ypts_backward, c="b", label="|δ⁻ - f'(x)|")

    # Plotting the absolute errors of approximation for centered finite difference approximation
    plt.plot(xpts, ypts_centered, c="g", label="|δᶜ - f'(x)|")

    # Displaying the curve
    plt.grid()
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    visualize(sin_x_2, sin_x_2_derivative, "sin(x²)", 0.01, 0, 1)
