"""
111901030
Mayank Singla
Coding Assignment 4 - Q3
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
    Derivative is: 2xcos(x²)
    """
    return 2 * x * cos(x * x)


def sin_x_2_double_derivative(x):
    """
    Returns the double derivative of the function sin(x²) at x
    Double Derivative is: 2(cos(x²) - 2x²sin(x²))
    """
    return 2 * (cos(x * x) - 2 * x * x * sin(x * x))


def sin_x_2_triple_derivative(x):
    """
    Returns the triple derivative of the function sin(x²) at x
    Triple Derivative is: -4x(3sin(x²) + 2x²cos(x²))
    """
    return -4 * x * (3 * sin(x * x) + 2 * x * x * cos(x * x))


def visualize(
    func,
    func_derivative,
    func_double_derivative,
    func_triple_derivative,
    func_str,
    x_min,
    x_max,
):
    """
    Visualize as a function of h, the maximum absolute error of approximations of δ⁺ and δᶜ
    Also plot the maximum absolute error of approximations of the derivatives δ⁺ and δᶜ
    """

    def ffd(x, h):
        """
        Returns the forward finite difference approximation of the input function at x
        """
        return (func(x + h) - func(x)) / h

    def cfd(x, h):
        """
        Returns the centered finite difference approximation of the input function at x
        """
        return (func(x + h) - func(x - h)) / (2 * h)

    # Number of points to generate
    numPoints = 100

    # Generate h values
    hpts = linspace(x_min, x_max, numPoints)
    hpts = list(filter(lambda x: x != 0, hpts))  # Removing 0 h value

    # Actual absolute error for forward and centered finite difference approximation
    ypts_fa, ypts_ca = [], []

    # Theoretical Absolute error for forward and centered finite difference approximation
    ypts_ft, ypts_ct = [], []

    # Evaluating above values
    for h in hpts:
        max_fa, max_ft = 0, 0
        max_ca, max_ct = 0, 0
        for x in linspace(x_min, x_max, numPoints):
            max_fa = max(max_fa, abs(ffd(x, h) - func_derivative(x)))
            max_ca = max(max_ca, abs(cfd(x, h) - func_derivative(x)))

            max_dd = 0
            max_td = 0
            # Finding maximum value of derivative in the interval [x, x+h]
            for y in linspace(x, x + h, numPoints):
                max_dd = max(max_dd, abs(func_double_derivative(y)))
                max_td = max(max_td, abs(func_triple_derivative(y)))

            max_ft = max(max_ft, (h / 2) * max_dd)
            max_ct = max(max_ct, (h * h / 6) * max_td)

        # Appending the computed value for each h
        ypts_fa.append(max_fa)
        ypts_ft.append(max_ft)

        ypts_ca.append(max_ca)
        ypts_ct.append(max_ct)

    # Plot the curves

    # Giving title and labels to the plot
    plt.title(
        f"Visualization of absolute errors of approximation of\nδ⁺(x), δ⁻(x) and δᶜ(x) of {func_str}"
    )
    plt.xlabel("h")
    plt.ylabel("Maximum absolute error")

    # Plotting the maximum absolute error of approximation for forward finite difference approximation
    plt.plot(hpts, ypts_fa, c="r", label="Forward approximation")

    # Plotting the maximum theoretical absolute error of approximation for forward finite difference approximation
    plt.plot(hpts, ypts_ft, c="b", label="Theretical forward approximation")

    # Plotting the maximum absolute error of approximation for centered finite difference approximation
    plt.plot(hpts, ypts_ca, c="g", label="Centered approximation")

    # Plotting the maximum theoretical absolute error of approximation for centered finite difference approximation
    plt.plot(hpts, ypts_ct, c="y", label="Theoretical centered approximation")

    # Displaying the curve
    plt.grid()
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    visualize(
        sin_x_2,
        sin_x_2_derivative,
        sin_x_2_double_derivative,
        sin_x_2_triple_derivative,
        "sin(x²)",
        0,
        1,
    )
