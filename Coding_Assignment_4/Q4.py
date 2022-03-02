"""
111901030
Mayank Singla
Coding Assignment 4 - Q4
"""

# %%
from math import exp
import matplotlib.pyplot as plt


def inp_func(x):
    """
    Returns the value of the function 2 * x * e^(x²) at x
    """
    return 2 * x * exp(x * x)


def inp_func_integral(x):
    """
    Returns the value of the integral of the function 2 * x * e^(x²) at x
    Integral is: e^(x²)
    """
    return exp(x * x)


def visualize(func, func_integral, func_str, a, b):
    """
    Visualize as a function of M (number of intervals), area under the curve of input function computed using trapezoidal formula.
    Also compute the exact area
    """

    numIntervals = 100  # Number of intervals

    xpts = []  # x-points
    ypts = []  # y-points

    actual_area = func_integral(b) - func_integral(a)  # Exact area

    # For each number of intervals, computing the area
    for M in range(1, numIntervals + 1):
        xpts.append(M)
        # Trapezoidal formula
        H = (b - a) / M
        area = ((b - a) * (func(a) + func(b))) / (2 * M)
        for k in range(1, M):
            xk = a + (k * H)
            area += (b - a) * func(xk) / M

        ypts.append(area)

    # Plotting the curve

    # Giving labels and title to the curve
    plt.title(
        f"Visualize as a function of M (number of intervals), area under\n the curve of {func_str} computed using trapezoidal formula."
    )
    plt.xlabel("M")
    plt.ylabel("area")

    # Plotting the value of area computed using trapezoidal formula
    plt.plot(xpts, ypts, c="r", label="Approximate Area")

    # Plotting the value of actual area as a horizontal line
    plt.axhline(y=actual_area, color="b", label="Exact Area")

    # Displaying the curve
    plt.grid()
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    visualize(inp_func, inp_func_integral, "2xe^(x²)", 1, 3)
