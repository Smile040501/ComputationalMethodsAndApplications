"""
111901030
Mayank Singla
Coding Assignment 1 - Q3
"""

# %%
import matplotlib.pyplot as plt
from random import uniform
from math import pi as PI  # To plot y = math.pi


def isInsideCircle(x0, y0, r, x1, y1):
    """
    Returns boolean indicating whether a point (x1, y1) lies inside the circle centered at (x0, y0) with radius r
    """
    return ((x0 - x1) ** 2) + ((y0 - y1) ** 2) <= (r ** 2)


def estimatePi(n: int):
    """
    Estimates π using the Monte Carlo method.
    Takes as argument a positive integer n that denotes the total number of points generated in the simulation.
    """
    x0, y0, a = 0, 0, 1  # Center of the square, and side length of the square
    pointsGeneratedList = []  # The number of points generated so far
    fraction4List = []  # (4 * fraction of points within the circle) so far
    numPointsInCircle = 0  # Number of points within the circle so far
    for i in range(n):  # Generating a random point for n time
        # x and y are uniformly and independently sampled within the range of square limits
        x = uniform(x0 - (a / 2), x0 + (a / 2))
        y = uniform(y0 - (a / 2), y0 + (a / 2))
        if isInsideCircle(x0, y0, a / 2, x, y):
            numPointsInCircle += 1
        numPointsGenerated = i + 1
        # (4 * fraction of points within the circle)
        fraction4 = 4 * (numPointsInCircle / numPointsGenerated)
        # Appending the required quantities to the list
        pointsGeneratedList.append(numPointsGenerated)
        fraction4List.append(fraction4)

    # Generating the plot
    # Giving title and labels to the plot
    plt.title("Estimates π using Monte Carlo Method")
    plt.xlabel("No. of points generated")
    plt.ylabel("4 x fraction of points within the circle")
    # Setting y-limits for the plot
    plt.ylim([3.10, 3.20])
    # Plotting the value of math.pi as a horizontal line
    plt.axhline(y=PI, color="r", label="Value of math.pi")
    # To show x10^() for the scaling of the axis
    plt.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=True)
    # Generating grid lines
    plt.grid(linestyle="--", linewidth=1.25)
    # Plotting the curve of Monte Carlo Method
    plt.plot(
        pointsGeneratedList, fraction4List, color="#1f77b4", label="Monte Carlo Method"
    )
    # Displaying the legend box
    plt.legend(loc="lower right")
    # Displaying the plot
    plt.show()


if __name__ == "__main__":
    estimatePi(2000000)
