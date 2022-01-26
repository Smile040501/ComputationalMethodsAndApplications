"""
111901030
Mayank Singla
Coding Assignment 1 - Q1
"""

# %%
import matplotlib.pyplot as plt
from math import log, pi as PI
from typing import List


def logStirling(n: int) -> float:
    """
    stirling(n) = sqrt(2πn) * (n \ e)ⁿ

    Args:
        n: An integer value

    Returns:
        stirling(n)
    """
    return ((1 / 2) * (log(2 * PI) + log(n))) + (n * (log(n) - 1))


def preprocess(N: int):
    """
    lhs = [Σlog(i)] where i ∈ [1, N]\n
    rhs = [log(stirling(i))] where i ∈ [1, N]\n
    diff = lhs - rhs\n

    Args:
        N: An integer value

    Returns:
        List of lhs, rhs and diff
    """
    logSum = 0  # Σlog(i)
    lhs = []
    rhs = []
    diff = []
    for i in range(1, N + 1):  # Making the lhs, rhs, and diff lists
        logSum += log(i)
        lhs.append(logSum)
        rVal = logStirling(i)
        rhs.append(rVal)
        diff.append(logSum - rVal)
    return [lhs, rhs, diff]


def main():
    """
    The main function to execute.
    Visualize the Stirling's approximation.
    stirling(n) = sqrt(2πn) * (n \ e)ⁿ
    """
    N = 1000000
    # Getting the LHS, RHS and the diff lists
    [lhs, rhs, diff] = preprocess(N)

    # Generating points from [1, N] for x-axis points
    xpoints = list(range(1, N + 1))
    # Font style for the plots
    font = {"family": "serif", "size": 13}  # Font Properties for the title and labels

    # 1. Plotting (xpoints vs lhs) and (xpoints vs rhs)

    # Giving Title and Labels to the plot
    plt.title("Visualizing Stirling's approximation", fontdict=font)
    plt.xlabel("n", fontdict=font)
    plt.ylabel("log(LHS)  and  log(RHS)", fontdict=font)

    # Plotting n vs log(n!)
    plt.plot(xpoints, lhs, "g", lw="3", label="log(n!)")

    # Plotting n vs log(stirling(n))
    plt.plot(xpoints, rhs, "r", lw="1.5", label="log(stirling(n))")

    # Displaying the plot
    plt.legend()
    plt.show()

    # 2. Plotting `diff` points on the x-axis

    # Giving Title and Labels to the plot
    plt.title("Visualizing Stirling's approximation", fontdict=font)
    plt.xlabel("log(LHS) - log(RHS)", fontdict=font)

    # Plotting log(n!) - log(stirling(n))
    plt.plot(diff, [0] * N, "-ob", lw="3", label="log(n!) - log(stirling(n))")
    plt.grid()

    # Displaying the plot
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
