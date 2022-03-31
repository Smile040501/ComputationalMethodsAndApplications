"""
111901030
Mayank Singla
Coding Assignment 6 - Q1
"""

# Below lines are to ignore the pylint warning in VSCode
# pylint: disable=abstract-method
# pylint: disable=pointless-string-statement
# pylint: disable=invalid-name

# %%
from math import exp
import matplotlib.pyplot as plt
from numpy import linspace


def solveODE(func, ode, x0, t0, T, stepSizes):
    """
    Uses the forward Euler method to solve the given ODE
    """

    def get_points(s):
        """
        Returns the points in [t0, T] with step size s
        """
        ans = []
        curr = t0
        while curr < T:
            ans.append(curr)
            curr += s
        ans.append(T)
        return ans

    # For each step size in the input list, applyting the forward Euler method to evaluate points
    for h in stepSizes:
        tnpts = get_points(h)  # Generating points with the given step size
        xnpts = [x0]  # Initializing the list of points

        # Applying the formula
        for i in range(0, len(tnpts) - 1):
            xnpts.append(xnpts[i] + h * ode(tnpts[i], xnpts[i]))

        # Plotting the points for each stepsize
        plt.plot(tnpts, xnpts, label=f"h = {h}")

    # Title and labels for the curve
    plt.title("Forward Euler Method x'(t) = -2x(t)")
    plt.xlabel("t")
    plt.ylabel("x(t)")

    plt.ylim(-5, 5)  # Setting the y-axis limits

    # Plotting the actual function
    xpts = linspace(t0, T, 100)
    ypts = [func(x) for x in xpts]
    plt.plot(xpts, ypts, label="Actual Function")

    # Displaying the curve
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # The actual function by solving the ODE x'(t) = -2x(t) and x(0) = 5
    inp_func = lambda t: 5 * exp(-2 * t)

    # The given ODE
    inp_ode = lambda t, x: -2 * x

    # Testing the function
    solveODE(
        func=inp_func, ode=inp_ode, x0=5, t0=0, T=10, stepSizes=[0.1, 0.5, 1, 2, 3]
    )
