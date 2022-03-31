"""
111901030
Mayank Singla
Coding Assignment 6 - Q4
"""

# %%
import matplotlib.pyplot as plt
from numpy import linspace
from scipy.integrate import solve_ivp


# Below lines are to ignore the pylint warning in VSCode
# pylint: disable=abstract-method
# pylint: disable=pointless-string-statement
# pylint: disable=invalid-name


def solveODE(x0, v0, mu, t0, T, n):
    """
    Uses the forward Euler method to solve the given ODE and evaluate the time period of the ODE.
    """

    def vdp_derivatives(t, y):
        """
        Returns the derivatives of the system of differential equations
        """
        x, v = y
        return [v, mu * (1 - x * x) * v - x]

    # Time values for the plot at which we will evaluate points
    t = linspace(t0, T, n)

    # Solving the system of ODEs
    sol = solve_ivp(fun=vdp_derivatives, t_span=[t0, T], y0=[x0, v0], t_eval=t)

    # Values of points on the curve
    xpts = sol.y[0]

    # Plotting the curve
    plt.title(f"Van der Pol equation for μ = {mu}")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.plot(t, xpts)
    plt.grid()
    plt.show()

    # Evaluating the Time Period
    i1 = 0
    for i in range(n - 1, 0, -1):
        if xpts[i] <= 0 and xpts[i - 1] >= 0:
            # We obtained the first crossing of the curve
            i1 = i
            break

    i2 = -1
    for i in range(i1 - 1, 0, -1):
        if xpts[i] <= 0 and xpts[i - 1] >= 0:
            # We obtained the second crossing of the curve
            i2 = i
            break

    # Time period of the curve will be the difference in time values at the two crossing points
    timePeriod = abs(t[i1] - t[i2])
    print(f"The time period of the curve for μ = {mu} is {timePeriod:.2f}")


if __name__ == "__main__":
    # Testing the function
    solveODE(x0=0, v0=10, mu=0, t0=0, T=200, n=10000)
