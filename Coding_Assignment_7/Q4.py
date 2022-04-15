"""
111901030
Mayank Singla
Coding Assignment 7 - Q4
"""

# %%
# Below lines are to ignore the pylint warning in VSCode
# pylint: disable=abstract-method
# pylint: disable=pointless-string-statement
# pylint: disable=invalid-name
import math
import matplotlib.pyplot as plt


def newton_raphson_method(fun, derivative, x0, K):
    """
    Approximates the root of the function using the Newton-Raphson Method
    """
    xvals = [x0]  # Values of x obtained in iterations
    x = x0
    # Applying the method for K iterations
    for _ in range(K):
        x = x - (fun(x) / derivative(x))
        xvals.append(x)
    return xvals


def secant_method(fun, x0, x1, K):
    """
    Approximates the root of the function using the Secant Method
    """
    xvals = [x0, x1]  # Values of x obtained in iterations
    x = x1
    # Applying the method for K iterations
    for _ in range(K):
        x = x - fun(x) * ((x - xvals[-2]) / (fun(x) - fun(xvals[-2])))
        xvals.append(x)
    return xvals


def get_rate_of_convergence(xvals):
    """
    Plots the rate of convergence of the sequence of points
                α = (Log |(xₙ₊₁ - xₙ) / (xₙ - xₙ₋₁)|) / (Log |(xₙ - xₙ₋₁) / (xₙ₋₁ - xₙ₋₂)|)
    """
    # Calculating the rate of convergence
    alpha = []
    for i in range(2, len(xvals) - 1):
        alpha.append(
            math.log(abs((xvals[i + 1] - xvals[i]) / (xvals[i] - xvals[i - 1])))
            / math.log(abs((xvals[i] - xvals[i - 1]) / (xvals[i - 1] - xvals[i - 2])))
        )
    return alpha


def inp_fun(x):
    """
    The input function
    """
    return x * math.exp(x)


def inp_fun_derivative(x):
    """
    The derivative of the input function
    """
    return x * math.exp(x) + math.exp(x)


if __name__ == "__main__":
    # Testing the function
    numIter = 212
    x0Init = 200
    x1Init = 201
    # Applying the Newton-Raphson Method
    xNR = newton_raphson_method(inp_fun, inp_fun_derivative, x0Init, numIter)
    # Applying the Secant Method
    xS = secant_method(inp_fun, x0Init, x1Init, numIter - 1)

    # Getting the rate of convergence of the sequence of points
    alphaNR = get_rate_of_convergence(xNR)
    alphaS = get_rate_of_convergence(xS)

    # Plotting the rate of convergence
    plt.title("Rate of Convergence")
    plt.ylabel("α")
    plt.xlabel("Iteration")
    plt.plot(list(range(2, len(alphaNR) + 2)), alphaNR, label="Newton-Raphson Method")
    plt.plot(list(range(2, len(alphaS) + 2)), alphaS, label="Secant Method")
    plt.legend()
    plt.grid()
    plt.show()
