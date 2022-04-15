"""
111901030
Mayank Singla
Coding Assignment 7 - Q5
"""

# %%
# Below lines are to ignore the pylint warning in VSCode
# pylint: disable=abstract-method
# pylint: disable=pointless-string-statement
# pylint: disable=invalid-name
import math
import matplotlib.pyplot as plt
import scipy.linalg as la


def newton_raphson_method(fun, fun_jacobi, x0, K):
    """
    Approximates the root of the function using the Newton-Raphson Method
    """
    xvals = [x0]  # Values of x obtained in iterations
    x = x0
    # Applying the method for K iterations
    for _ in range(K):
        x = x - la.inv(fun_jacobi(x)) @ fun(x)
        xvals.append(x)
    return xvals


def inp_fun(x):
    """
    The input function F(xₖ)
    """
    x1, x2, x3 = x
    f1 = 3 * x1 - math.cos(x2 * x3) - (3 / 2)
    f2 = 4 * (x1**2) - 625 * (x2**2) + 2 * x3 - 1
    f3 = 20 * x3 + math.exp(-1 * x1 * x2) + 9
    return [f1, f2, f3]


def inp_fun_jacobi(x):
    """
    The Jacobi matrix of the input function J(xₖ)
    """
    x1, x2, x3 = x
    j1 = [3, x3 * math.sin(x2 * x3), x2 * math.sin(x2 * x3)]
    j2 = [8 * x1, -1250 * x2, 2]
    j3 = [-x2 * math.exp(-1 * x1 * x2), -x1 * math.exp(-1 * x1 * x2), 20]
    return [j1, j2, j3]


if __name__ == "__main__":
    # Testing the function
    numIter = 20
    x0Init = [1, 2, 3]

    # Applying the Newton-Raphson Method
    xNR = newton_raphson_method(inp_fun, inp_fun_jacobi, x0Init, numIter)

    print(f"The root of the function is {xNR[-1]}")

    # Evaluating the norm of the function at the obtained points
    fNR = [la.norm(inp_fun(x)) for x in xNR]

    # Plotting the rate of convergence
    plt.title("||f(xₖ)|| vs Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("||f(xₖ)||")
    plt.plot(list(range(0, len(xNR))), fNR, label="Newton-Raphson Method")
    plt.legend()
    plt.grid()
    plt.show()
