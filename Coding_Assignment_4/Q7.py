"""
111901030
Mayank Singla
Coding Assignment 4 - Q7
"""

# %%
from math import exp, sin, cos
import matplotlib.pyplot as plt
from numpy import linspace, linalg


def handleError(method):
    """
    Decorator Factory function.
    Returns a decorator that normally calls the method of a class by forwarding all its arguments to the method.
    It surrounds the method calling in try-except block to handle errors gracefully.
    """

    def decorator(ref, *args, **kwargs):
        """
        Decorator function that surrounds the method of a class in try-except block and call the methods and handles error gracefully.
        """
        try:
            # Return the same value as that of the method if any
            return method(ref, *args, **kwargs)
        except Exception as err:
            print(type(err))
            print(err)

    return decorator


def getSuperScript(inp):
    """
    Returns the superscript notation of the input number/string.
    """
    normal = "0123456789"
    super_s = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    # Building the translation map
    trans = str.maketrans("".join(normal), "".join(super_s))
    # Returning the converted string
    return str(inp).translate(trans)


class Polynomial:
    """
    Represents an algebraic polynomial
    """

    @handleError
    def _validateCoefficients(self, coff):
        """
        Validates the list of coefficients
        Returns True if correct.
        """
        if not isinstance(coff, list):
            raise Exception("Invalid input - Expected list")

        for i in coff:
            if not isinstance(i, (int, float)):
                raise Exception(
                    f"Invalid type of coefficient received {type(i)}.\nExpected float or int."
                )

        return True

    @handleError
    def __init__(self, coff):
        """
        Initializes the polynomial with the list of coefficients.
        """
        if not self._validateCoefficients(coff):
            return
        self.degree = max(0, len(coff) - 1)
        self.coff = coff

    @handleError
    def __str__(self):
        """
        Returns a string representation of the polynomial.
        """
        ans = "Coefficients of the polynomial are:\n"
        if len(self.coff) == 0:
            ans += "0"
        else:
            ans += " ".join(str(i) for i in self.coff)
        return ans

    @handleError
    def _addOrSub(self, p, isAdd=True):
        """
        Adds or subtracts the polynomial with the polynomial passed as argument.
        """
        # Raising the exception if the input is not a Polynomial object
        if not isinstance(p, Polynomial):
            raise Exception("Invalid input - Expected Polynomial")

        # Evaluating the summation of the two polynomials
        ansCoff = []
        n, m = self.degree + 1, p.degree + 1
        for i in range(max(n, m)):
            sumCoff = 0
            if i < n:
                sumCoff += self.coff[i]
            if i < m:
                sumCoff += p.coff[i] if isAdd else (-p.coff[i])
            ansCoff.append(sumCoff)
        return Polynomial(ansCoff)

    @handleError
    def __add__(self, p):
        """
        Overloading the + operator for the polynomial class
        """
        return self._addOrSub(p, isAdd=True)

    @handleError
    def __sub__(self, p):
        """
        Overloading the - operator for the polynomial class
        """
        return self._addOrSub(p, isAdd=False)

    @handleError
    def __mul__(self, p):
        """
        Overloading the * operator for the polynomial class to multiply it with a polynomial
        """
        # Raising the exception if the input is not a Polynomial object
        if not isinstance(p, Polynomial):
            raise Exception("Invalid input - Expected Polynomial")

        # Evaluating the multiplication of the two polynomials
        ansCoff = dict()
        n, m = self.degree + 1, p.degree + 1
        for i in range(n):
            for j in range(m):
                ansCoff[i + j] = ansCoff.get(i + j, 0) + self.coff[i] * p.coff[j]

        ansCoff = [ansCoff[i] for i in sorted(ansCoff.keys())]
        return Polynomial(ansCoff)

    @handleError
    def __rmul__(self, scalar):
        """
        Overloading the * operator for the polynomial class to pre-multiply it with a scalar
        """
        if not isinstance(scalar, (int, float)):
            raise Exception("Invalid input - Expected scalar")
        return Polynomial([scalar * i for i in self.coff])

    @handleError
    def __getitem__(self, x):
        """
        Evaluate the polynomial at the given real number `x`.
        """
        ans = 0
        for i in range(self.degree + 1):
            ans += self.coff[i] * (x**i)
        return ans

    @handleError
    def _getPolyString(self):
        """
        Returns the actual string representation of the polynomial.
        """
        ans = ""
        for i in range(self.degree + 1):
            if self.coff[i] != 0:
                if i == 0:
                    ans += (
                        f"{self.coff[i]:.2f}"
                        if isinstance(self.coff[i], float)
                        else f"{self.coff[i]}"
                    )
                else:
                    ans += " + " if self.coff[i] > 0 else " - "
                    if abs(self.coff[i]) != 1:
                        ans += (
                            f"{abs(self.coff[i]):.2f}"
                            if isinstance(self.coff[i], float)
                            else f"{abs(self.coff[i])}"
                        )
                    ans += f"x{getSuperScript(i)}" if i != 1 else "x"
        return ans

    @handleError
    def _plotPolynomial(self, a, b):
        """
        Plots the polynomial in the given interval [a, b]
        """
        a = min(a, b)
        b = max(a, b)
        numPoints = 100
        xpoints = list(linspace(a, b, numPoints))
        ypoints = [self[i] for i in xpoints]
        plt.plot(xpoints, ypoints, c="b")
        plt.grid()

    @handleError
    def show(self, a, b):
        """
        Visualize the polynomial in the given interval [a, b]
        """
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise Exception("Invalid input - Expected scalar")
        plt.title(f"Plot of the polynomial {self._getPolyString()}")
        plt.xlabel("x")
        plt.ylabel("P(x)")
        self._plotPolynomial(a, b)
        plt.show()

    @handleError
    def fitViaMatrixMethod(self, points):
        """
        Using the idea of linear systems, it fits a polynomial to the points passed as its argument.
        Displays a plot with the given points and the computed polynomial.
        """
        # Raising the exception if the input is not valid
        if not isinstance(points, list):
            raise Exception("Invalid input - Expected list of tuples")
        for i in points:
            if not isinstance(i, tuple) or len(i) != 2:
                raise Exception("Invalid input - Expected list of tuples")
            if not isinstance(i[0], (int, float)) or not isinstance(i[1], (int, float)):
                raise Exception(
                    "Invalid input - Expected list of tuples representing points"
                )

        # Creating the matrix and the vector b
        degree = len(points) - 1
        A, b = [], []
        minX, maxX = 0, 0
        xpoints, ypoints = [], []
        for p in points:
            A.append([p[0] ** j for j in range(degree + 1)])
            b.append(p[1])
            minX = min(minX, p[0])
            maxX = max(maxX, p[0])
            xpoints.append(p[0])
            ypoints.append(p[1])

        # Solving the linear system
        x = list(linalg.solve(A, b))
        ans = Polynomial(x)

        # Plotting the given points and the computed polynomial
        plt.title(
            f"Polynomial interpolation using matrix method\nComputed Polynomial {ans._getPolyString()}"
        )
        plt.xlabel("x")
        plt.ylabel("f̃(x)")
        plt.plot(xpoints, ypoints, "ro")
        ans._plotPolynomial(minX, maxX)
        plt.show()

    @handleError
    def fitViaLagrangePoly(self, points):
        """
        Computes the Lagrange polynomial for the points passed as argument to this method.
        Displays a plot with the given points and the computed polynomial.
        """
        # Raising the exception if the input is not valid
        if not isinstance(points, list):
            raise Exception("Invalid input - Expected list of tuples")
        for i in points:
            if not isinstance(i, tuple) or len(i) != 2:
                raise Exception("Invalid input - Expected list of tuples")
            if not isinstance(i[0], (int, float)) or not isinstance(i[1], (int, float)):
                raise Exception(
                    "Invalid input - Expected list of tuples representing points"
                )

        # Calculating some required values
        degree = len(points) - 1
        minX, maxX = 0, 0
        xpoints, ypoints = [], []
        for p in points:
            minX = min(minX, p[0])
            maxX = max(maxX, p[0])
            xpoints.append(p[0])
            ypoints.append(p[1])

        # Evaluating the Lagrange polynomial
        ansP = Polynomial([0])
        for j in range(degree + 1):
            # Evaluating Ψⱼ
            numerator, denominator = Polynomial([1]), 1
            for i in range(degree + 1):
                if i == j:
                    continue
                numerator = numerator * Polynomial([-xpoints[i], 1])
                denominator *= xpoints[j] - xpoints[i]
            ansP = ansP + ((ypoints[j] / denominator) * numerator)

        # Plotting the given points and the computed polynomial
        plt.title(
            f"Interpolation using Lagrange polynomial\nComputed Polynomial {ansP._getPolyString()}"
        )
        plt.xlabel("x")
        plt.ylabel("f̃(x)")
        plt.plot(xpoints, ypoints, "ro")
        ansP._plotPolynomial(minX, maxX)
        plt.show()

    @handleError
    def derivative(self):
        """
        Returns the derivative of the polynomial
        """
        # Calculating the coefficients of the derivative polynomial
        result_coff = []
        for i in range(1, len(self.coff)):
            result_coff.append(i * self.coff[i])
        return Polynomial(result_coff)

    @handleError
    def integral(self):
        """
        Returns the integral of the polynomial
        """
        # Calculating the coefficients of the integral polynomial
        result_coff = [0]
        for i in range(len(self.coff)):
            result_coff.append(self.coff[i] / (i + 1))
        return Polynomial(result_coff)

    @handleError
    def area(self, a, b):
        """
        Returns the area under the polynomial in the interval [a, b]
        """
        # Checking input
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise Exception("Invalid input - Expected scalar")
        if a > b:
            raise Exception("Invalid input - Expected a <= b")

        # The integral of the polynomial
        int_p = self.integral()

        # Returning answer
        return int_p[b] - int_p[a]


def func_integral(x):
    """
    Returns the integral of the function eˣ.sin(x)
    """
    return exp(x) * (sin(x) - cos(x)) / 2


if __name__ == "__main__":
    """
    We will use the Taylor's Series expansion about a point x = a to get an approximate polynomial for the given curve eˣ.sin(x)
    f(x) = f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + f'''(a)(x-a)³/3! + ...

    We will put a = 0, to get the Maclaurin Series expansion of it

    f(x) = f(0) + f'(0)(x) + f''(0)(x)²/2! + f'''(0)(x)³/3! + ...
        = x + x² + x³/3 - x⁵/30 - x⁶/90 - x⁷/630 + x⁹/22680 + x¹⁰/113400 + O(x¹¹)
    """
    # Creating the polynomial
    fx = Polynomial(
        [0, 1, 1, 1 / 3, 0, -1 / 30, -1 / 90, -1 / 630, 0, 1 / 22680, 1 / 113400]
    )
    approx_area = fx.area(0, 1 / 2)
    actual_area = func_integral(1 / 2) - func_integral(0)
    error = abs(actual_area - approx_area)

    # Printing the results
    print("The computed area under the curve is:", approx_area)
    print("The actual area under the curve is:", actual_area)
    print("The error is:", error)

    # The error is indeed within a guaranteed range of 10⁻⁶
