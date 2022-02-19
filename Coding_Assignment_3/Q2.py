"""
111901030
Mayank Singla
Coding Assignment 3 - Q2
"""

# %%
from random import uniform
import numpy as np


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


class RowVectorFloat:
    """
    Represents a row vector of float values.
    """

    @handleError
    def _validateListValues(self, lst):
        """
        Validates the values in the list.
        Returns True if correct.
        """
        if not isinstance(lst, list):
            raise Exception("Invalid input - Expected list")

        for i in lst:
            if not isinstance(i, float) and not isinstance(i, int):
                raise Exception(
                    f"Invalid type of value received {type(i)}.\nExpected float or int."
                )

        return True

    @handleError
    def _validateIndex(self, index):
        """
        Validates the input index.
        Returns True if correct.
        """
        if not isinstance(index, int):
            raise Exception(
                f"Invalid type of index received {type(index)}.\nExpected int."
            )
        n = len(self.vec)
        if index >= n or index < (-n):
            raise Exception(f"Index out of range.")
        return True

    @handleError
    def __init__(self, lst):
        """
        Initializes the row vector with the values in the list.
        """
        if not self._validateListValues(lst):
            return

        # Creating a new copy of the list
        self.vec = list(lst)

    @handleError
    def __str__(self):
        """
        Returns the string representation of the row vector.
        """
        return " ".join(
            f"{i:.2f}" if isinstance(i, float) else str(i) for i in self.vec
        )

    @handleError
    def __len__(self):
        """
        Returns the length of the row vector.
        """
        return len(self.vec)

    @handleError
    def __getitem__(self, index):
        """
        Returns the value at the given index.
        """
        if not self._validateIndex(index):
            return
        return self.vec[index]

    @handleError
    def __setitem__(self, index, value):
        """
        Sets the value at the given index
        """
        if not self._validateIndex(index):
            return
        elif not self._validateListValues([value]):
            return
        self.vec[index] = value

    @handleError
    def __add__(self, rv):
        """
        Adds two row vectors.
        Operator looks for __add__ in left operand.
        """
        if not isinstance(rv, RowVectorFloat):
            raise Exception("Invalid input - Expected RowVectorFloat")
        elif len(self) != len(rv):
            raise Exception("Invalid input - Expected same length vectors")

        ans = [self.vec[i] + rv.vec[i] for i in range(len(self))]
        return RowVectorFloat(ans)

    @handleError
    def __radd__(self, rv):
        """
        Adds two row vectors.
        Operator looks for __add__ in right operand.
        """
        return self.__add__(rv)

    @handleError
    def __mul__(self, scalar):
        """
        Multiplies a row vector with a scalar.
        Operator looks for __mul__ in left operand.
        """
        if not isinstance(scalar, (int, float)):
            raise Exception("Invalid input - Expected scalar")
        ans = [
            self.vec[i] * scalar if self.vec[i] != 0 else 0.00 for i in range(len(self))
        ]
        return RowVectorFloat(ans)

    @handleError
    def __rmul__(self, scalar):
        """
        Multiplies a row vector with a scalar.
        Operator looks for __mul__ in right operand.
        """
        return self.__mul__(scalar)


class SquareMatrixFloat:
    """
    Represents a square matrix
    """

    @handleError
    def __init__(self, n):
        """
        Initializes the square matrix as a list of RowVectorFloat objects.
        """
        if not isinstance(n, int):
            raise Exception("Invalid input - Expected int")
        self.n = n
        self.mat = [RowVectorFloat([0] * n) for _ in range(n)]

    @handleError
    def __str__(self):
        """
        Returns the string representation of the square matrix.
        """
        ans = "The matrix is:\n"
        for i in range(self.n):
            ans += str(self.mat[i]) + "\n"
        return ans

    @handleError
    def sampleSymmetric(self):
        """
        Samples a random symmetric matrix.
        """
        for i in range(self.n):
            for j in range(i):
                self.mat[i][j] = uniform(0, 1)
                self.mat[j][i] = self.mat[i][j]
            self.mat[i][i] = uniform(0, self.n)

    @handleError
    def toRowEchelonForm(self):
        """
        Converts the matrix to its row echelon form.
        """
        pivotRow, pivotCol = 0, 0
        while pivotRow < self.n and pivotCol < self.n:
            # Find the row with first non-zero entry in the pivot column
            nonZeroRow = pivotRow
            while nonZeroRow < self.n and self.mat[nonZeroRow][pivotCol] == 0:
                nonZeroRow += 1

            # If no non-zero entry found, move to next pivot column
            if nonZeroRow == self.n:
                pivotCol += 1
                continue

            # Swap the pivot row with the nonZeroRow
            if nonZeroRow != pivotRow:
                self.mat[pivotRow], self.mat[nonZeroRow] = (
                    self.mat[nonZeroRow],
                    self.mat[pivotRow],
                )

            # Multiply each element in the pivot row by the inverse of the pivot, so the pivot equals 1.
            self.mat[pivotRow] = self.mat[pivotRow] * (1 / self.mat[pivotRow][pivotCol])
            # Make the pivot entry 1 (to avoid python precision issues)
            self.mat[pivotRow][pivotCol] = 1.00

            # Add multiples of the pivot row to each of the lower rows, so every element in the pivot column of the lower rows equals 0.
            for i in range(pivotRow + 1, self.n):
                if self.mat[i][pivotCol] != 0:
                    self.mat[i] = (
                        self.mat[i] + (-self.mat[i][pivotCol]) * self.mat[pivotRow]
                    )
                    self.mat[i][pivotCol] = 0.00  # To avoid python precision issues

            # Move to the next column and next row
            pivotCol += 1
            pivotRow += 1

        return self

    @handleError
    def isDRDominant(self):
        """
        Checks if the matrix is Diagonally row dominant.
        Here, I am checking strictly for DRDominant though it was not mentioned in the question, but it is required for checking the correctness of Jacobi method.
        """
        for i in range(self.n):
            sumRemRow = -self.mat[i][i]
            for j in range(len(self.mat[i])):
                sumRemRow += self.mat[i][j]
            if self.mat[i][i] <= sumRemRow:
                return False
        return True

    @handleError
    def _validateListValues(self, lst):
        """
        Validates the values in the list.
        Returns True if correct.
        """
        if not isinstance(lst, list):
            raise Exception("Invalid input - Expected list")

        for i in lst:
            if not isinstance(i, float) and not isinstance(i, int):
                raise Exception(
                    f"Invalid type of value received {type(i)}.\nExpected float or int."
                )

        return True

    @handleError
    def _iterativeMethod(self, b, m, isJacobi=True):
        """
        Takes a list (denoting vector `b`) and number of iterations `m` as its arguments, and performs `m` iterations of the Jacobi / Gauss-Siedel iterative procedure.
        Return the final iteration value, and value of the term `||Ax⁽ᵏ⁾ - b||₂` from all the m iterations
        """
        if not self._validateListValues(b):
            return
        if not isinstance(m, int) or m < 1:
            raise Exception(f"Invalid m received {m}.\nExpected a positive int.")
        if isJacobi and not self.isDRDominant():
            raise Exception("Not solving because convergence is not guranteed.")

        # Converting the class matrix to numpy array
        A = []
        for i in range(self.n):
            el = []
            for j in range(self.n):
                el.append(self.mat[i][j])
            A.append(el)
        A = np.array(A)

        # Initialize the iteration vector
        prevX = [0] * self.n
        x = [0] * self.n
        ansError = []

        # Doing m iterations
        for _ in range(m):
            # Computing each xᵢ
            for i in range(self.n):
                sumRemRow = 0
                for j in range(self.n):
                    if isJacobi:
                        if j != i:
                            sumRemRow += self.mat[i][j] * prevX[j]
                    else:
                        if j < i:
                            sumRemRow += self.mat[i][j] * x[j]
                        elif j > i:
                            sumRemRow += self.mat[i][j] * prevX[j]
                x[i] = (b[i] - sumRemRow) / self.mat[i][i]
            # ||Ax⁽ᵏ⁾ - b||₂
            ansError.append(np.linalg.norm(A @ np.array(x) - np.array(b)))
            prevX = x[:]

        ansX = prevX[:]  # Final iteration value

        return (ansError, ansX)

    @handleError
    def jSolve(self, b, m):
        """
        Takes a list (denoting vector `b`) and number of iterations `m` as its arguments, and performs `m` iterations of the Jacobi iterative procedure.
        Return the final iteration value, and value of the term `||Ax⁽ᵏ⁾ - b||₂` from all the m iterations
        """
        return self._iterativeMethod(b, m, isJacobi=True)

    @handleError
    def gsSolve(self, b, m):
        """
        Takes a list (denoting vector `b`) and number of iterations `m` as its arguments, and performs `m` iterations of the Gauss-Siedel iterative procedure.
        Return the final iteration value, and value of the term `||Ax⁽ᵏ⁾ - b||₂` from all the m iterations
        """
        return self._iterativeMethod(b, m, isJacobi=False)


if __name__ == "__main__":
    # Sample Test Case 1
    s = SquareMatrixFloat(3)
    print(s)

    # Sample Test Case 2
    s = SquareMatrixFloat(4)
    s.sampleSymmetric()
    print(s)

    # Sample Test Case 3
    s = SquareMatrixFloat(4)
    s.sampleSymmetric()
    print(s)
    s.toRowEchelonForm()
    print(s)

    # Sample Test Case 4
    s = SquareMatrixFloat(4)
    s.sampleSymmetric()
    print(s.isDRDominant())
    print(s)

    # Sample Test Case 5
    s = SquareMatrixFloat(4)
    s.sampleSymmetric()
    (e, x) = s.jSolve([1, 2, 3, 4], 10)
    print(x)
    print(e)

    # Sample Test Case 6
    s = SquareMatrixFloat(4)
    s.sampleSymmetric()
    (e, x) = s.gsSolve([1, 2, 3, 4], 10)
    print(x)
    print(e)
