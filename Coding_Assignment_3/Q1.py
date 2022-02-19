"""
111901030
Mayank Singla
Coding Assignment 3 - Q1
"""

# %%
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
        return True

    @handleError
    def __init__(self, lst):
        """
        Initializes the row vector with the values in the list.
        """
        if not self._validateListValues(lst):
            return

        # Using numpy array to store the values
        self.vec = np.array(lst)

    @handleError
    def __str__(self):
        """
        Returns the string representation of the row vector.
        """
        return " ".join(str(i) for i in self.vec)

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

        ans = self.vec + rv.vec
        return RowVectorFloat(ans.tolist())

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
        ans = self.vec * scalar
        return RowVectorFloat(ans.tolist())

    @handleError
    def __rmul__(self, scalar):
        """
        Multiplies a row vector with a scalar.
        Operator looks for __mul__ in right operand.
        """
        return self.__mul__(scalar)


if __name__ == "__main__":
    # Sample Test Case 1
    r = RowVectorFloat([1, 2, 4])
    print(r)
    print(len(r))
    print(r[1])
    r[2] = 5
    print(r)

    r = RowVectorFloat([])
    print(len(r))

    # Sample Test Case 2
    r1 = RowVectorFloat([1, 2, 4])
    r2 = RowVectorFloat([1, 1, 1])
    r3 = 2 * r1 + (-3) * r2
    print(r3)
