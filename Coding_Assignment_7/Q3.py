"""
111901030
Mayank Singla
Coding Assignment 7 - Q3
"""

# %%
# Below lines are to ignore the pylint warning in VSCode
# pylint: disable=abstract-method
# pylint: disable=pointless-string-statement
# pylint: disable=invalid-name


def computeNthRoot(n, a, ep):
    """
    Computes the nth root of `a` with an error tolerance of ep
    """
    low, high = 0, a  # lower and higher bound

    def f(x):
        """
        Function to be evaluated
        """
        return x**n - a

    while abs(low - high) > ep:
        # Using the Bisection method to find the root
        mid = (low + high) / 2  # midpoint
        if f(mid) == 0:
            # if the midpoint is the root
            return mid
        elif f(mid) * f(low) >= 0:
            # if the midpoint is in the same direction as the lower bound
            low = mid
        else:
            # if the midpoint is in the same direction as the higher bound
            high = mid

    return (low + high) / 2


if __name__ == "__main__":
    # Testing the function
    m = 19
    num = 6**m
    eps = 0.00001
    print(f"The {m}th root of {num} is {computeNthRoot(m, num, eps)}")
