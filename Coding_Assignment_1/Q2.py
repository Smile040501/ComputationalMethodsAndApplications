"""
111901030
Mayank Singla
Coding Assignment 1 - Q2
"""

# %%
import matplotlib.pyplot as plt
from random import random


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
            method(ref, *args, **kwargs)
        except Exception as err:
            print(type(err))
            print(err)

    return decorator


class Dice:
    """
    Dice class as mentioned in the question
    Attributes:
        numSides: int
            The number of sides of the dice

        probDist: list[float]
            The probability distribution of the sides of the dice

    Methods:
        _validateSides(numSides);
        _validateProbdist(dist);
        _computeCDFIntervals(self);
        __init__(numSides);
        __str__();
        setProb(dist);
        roll(n);
    """

    def _validateSides(self, numSides):
        """
        Validates the given number of sides from input
        """
        if (not isinstance(numSides, int)) or (numSides <= 4):
            raise Exception("Cannot construct the dice")

    def _validateProbDist(self, dist):
        """
        Validates the given probability distribution of the sides
        """
        if (
            len(dist) != self.numSides
        ):  # We need to assign probability to each side of dice
            raise Exception("Invalid probability distribution")

        sumProb = 0
        for i in dist:
            sumProb += i
            if i < 0:  # All the Probabilites should be >= 0
                raise Exception("Invalid probability distribution")

        if round(sumProb) != 1:  # Sum of probabilites should be 1
            raise Exception("Invalid probability distribution")

    def _computeCDFIntervals(self):
        """
        Returns the list of intervals of the CDF of the probability distribution
        """
        ans = []
        sum = 0
        # Appending the intervals to the answer list
        for val in self.probDist:
            prevSum = sum
            sum += val
            ans.append((prevSum, sum))
        return ans

    @handleError
    def __init__(self, numSides=6):
        """
        Constructor for the class Dice with default numSides as 6.
        Generates default probability distribution with equal probabilities for all the sides.
        """
        self._validateSides(numSides)
        self.numSides = numSides
        self.probDist = []
        # Setting default probabilites for each side
        for _ in range(self.numSides):
            self.probDist.append(1 / self.numSides)

    def __str__(self):
        """
        String dunder method, so that the object of class Dice should be printable in the format specified
        """
        val = "Dice with {numFaces} faces and probability distribution {{".format(
            numFaces=self.numSides
        )
        # Appending probability of each side to the string
        for i, num in enumerate(self.probDist):
            val = val + str(num)
            if i != len(self.probDist) - 1:
                val = val + ", "
        val = val + "}"
        return val

    @handleError
    def setProb(self, dist):
        """
        Validates and sets the given probability distribution for the sides of the dice
        """
        self._validateProbDist(dist)
        self.probDist = list(dist)

    def roll(self, n: int):
        """
        Simulate n throws of a dice and generates random number based on the sampling.
        Displays a bar chart showing the expected and actual number of occurrences of each face when the dice is thrown n times.
        """
        # Getting the CDF intervals
        intervals = self._computeCDFIntervals()

        # The expected number of occurrences of each face
        expected = list(map(lambda x: n * x, self.probDist))

        # Computing the actual number of occurrences of each face
        actual = [0] * self.numSides
        for _ in range(n):
            U = random()  # Generating a random number
            # Finding in which CDF interval, U lies
            for i, (l, r) in enumerate(intervals):
                if l < U and U < r:
                    actual[i] += 1  # Incrementing the count of the found interval
                    break

        # Set of points for the x-axis
        xpoints = list(range(1, self.numSides + 1))

        # Shifting the points to the left for shifting the bar graph
        shiftLeftXpoints = list(map(lambda x: x - 0.2, xpoints))

        # Shifting the points to the right for shifting the bar graph
        shiftRightXpoints = list(map(lambda x: x + 0.2, xpoints))

        # Giving labels and title to the plot
        plt.title(
            "Outcome of {n} throws of a {numSides}-faced dice".format(
                n=n, numSides=self.numSides
            )
        )
        plt.xlabel("Sides")
        plt.ylabel("Occurrences")

        # Plotting the bar graph for actual and expected occurrences in blue and red
        plt.bar(shiftLeftXpoints, actual, width=0.4, color="b", label="Actual")
        plt.bar(shiftRightXpoints, expected, width=0.4, color="r", label="Expected")

        # Locating the legend box as expected in the question
        plt.legend(bbox_to_anchor=(0.5, 1.20), loc="upper center", ncol=2)

        # Displaying the graph
        plt.show()


if __name__ == "__main__":

    # Sample Test Case1
    d = Dice(5)
    d.setProb((0.1, 0.2, 0.3, 0.2, 0.2))
    print(d)
    d.roll(10000)

    # Sample Test Case 2
    d = Dice(8)
    print(d)
    d.roll(10000)
