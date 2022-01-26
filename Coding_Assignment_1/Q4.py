"""
111901030
Mayank Singla
Coding Assignment 1 - Q4
"""

# %%
from random import choices


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


class TextGenerator:
    prefDict = {}  # Prefix Dictionary for each pair
    _freqDist = {}  # Frequency Distribution for each pair

    def _createFreqDist(self):
        """
        Creates the frequency distribution of next words for each pair of words
        """
        self._freqDist.clear()  # Clearing the previous frequency distribution

        # Looping all the (key, val) in prefDict
        for p, words in self.prefDict.items():
            if not (
                p in self._freqDist
            ):  # Adding the pair if it is not in the dictionary
                self._freqDist[p] = {}

            # Looping through all the words and storing their frequency in a dictionary
            for word in words:
                if not (word in self._freqDist[p]):
                    self._freqDist[p][word] = 1
                else:
                    self._freqDist[p][word] += 1

            # Storing the list of words and their frequencies for each pair
            self._freqDist[p] = [
                list(self._freqDist[p].keys()),
                list(self._freqDist[p].values()),
            ]

    def assimilateText(self, filename):
        """
        Takes filename as its argument and reads all the text in the file.
        Creates a prefix dictionary that maps a pair (2-tuple) of words to a list of words which follow that pair in the text.
        """
        self.prefDict.clear()  # Clearing the prefix Dictionary

        # Read all the contents of the file
        with open(filename) as inputFile:
            text = inputFile.read()

        # text = """Hello this is "Test"! 😁😊 World

        # Hello this I'm "is2 Test!" World2🌷🌸💐
        # Hello this is another `Test`World3"""

        words = text.split()  # Extracting the list of words
        numWords = len(words)  # Number of words in the file

        if numWords < 3:  # If number of words are less than 3, then return
            return

        first, second = words[0], words[1]  # first and second word
        # Creating the prefix dictionary that maps a pair (2-tuple) of words to a list of words which follow that pair in the text.
        for i in range(2, numWords):
            currWord = words[i]  # current word
            if not (
                (first, second) in self.prefDict
            ):  # If that pair is not in the dictionary already
                self.prefDict[(first, second)] = []
            # Adding the current word to the list of words of the current pair
            self.prefDict[(first, second)].append(currWord)
            first = second  # Updating the first member of the pair
            second = currWord  # Updating the second member of the pair

        # Creating the frequency distribution for the list of words for each pair
        self._createFreqDist()

    @handleError
    def generateText(self, n: int, startWord=""):
        """
        Creates random text based on the triplets contained in the prefix dictionary.
        Args:
            n(int): Number of words of the text to generate
            startWord(str)?: Starting word of the text to generate
        """
        pairs = list(self.prefDict.keys())  # List of all the pairs of words
        currPair = ()  # The current pair in the text

        if not startWord:
            # If start word is not provided
            currPair = choices(pairs)[0]  # choosing any random pair as the current pair
        else:
            # If start word is provided
            # Finding list of pairs having start word as first member
            startList = list(filter(lambda p: p[0] == startWord, pairs))
            if len(startList) == 0:
                # If there is no pair found, we can't make the text
                raise Exception("Unable to produce text with the specified start word.")
            # Choosing any random pair from the found pairs as the current pair
            currPair = choices(startList)[0]

        if n == 1:
            # If only one word needs to be generated
            print(currPair[0])
            return

        # Building the final text generated
        text = "{first} {second}".format(first=currPair[0], second=currPair[1])
        wordCount = 2  # Current number of words added to the final text

        # Looping till number of words in text are less than n
        while wordCount < n:
            text += " "  # Adding space character b/w the words

            if currPair in self._freqDist:
                # If currPair of word in present in the dictionary
                # Generating a random next word from the list of next words of currPair and based on the weights(frequencies) of the next word
                nextWord = choices(
                    self._freqDist[currPair][0], self._freqDist[currPair][1]
                )[0]
                text += nextWord  # Appending that word to the text
                currPair = (currPair[1], nextWord)  # Updating the current pair of words
                wordCount += 1  # Updating the word count
            else:
                # If current pair of word in not present in the dictionary
                currPair = choices(pairs)[
                    0
                ]  # Choosing any random pair of word as the next word
                if wordCount != n - 1:
                    # If it is not the last word to be added to the text
                    text += currPair[0] + " " + currPair[1]  # Appending both the words
                    wordCount += 2  # Updating the word count
                else:
                    # If it is the last word to be added to the text
                    text += currPair[0]  # Appending the word to the text
                    wordCount += 1  # Updating the word count

        # Printing the final random text generated
        print(text)


if __name__ == "__main__":
    tg = TextGenerator()
    tg.assimilateText("sherlock.txt")

    # Sample Test Case 1
    tg.generateText(100)
    print()

    # Sample Test Case 2
    tg.generateText(100, "London")
    print()

    # Sample Test Case 3
    tg.generateText(50, "Wedge")
    print()
