"""
111901030
Mayank Singla
Coding Assignment 5 - Q7
"""

# %%
from scipy.fft import fft, ifft


def computeProduct(a, b):
    """
    Computes the product of two large n-digit numbers
    """
    # Making the polynomial form
    num_digits = len(str(a)) + len(str(b))
    ta, tb = a, b
    xa, xb = [], []
    for _ in range(num_digits):
        xa.append(ta % 10)
        ta //= 10
        xb.append(tb % 10)
        tb //= 10

    # Computing the FFTs of the polynomial forms
    xa = fft(xa)
    xb = fft(xb)

    # Multiplying the FFTs
    prod = xa * xb

    # Computing the inverse FFT
    prod = ifft(prod)

    # Computing the actual final product
    ans, ex = 0, 0
    for i in prod:
        ans += (i.real) * (10**ex)
        ex += 1
    print(f"Computed Product: {int(ans)}")


if __name__ == "__main__":
    a = 123456789
    b = 12345
    computeProduct(a, b)
    print(f"Actual Product: {a * b}")
