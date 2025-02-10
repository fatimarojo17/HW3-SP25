import scipy.integrate as integrate
import scipy.special as special
import numpy as np


def gamma_function(alpha):
    """ Compute the Gamma function for a given alpha. """
    return special.gamma(alpha)


def t_distribution_probability(m, z):
    """ Compute the probability P(T ≤ z) for a t-distribution with m degrees of freedom. """

    Km = gamma_function(0.5 * m + 0.5) / (np.sqrt(m * np.pi) * gamma_function(0.5 * m))

    def integrand(u):
        return (1 + (u ** 2) / m) ** (-(m + 1) / 2)

    result, _ = integrate.quad(integrand, -np.inf, z)

    return Km * result


def main():
    """ Main function to solicit input and compute probabilities. """

    while True:
        try:
            m = int(input("Enter degrees of freedom (m): "))
            z = float(input("Enter value of z: "))

            probability = t_distribution_probability(m, z)

            print(f"P(T ≤ {z} | m = {m}) = {probability:.4f}")

        except ValueError:
            print("Invalid input. Please enter numeric values.")

        again = input("Do you want to compute another probability? (y/n): ").strip().lower()
        if again not in ["y", "yes"]:
            break


if __name__ == "__main__":
    main()
