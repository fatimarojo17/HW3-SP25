import math


def gamma_function(alpha):
    """ Compute the Gamma function for a given alpha. """
    if alpha == int(alpha) and alpha > 0:
        # If alpha is a positive integer, gamma(alpha) = (alpha-1)!
        result = 1
        for i in range(1, int(alpha)):
            result *= i
        return result
    else:
        # Use an approximation method for non-integer values of alpha
        # Using Stirling's approximation: gamma(alpha) ~ sqrt(2*pi/alpha) * (alpha/e)^alpha
        return math.sqrt(2 * math.pi / alpha) * (alpha / math.e) ** alpha


def t_distribution_probability(m, z):
    """ Compute the probability P(T ≤ z) for a t-distribution with m degrees of freedom. """

    # Calculate Km for the t-distribution
    Km = gamma_function(0.5 * m + 0.5) / (math.sqrt(m * math.pi) * gamma_function(0.5 * m))

    # Numerical integration using the Trapezoidal rule
    def integrand(u):
        return (1 + (u ** 2) / m) ** (-(m + 1) / 2)

    # Implementing Trapezoidal rule to integrate from -∞ to z
    def trapezoidal_integration(func, a, b, n=1000):
        """ Approximate the integral of func from a to b using the trapezoidal rule. """
        h = (b - a) / n
        integral = (func(a) + func(b)) / 2
        for i in range(1, n):
            integral += func(a + i * h)
        return integral * h

    # Integrate the function from -∞ to z
    result = trapezoidal_integration(integrand, -100, z)

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
