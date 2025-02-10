from numericalMethods import GPDF, Probability, Secant


def main():
    """
    This program solicits input from the user to compute probabilities from the Gaussian Normal Distribution
    using Simpson's 1/3 rule for numerical integration. It can also determine the value of c given a probability using the Secant method.
    """
    Again = True
    mean = 0.0
    stDev = 1.0
    c = 0.5
    probability = 0.5
    OneSided = True
    GT = False
    yesOptions = ["y", "yes", "true"]

    while Again:
        response = input(f"Population mean? ({mean:0.3f}) ").strip().lower()
        mean = float(response) if response else mean

        response = input(f"Standard deviation? ({stDev:0.3f}) ").strip().lower()
        stDev = float(response) if response else stDev

        mode = input("Are you specifying c to find P or specifying P to find c? (Enter 'c' or 'p') ").strip().lower()

        if mode == 'c':
            response = input(f"c value? ({c:0.3f}) ").strip().lower()
            c = float(response) if response else c

            response = input(f"Probability greater than c? ({GT}) ").strip().lower()
            GT = True if response in yesOptions else False

            response = input(f"One sided? ({OneSided}) ").strip().lower()
            OneSided = True if response in yesOptions else False

            if OneSided:
                prob = Probability(GPDF, (mean, stDev), c, GT=GT)
                print(f"P(x {'>' if GT else '<'} {c:0.2f} | {mean:0.2f}, {stDev:0.2f}) = {prob:0.3f}")
            else:
                prob = Probability(GPDF, (mean, stDev), c, GT=True)
                prob = 1 - 2 * prob
                if GT:
                    print(
                        f"P({mean - (c - mean)} > x > {mean + (c - mean)} | {mean:0.2f}, {stDev:0.2f}) = {1 - prob:0.3f}")
                else:
                    print(f"P({mean - (c - mean)} < x < {mean + (c - mean)} | {mean:0.2f}, {stDev:0.2f}) = {prob:0.3f}")

        elif mode == 'p':
            response = input(f"Desired probability? ({probability:0.3f}) ").strip().lower()
            probability = float(response) if response else probability

            response = input(f"Probability greater than c? ({GT}) ").strip().lower()
            GT = True if response in yesOptions else False

            response = input(f"One sided? ({OneSided}) ").strip().lower()
            OneSided = True if response in yesOptions else False

            if OneSided:
                c = Secant(lambda c: Probability(GPDF, (mean, stDev), c, GT=GT) - probability, 0.0, 1.0)
                print(f"c value for P(x {'>' if GT else '<'} {c:0.2f} | {mean:0.2f}, {stDev:0.2f}) = {c:0.3f}")
            else:
                func = lambda c: (1 - 2 * Probability(GPDF, (mean, stDev), c, GT=True)) - probability
                c = Secant(func, 0.0, 1.0)
                print(
                    f"c value for P({mean - (c - mean)} < x < {mean + (c - mean)} | {mean:0.2f}, {stDev:0.2f}) = {c:0.3f}")

        response = input("Go again? (Y/N) ").strip().lower()
        Again = response in yesOptions


if __name__ == "__main__":
    main()
