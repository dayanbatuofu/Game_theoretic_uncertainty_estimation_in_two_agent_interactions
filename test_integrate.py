from scipy import integrate


def integrand(x, a, b):
    return a*x**2 + b


if __name__ == "__main__":
    a = 1
    b = 2
    res = integrate.quad(integrand, 0, 1, args=(a, b))
    print(res)
