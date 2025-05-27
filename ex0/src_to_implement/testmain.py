# testmain.py
# Author: Kavya Jayaramaiah

from pattern import Checker, Circle, Spectrum

def test_checker():
    print("\n--- Checkerboard Pattern ---")
    resolution = 200
    tile = 25
    checker = Checker(resolution, tile)
    checker.show()


def test_circle():
    print("\n--- Circle Pattern ---")
    resolution = 200
    radius = 50
    position = (100, 100)  # (x, y) center of the circle
    circle = Circle(resolution, radius, position)
    circle.show()


def test_spectrum():
    print("\n--- Spectrum Pattern ---")
    resolution = 300
    spectrum = Spectrum(resolution)
    spectrum.show()


def main():
    test_checker()
    test_circle()
    test_spectrum()


if __name__ == "__main__":
    main()
