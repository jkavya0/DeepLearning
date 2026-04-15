from generator import ImageGenerator
#from pattern import Checker, Circle, Spectrum

file_path = r"exercise_data/"
label_path = r"Labels.json"

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

    
#for generator

batch_size = 9
image_size = [64, 64, 3]  


gen = ImageGenerator(
    file_path=file_path,
    label_path=label_path,
    batch_size=batch_size,
    image_size=image_size,
    rotation=True,
    mirroring=True,
    shuffle=True
)

gen.show()

print("you got your image generator!!")


