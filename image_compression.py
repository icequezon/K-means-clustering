from PIL import Image
import numpy as np

from main import K_Means


def main():

        my_pic = Image.open('kmimg1.png')
        width, height = my_pic.size
        px = my_pic.load()
        pixels = []
        for x in range(height):
            for y in range(width):
                pixels = pixels + [(px[x, y])]

        X = np.array(pixels)
        km = K_Means(16)
        values = km.fit(X)

        new_image = Image.new('RGB', (width, height))
        new_px = new_image.load()
        for x in range(height):
            for y in range(width):
                new_px[x, y] = tuple(np.array(values[km.pred(px[x, y])]).astype(int))
        new_image.show()
        my_pic.show()


if __name__ == "__main__":
        main()
