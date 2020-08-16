import numpy as np
from PIL import Image


def main():

    sz = 256

    white = 0xff
    major = 0xcc
    minor = 0xee

    tile = white * np.ones((sz, sz), dtype=np.uint8)

    tile[0] = major
    tile[sz//4::sz//4] = minor

    tile = np.minimum(tile, tile[::-1])
    tile = np.minimum(tile, tile.T)

    img = Image.fromarray(tile)

    img.save('tile.png')


if __name__ == '__main__':
    main()
