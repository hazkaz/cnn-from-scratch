import numpy as np


class Pool2x2:

    def iterate_regions(self, image):
        h, w,_ = image.shape

        for i in range(0, h - 1,2):
            for j in range(0, w - 1,2):
                im_region = image[i:(i + 2), j:(j + 2)]
                yield im_region, i//2, j//2

    def max_pool(self, input):
        h, w, num_filters = input.shape
        output = np.zeros((h//2, w//2, num_filters),np.int32)
        print(output)
        for im_region, i, j in self.iterate_regions(input):
            # print(im_region)
            # print()
            output[i, j] = np.amax(im_region,axis=(0,1))
        return output


def main():
    a = Pool2x2()
    b=a.max_pool(
        np.array([[[0, 50, 0, 29], [0, 80, 31, 2],[33, 90, 0, 75], [0, 9, 0, 95]]],np.int32))
    print(b)


if __name__ == '__main__':
    main()
