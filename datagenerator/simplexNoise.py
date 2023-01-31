import os

import numpy as np

import noise


def noise_3d(cube_shape, verbose=False):
    if verbose:
        print("   ... inside noise3D")
    noise3d = np.random.exponential(
        1.0 / 100.0, size=cube_shape[0] * cube_shape[1] * cube_shape[2]
    )
    sign = np.random.binomial(
        1, 0.5, size=cube_shape[0] * cube_shape[1] * cube_shape[2]
    )
    sign[sign == 0] = -1
    noise3d *= sign
    noise3d = noise3d.reshape((cube_shape[0], cube_shape[1], cube_shape[2]))
    return noise3d


def perlin(xsize, ysize, base=None, octave=1, lac=1.9, do_rotate=True):
    # print "   ...inside perlin"
    if base is None:
        base = np.random.randint(255)
    temp = np.array(
        [
            [
                noise.pnoise2(
                    float(i) / xsize,
                    float(j) / ysize,
                    lacunarity=lac,
                    octaves=octave,
                    base=base,
                )
                for j in range(ysize)
            ]
            for i in range(xsize)
        ]
    )
    # randomly rotate image
    if do_rotate:
        if xsize == ysize and np.random.binomial(1, 0.5) == 1:
            number_90_deg_rotations = int(np.random.uniform(1, 4))
            temp = np.rot90(temp, number_90_deg_rotations)
        # randomly flip left and right, top and bottom
        if np.random.binomial(1, 0.5) == 1:
            temp = np.fliplr(temp)
        if np.random.binomial(1, 0.5) == 1:
            temp = np.flipud(temp)
    return temp


def noise_2d(xsize, ysize, threshold, octaves=9):
    for i in range(25):
        im = perlin(xsize, ysize, octave=octaves, lac=1.9)

        im_x = np.mean(im, axis=0)
        im_x1 = np.mean(im[: -im.shape[0] / 2, :], axis=0)
        im_x2 = np.mean(im[im.shape[0] / 2 :, :], axis=0)
        nxcor1 = np.mean((im_x - im_x.mean()) * (im_x1 - im_x1.mean())) / (
            im_x.std() * im_x1.std()
        )
        nxcor2 = np.mean((im_x - im_x.mean()) * (im_x2 - im_x2.mean())) / (
            im_x.std() * im_x2.std()
        )
        nxcor3 = np.mean((im_x1 - im_x1.mean()) * (im_x2 - im_x2.mean())) / (
            im_x1.std() * im_x2.std()
        )
        test_x = np.mean((nxcor1, nxcor2, nxcor3))
        if np.isnan(test_x):
            test_x = 1.0

        im_y = np.mean(im, axis=1)
        im_y1 = np.mean(im[:, : -im.shape[0] / 2], axis=1)
        im_y2 = np.mean(im[:, im.shape[0] / 2 :], axis=1)
        nycor1 = np.mean((im_y - im_y.mean()) * (im_y1 - im_y1.mean())) / (
            im_y.std() * im_y1.std()
        )
        nycor2 = np.mean((im_y - im_y.mean()) * (im_y2 - im_y2.mean())) / (
            im_y.std() * im_y2.std()
        )
        nycor3 = np.mean((im_y1 - im_y1.mean()) * (im_y2 - im_y2.mean())) / (
            im_y1.std() * im_y2.std()
        )
        test_y = np.mean((nycor1, nycor2, nycor3))
        if np.isnan(test_y):
            test_y = 1.0

        print(
            i,
            (test_x, test_y),
            "    thresholds = ",
            (threshold, 1.0 / threshold),
            threshold < test_x < 1.0 / threshold
            or threshold < test_y < 1.0 / threshold,
        )

        if test_x > threshold or test_y > threshold:
            continue
        else:
            return im, test_x, test_y

    return im, test_x, test_y
