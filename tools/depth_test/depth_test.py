import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image


def visual(img, new_img):
    cv2.imshow('orig', cv2.resize(img, (800, 1000)))
    # cv2.imwrite('auto_result.png', auto_result)
    cv2.imshow('clahe', cv2.resize(new_img, (0, 0), fx=0.2, fy=0.2))

    img = img.astype(np.float32)

    x = np.arange(0, img.shape[1], step=1)
    y = np.arange(0, img.shape[0], step=1)
    x, y = np.meshgrid(x, y)
    z = img[:, :]
    print(x.shape, y.shape, z.shape)
    fig = plt.figure()
    ax = Axes3D(fig)

    surf = ax.plot_surface(x, y, z,
                           linewidth=0, antialiased=False)

    new_img = new_img.astype(np.float32)

    new_x = np.arange(0, new_img.shape[1], step=1)
    new_y = np.arange(0, new_img.shape[0], step=1)
    new_x, new_y = np.meshgrid(new_x, new_y)
    new_z = new_img[:, :]
    # new_z[(new_z > 71) * (new_z < 88)] = 80
    print(new_x.shape, new_y.shape, new_z.shape)
    new_fig = plt.figure()
    new_ax = Axes3D(new_fig)
    # new_ax.plot_wireframe(new_x, new_y, new_z,)
    # cv2.imwrite(path + 'new_list.jpg', new_img)
    new_surf = new_ax.plot_surface(new_x, new_y, new_z, cmap='inferno')
    # linewidth=0, antialiased=False)

    cv2.waitKey()
    plt.show()


if __name__ == '__main__':
    path = '../../debug_detector/content_mono10_2918/'
    frame = cv2.imread(path + 'content775.jpg', 0)

    visual(frame, frame[:, 2250:2600])
