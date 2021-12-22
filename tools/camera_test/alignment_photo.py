import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image


def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)


# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = [float(hist[0])]
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # alpha, beta
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    new_hist = cv2.calcHist([gray], [0], None, [256], [minimum_gray, maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0, 256])
    plt.show()

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


def increase_brightness(img, value=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def clahe(image, iteration=1):
    for i in range(iteration):
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])

        cdf = hist.cumsum()
        clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(15, 1))

        image = clahe.apply(image)
    return image


def visual(img, new_img, h=1000, w=100):
    cv2.imshow('orig', cv2.resize(img, (h, w)))
    # cv2.imwrite('auto_result.png', auto_result)
    cv2.imshow('clahe', cv2.resize(new_img, (h, w)))

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
    cv2.imwrite(path + 'new_list.jpg', new_img)
    new_surf = new_ax.plot_surface(new_x, new_y, new_z, cmap='inferno')
    # linewidth=0, antialiased=False)

    cv2.waitKey()
    plt.show()


def find_list(img):
    bounding_rect = []
    background = cv2.imread('content194.jpg', 0)
    diff_img = cv2.absdiff(background, img)
    # img[diff_img[]] =
    kernel = np.ones((10, 10), np.uint8)
    fgmask = cv2.erode(img, kernel, iterations=5)
    diff_img = cv2.threshold(fgmask, 34, 255, 0)[1]
    # cv2.imshow('fgmask', cv2.resize(fgmask, (1000, 100)))
    cv2.imshow('diff', cv2.resize(diff_img, (1000, 1000)))
    # cv2.waitKey()
    new_img = img * (diff_img > 0)
    contours, hierarchy = cv2.findContours(new_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('len contours', len(contours))
    c = max(contours, key=cv2.contourArea)
    # print(c)
    cv2.drawContours(new_img, [c], -1, (255, 0), 3)
    cv2.imshow('contours', cv2.resize(new_img, (1000, 1000)))

    return img * (diff_img > 0)


def aligment(img):
    new_img = img.copy()  # cv2.imread(path + 'content161.jpg', 0)
    # new_img = new_img[:, 480:-870]
    print(img.shape, new_img.shape)

    h, w = img.shape

    mu_list = np.mean(new_img)
    print('mu_list =', mu_list)
    mu_vert = [np.sum(new_img[:, x]) / h for x in range(w)]
    print('len mu_vert =', len(mu_vert))
    # print(mu_vert)
    alpha_vert = mu_list / mu_vert
    # print(alpha_vert)
    a = [new_img[:, x] * alpha_vert[x] for x in range(w)]
    print(len(a))
    # new_img = np.array([new_img[:, x] * alpha_vert[x] for x in range(w)])
    for x in range(w):
        new_img[:, x] = new_img[:, x] * alpha_vert[x]

    return new_img


if __name__ == "__main__":
    path = '/home/mix_kup/Desktop/intact/intact/knauf/debug_detector/det_proccessing/'
    # print(path + 'content62.jpg')
    img1 = cv2.imread(path + 'IMG_7929.jpg', 0)
    # img1 = img1[100:-100, 100:-100]
    print(img1.shape)
    new_img = aligment(img1)
    visual(img1, new_img)

'''
#==================TRACK BAR===========================


import cv2
import numpy as np
alpha = 0.3
beta = 80
img_path = "../debug_detector/content2/clahe.jpg"
img = cv2.imread(img_path)
img2 = cv2.imread(img_path)
img = cv2.resize(img, (2000, 270))
img2 = cv2.resize(img2, (2000, 270))
def updateAlpha(x):
    global alpha, img, img2
    alpha = cv2.getTrackbarPos('Alpha', 'image')
    alpha = alpha * 0.01
    img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))
def updateBeta(x):
    global beta, img, img2
    beta = cv2.getTrackbarPos('Beta', 'image')
    img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))
# Создать окно
cv2.namedWindow('image')
cv2.createTrackbar('Alpha', 'image', 0, 300, updateAlpha)
cv2.createTrackbar('Beta', 'image', 0, 255, updateBeta)
cv2.setTrackbarPos('Alpha', 'image', 100)
cv2.setTrackbarPos('Beta', 'image', 10)
while (True):
    cv2.imshow('image', img)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()


'''
