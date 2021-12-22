import argparse
from imutils import perspective
import imutils
import cv2
import numpy as np
import time
from scipy import stats
import logging


def compute_plate_shape(image):
    """
    compute plate's length, width
    """
    list_width = cropped_plate.shape[1] * 2.504 * 5
    list_length = cropped_plate.shape[0] * 2.504 * 5

    if abs(list_width - apcs_parameters['board_width']) / (0.1 + apcs_parameters['board_width']) < 0.07:
        list_width = apcs_parameters['board_width']

    if abs(list_length - apcs_parameters['board_length']) / (0.1 + apcs_parameters['board_length']) < 0.07:
        list_length = apcs_parameters['board_length']

    return list_width, list_length

def column_parsing(column, upper_thresh=2, downer_thresh=5):
    """
    analysis each column. compute mode and choose threshold

    # Arguments:
        upper_thresh: upper indent by column mode;
        downer_thresh: downer indent by column mode.

    # output:
        fgmask: binary image with defects for one column
    """

    moda = stats.mode(column, axis=None)[0][0]

    fgmask = cv2.threshold(column, moda + upper_thresh, 255, cv2.THRESH_BINARY)[1]  # подсвечивать светлые
    fgmask2 = cv2.threshold(255 - column, moda + 255 - downer_thresh, 255, cv2.THRESH_BINARY)[1]  # подсвечивать темные
    fgmask = fgmask2 + fgmask

    return fgmask


def thresh_detector(img_input, step, thresh_up=2, thresh_down=5, scale_thresh=6, color_thresh=5):
    """
    make a binary image with defects for all image
    # Arguments:
        img_input: concated image from camera;
        step: column width, for each of which is detected defects;
        crop_thresh: threshold for cropping plate;
        thresh_up: upper indent by column mode;
        thresh_down: downer indent by column mode;
        scale_thresh: physical radius;
        color_thresh: color radius.

    # output:
        mask: binary image with defects for all image;
        img: image after filtering (not used now).
    """

    img = img_input.copy()
    mask = []
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.pyrMeanShiftFiltering(img, scale_thresh, color_thresh)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    for x in range(img.shape[1] // step + 1):
        fgmask = column_parsing(img[:, x * step:step * x + step], thresh_up, thresh_down)

        mask.append(fgmask)
    mask = cv2.hconcat(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask = cv2.erode(mask, kernel, iterations=6)
    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=6)
    return mask, img


def four_point_transform(image, box):
    """
    Transform box to vertical position.
    Used for transform plate.

    # Arguments:
        image: image;
        box: x,y,h,w

    # output:
        warped: transformed and cropped image by input box
    """
    rect = perspective.order_points(box)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def crop_plate2(img, crop_thresh=30):
    """
    detect and crop plate on image

    # Arguments:
        img: image;
        crop_thresh: threshold for detect.

    # output:
        orig_img: cropped plate;
        img: cropped plate without borders.
    """
    kernel = np.ones((2, 15), np.uint8)

    fgmask = cv2.erode(img, kernel, iterations=5)
    fgmask = cv2.dilate(fgmask, kernel, iterations=5)
    fgmask = cv2.threshold(fgmask, crop_thresh, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img, img[:, 35:img.shape[1] - 40]
    c = max(contours, key=cv2.contourArea)

    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    rect = perspective.order_points(box)
    orig_img = four_point_transform(img, box)
    img = orig_img[:, 35:orig_img.shape[1] - 40]
    return orig_img, img


def crop_plate(img, crop_thresh=30):
    """
    now not used.
    """
    kernel = np.ones((2, 15), np.uint8)

    fgmask = cv2.erode(img, kernel, iterations=5)
    fgmask = cv2.dilate(fgmask, kernel, iterations=5)
    fgmask = cv2.threshold(fgmask, crop_thresh, 255, cv2.THRESH_BINARY)[1]

    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
    else:
        return None, img
    x, y, w, h = cv2.boundingRect(c)

    x1 = x - 10 if x - 10 > 0 else 0
    y1 = y - 10 if y - 10 > 0 else 0

    orig_img = img[y:y + h, x:x + w]
    img = orig_img[:, 35:orig_img.shape[1] - 40]
    return orig_img, img


def detector(fgmask, img):
    """
    find contours on image

    # Arguments:
        fgmask: binary image with defects for all image;
        img: image without borders.

    # output:
        img: image without borders;
        masks: list of numpy array masks.
    """
    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_RGB2GRAY)
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    num_cont = 0
    masks = []
    for c in contours:
        if cv2.contourArea(c) > 5 and cv2.contourArea(c) < 10000 and cv2.arcLength(c, True) < 5000:
            cv2.drawContours(image=img, contours=[c], contourIdx=0, color=(0, 0, 255), thickness=2)
            c[:, :, 0] = c[:, :, 0] + 35
            masks.append(c)
            num_cont += 1

    return img, masks


def defect_overlaying(img, step=30, crop_thresh=30, thresh_up=3, thresh_down=-100, scale_thresh=7, color_thresh=10):
    """
    Cropping and detecting defects on plate

    # Arguments:
        img: concated image from camera;
        step: column width, for each of which is detected defects;
        crop_thresh: threshold for cropping plate;
        thresh_up: upper indent by column mode;
        thresh_down: downer indent by column mode;
        scale_thresh: physical radius;
        color_thresh: color radius.

    # output:
        origin_img: cropped image;
        masks: list of numpy array masks;
        drawed_img: cropped image with drawn masks
    """
    orig_img, img = crop_plate2(img=img, crop_thresh=int(crop_thresh))
    if orig_img is None:
        return orig_img, [], None
    if img is None or 0 in img.shape:
        return img, [], orig_img
    mask, _ = thresh_detector(img, step, thresh_up, thresh_down, scale_thresh, color_thresh)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
    # logging.info('img shape befor detector' + str(img.shape))
    img, masks = detector(mask, img)
    drawed_img = orig_img.copy()
    drawed_img[:, 35:orig_img.shape[1] - 40] = img
    # image_with_masks += 50 # correct brightness

    return orig_img, masks, drawed_img
