import random


def estimate_depth(plate_image, defect_masks):
    # crop image with masks
    # do some things
    # estimate depths
    if len(defect_masks) > 0:
        print('len masks', len(defect_masks))
        high_precision_depths = [random.randint(0, 500) for _ in range(len(defect_masks))]
    else:
        high_precision_depths = []
    return high_precision_depths
