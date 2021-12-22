import json
from pycocotools import mask
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import cv2


# ============ decode to binary mask =========
def decode_mask(file="/home/mix_kup/Desktop/intact/intact/knauf/defectoscopy/depth_test/test_out_yolact/json_by_dataset"
                     "/mask_detections.json"):
    with open(file=file,
              mode="r") as data_file:
        data = json.load(data_file)  # list по каждому дефекту
    # print(data)
    rle = data[0]['segmentation']
    rle['counts'] = str.encode(rle['counts'])
    mask_def = mask.decode(rle)  # бинарная маска
    # np.savetxt('example1_1.txt', np.array(mask.decode(rle), dtype=int))
    # heat_map = sb.heatmap(mask_def, cmap='Greys')
    # plt.show()
    return mask_def

def unhash(x):
    """
    Source:
    https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    """
    x = (((x >> 16) ^ x) * 0x119de1f3) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x119de1f3) & 0xFFFFFFFF
    x = ((x >> 16) ^ x) & 0xFFFFFFFF
    return x


def badhash(x):
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = ((x >> 16) ^ x) & 0xFFFFFFFF
    return x


image_id = data[0]['image_id']
img = cv2.imread('test_out_yolact/json_by_dataset/4_1.bmp', 0)
print(image_id, unhash(np.uint32(image_id)))
print(img.shape)
img_mask = img * decode_mask()
cv2.imshow('img', img)
cv2.imshow('defect', img_mask)
cv2.waitKey(30)
