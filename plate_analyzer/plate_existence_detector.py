import cv2
import numpy as np


class plate_existence_detector:
    def __init__(self, k, background):
        self.k = k
        self.background = background
        self.concated_img = None
        self.state = 0  # no list
        self.kernel = np.ones((2, 1), np.uint8)

    def is_list(self, cur_frame):
        """
        detector of existence plate
        # output:
            True/False - is list on current frame
        """
        fgmask = cv2.absdiff(self.background, cur_frame)
        fgmask = cv2.erode(fgmask, self.kernel, iterations=5)
        fgmask = cv2.dilate(fgmask, self.kernel, iterations=5)
        fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)[1]
        unique, counts = np.unique(fgmask, return_counts=True)
        df = {0: 0, 255: 0}
        for u, c in zip(unique, counts):
            df[u] = c

        return int(df[255] > self.k * df[0])

    def state_of_existence(self, prev_frame, cur_frame):
        """
        concate frames, which contains plate

        # Arguments:
            prev_frame: previously frame;
            cur_frame: current frame.

        # output:
            concated_image;
            state: 0 if list is not exist, 1 if list is started, 2 if list in process, 3 if list is completed
        """
        flag = self.is_list(cur_frame)
        if flag == 0 and self.concated_img is not None:
            self.concated_img = cv2.vconcat([self.concated_img, cur_frame[:100, :]])
            self.state = 3  # list is completed

        elif flag == 1 and self.concated_img is not None:
            self.concated_img = cv2.vconcat([self.concated_img, cur_frame])
            self.state = 2  # list in process

        elif flag == 1 and self.concated_img is None:
            self.concated_img = cur_frame if prev_frame is None else cv2.vconcat([[prev_frame][-100:, :], cur_frame])
            self.state = 1  # list is started

        return self.concated_img, self.state
