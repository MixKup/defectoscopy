from harvesters.core import Harvester
from harvesters.util.pfnc import mono_location_formats, \
    rgb_formats, bgr_formats, \
    rgba_formats, bgra_formats
import numpy as np
import cv2
import time
import glob
import os


class FocusMeasurer:

    def __init__(self):
        self.max_fm = -1
        self.fm = 0

    def variance_of_laplacian(self, image):
        print('============= shape', image.shape, '===========')
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian 
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def add_focus_measure(self, frame):
        self.fm = self.variance_of_laplacian(frame)
        # save maximum registered value
        if self.max_fm < self.fm:
            self.max_fm = self.fm
        text = 'Sharpness'
        cv2.putText(frame, "{}: {:.4f}".format(text, self.fm), (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 3.8, (0, 255, 0),
                    10)
        text = 'Max sharp'
        cv2.putText(frame, "{}: {:.4f}".format(text, self.max_fm), (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 3.8,
                    (0, 255, 0), 10)

        return frame


def test_sharpness():
    cap = cv2.VideoCapture('output2_mono8_7000.avi')
    ret, frame = cap.read()
    i = 1
    focus_measurer = FocusMeasurer()
    while ret:
        ret, frame = cap.read()
        if frame is None:
            print("End of video")
            break

        # blur img progressively
        ksize = (i * 2, i * 2)
        frame = cv2.blur(frame, ksize, cv2.BORDER_DEFAULT)

        frame = focus_measurer.add_focus_measure(frame)

        # show img
        cv2.imshow('Plate', frame)
        time.sleep(.5)

        i += 1
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def init_camera(width, height, pixel_format, acquisition_frame_rate):
    h = Harvester()

    cti_file_path = r'/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti'  # r'C:\Users\mdubinin\Downloads\mvGen\bin\x64\mvGenTLProducer.cti'
    print('cti file path', cti_file_path)
    h.add_file(cti_file_path)
    if os.path.exists(cti_file_path):
        print('File is exist')
    else:
        print('File is not exist')
    h.update()
    print('device', h.device_info_list)
    ia = h.create_image_acquirer(serial_number='0110027')  # 0110026
    ia.remote_device.node_map.Width.value = width
    ia.remote_device.node_map.Height.value = height
    ia.remote_device.node_map.PixelFormat.value = pixel_format
    ia.remote_device.node_map.AcquisitionLineRate = acquisition_frame_rate
    ia.remote_device.node_map.GevSCPD = 500
    ia.remote_device.node_map.NED_PRNUTarget = 100
    ia.remote_device.node_map.NED_AnalogGain = 'x200'
    ia.remote_device.node_map.NED_FFCMode = 'UserWhite'
    # ia.remote_device.node_map.GevSCPSPacketSize = 8500
    ia.start_acquisition()  # run_in_background=True)
    # time.sleep(5)
    # files = glob.glob('./plates/*')
    # for f in files:
    #     os.remove(f)

    return h, ia


def shutdown_camera(image_acquirer, harvester):
    image_acquirer.stop_image_acquisition()
    image_acquirer.destroy()
    harvester.reset()


def list_detector(frame1, frame, kernel):
    fgmask = cv2.absdiff(frame1, frame)
    fgmask = cv2.erode(fgmask, kernel, iterations=5)
    fgmask = cv2.dilate(fgmask, kernel, iterations=5)
    fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)[1]
    unique, counts = np.unique(fgmask, return_counts=True)
    df = {0: 0, 255: 0}
    for u, c in zip(unique, counts):
        df[u] = c

    return int(df[255] > 0.02 * df[0])


def run_save():
    try:
        h, ia = init_camera(width=4096, height=550,
                            pixel_format='Mono10', acquisition_frame_rate=2918)
        # time.sleep(1)
        focus_measurer = FocusMeasurer()
        plate_list = []
        # ====== list detector's_init ========
        frame1 = cv2.imread('1st_background_mono10.jpg', 0).astype('uint16')
        kernel = np.ones((2, 1), np.uint8)
        egami = None
        old_img = frame1
        # ====== end of list detector's_init ========
        i = 0
        while True:
            try:
                i += 1
                # time.sleep(0.05)
                with ia.fetch_buffer() as buffer:
                    payload = buffer.payload
                    if len(payload.components) < 1:
                        continue
                    component = payload.components[0]
                    print('component =', component)
                    width = component.width
                    height = component.height
                    data_format = component.data_format
                    if data_format in mono_location_formats:
                        content = component.data.reshape(height, width)
                    else:
                        # The image requires you to reshape it to draw it on the
                        # canvas:
                        if data_format in rgb_formats or \
                                data_format in rgba_formats or \
                                data_format in bgr_formats or \
                                data_format in bgra_formats:
                            #
                            content = component.data.reshape(
                                height, width,
                                int(component.num_components_per_pixel)  # Set of R, G, B, and Alpha
                            )
                            #
                            if data_format in bgr_formats:
                                # Swap every R and B:
                                content = content[:, :, ::-1]
                    if content.shape[0] == 0 or content.shape[1] == 0:
                        print('shape is 0')
                        # exit()
                        continue
                    # print("Image: ", './plates/content' + str(i) + '.jpg')

                    # cv2.imwrite('./plates/content' + str(i) + '.jpg', content)

                    flag = list_detector(frame1, content, kernel)
                    print('================', flag, egami is not None, '================')
                    # cv2.imwrite('plates/content' + str(i) + '.jpg', content)
                    if flag == 0 and egami is not None:
                        egami = cv2.vconcat([egami, content[:100]])
                        cv2.imwrite('plates/content' + str(i) + '.jpg', egami)
                        print('***** save *********')
                        egami = None

                    elif flag == 1 and egami is not None:
                        egami = cv2.vconcat([egami, content])
                        # cv2.imwrite('plates/content' + str(i) + '.jpg', egami)
                        # print(egami.shape)

                    elif flag == 1 and egami is None:
                        egami = content
                        egami = cv2.vconcat([old_img[-100:], content])

                    old_img = content
            except Exception as e:
                print('ERROR')
                print(e)
                continue
    except Exception as e:
        print('ERROR')
        print(e)
        pass
    finally:  # cv2.imwrite('./detector/plates/content' + str(i) + '.jpg', content)
        cv2.destroyAllWindows()  #

        shutdown_camera(ia, h)
    cv2.destroyAllWindows()

    shutdown_camera(ia, h)
    exit()


if __name__ == "__main__":
    run_save()
    # test_sharpness()

# pixel_location = component.represent_pixel_location()
# rgb_2d = np.zeros(shape=(height, width, 3), dtype='uint8')
