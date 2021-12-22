"""
Detector microservice initialization.
"""

from datetime import datetime
import sys
import cv2
import numpy as np
# import pycuda.autoinit  # This is needed for initializing CUDA driver

import utils.utils as utils
# from utils.camera_multiprocess import Camera
from utils.camera import Camera
from utils.mjpeg import MjpegServer
from plate_analyzer.defects_detect import detector, tresh_detector, crop_plate, crop_plate2, defect_overlaying, \
    compute_plate_shape

from plate_analyzer import depth_estimation

# from segmenter.yolact import init_segmenter_yolact
from backend_sender.api import create_api, APCS_communicator
from plate_analyzer.plate_existence_detector import plate_existence_detector


def loop_and_detect(cam, segmenter, backend_api, conf_th, mjpeg_server, im_saver, args, logger):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
      mjpeg_server
    """
    fps_measurer = utils.FpsMeasurer()
    apcs_communicator = APCS_communicator(backend_api=backend_api)

    i = 0
    past_img = None
    conveyor_speed = 0
    plate_exist_detector = plate_existence_detector(args['camera']['ned']['list_detector_k'],
                                                    cam.get_background())

    while cam.isOpened():
        img_id = None
        img = cam.read()
        if img is None:
            continue

        fps_measurer.start()
        concated_plate, state = plate_exist_detector.state_of_existence(past_img, img)

        # Plate ended, now send it
        if state == 3:
            concated_plate = cv2.resize(concated_plate, (0, 0), fx=0.2, fy=0.2)
            cropped_plate, masks, image_with_masks = defect_overlaying(concated_plate.astype('uint8'),
                                                                       int(args['camera']['ned']['step_colomns']),
                                                                       int(args['camera']['ned']['crop_tresh']),
                                                                       int(args['camera']['ned']['tresh_up']),
                                                                       int(args['camera']['ned']['tresh_down']),
                                                                       int(args['camera']['ned']['scale_tresh']),
                                                                       int(args['camera']['ned']['color_tresh']))
            if len(masks) >= 2:
                im_saver.append(cv2.hconcat([concated_plate, image_with_masks]))
                
            if image_with_masks is not None:
                img_to_send = cv2.imencode(".bmp", image_with_masks)[1].tobytes()

            else:
                cropped_plate = None
                continue

            depths = depth_estimation.estimate_depth(cropped_plate, masks)

            # sending image to image keeper microservice
            img_id = backend_api.store_image(img_to_send)

            # yolact segmenter inference
            # preds = segmenter.predict(img, False)

            # Getting data from APCS microservice
            apcs_parameters = apcs_communicator.get_parameters(timeout=5)
            logger.info('APCS Parameters' + str(apcs_parameters))

            # change FrameRate if conveyor speed changed
            if args['camera']['type'] == 'file':
                pass

            elif float(apcs_parameters['speed']) != conveyor_speed and float(apcs_parameters['speed']) != 0.:
                conveyor_speed = float(apcs_parameters['speed'])
                cam.change_framerate_from_speed(conveyor_speed)

            product_type = '-'.join([str(apcs_parameters['board_wickness']),
                                     str(apcs_parameters['board_width']),
                                     str(apcs_parameters['board_length']),
                                     str(apcs_parameters['board_type']),
                                     str(apcs_parameters['board_edge']),
                                     ])

            # send collected metadata about stored plate image
            if img_id is not None:
                list_width, list_length = compute_plate_shape(cropped_plate)
                # logger.info('MASKS:' + str(masks))

                time_created_at = datetime.now()
                apcs_communicator.send_parameters(img_id=1, time_created_at=time_created_at,
                                                  defects={'A': 5, 'B': 10, 'C': 15})
                backend_api.store_plate(image_object=img_id,
                                        depths=depths,
                                        defects=masks,
                                        size=(list_width, list_length),
                                        # size=(apcs_parameters['board_width'], apcs_parameters['board_length']),
                                        created_at=time_created_at,
                                        tags={'line': args['conveyor_line_number'], 'product_type': product_type})

            logger.info('Plate detected and sent')
            concated_plate = None

        # Plate in process, concatenating
        elif state == 2:
            if concated_plate.shape[0] > 10000:
                logger.info('List over 10000 height')
                concated_plate = None
                continue

        # Plate started, create new plate
        elif state == 1:
            apcs_communicator.start_get_parameters()
            logger.info('List start detected')
            # resp_future = backend_api.send_arbitrary_request(url='', method='GET')

        past_img = img
        im_saver.append(img)

        fps_measurer.end()

        # mjpeg_server.send_img(img)
        if i % 30 == 0:
            logger.info('FPS: ' + fps_measurer.fps())

        i += 1


def main():
    args = utils.parse_args(config_path='')
    logger = utils.setup_logging('')

    backend_api = create_api(base_url='http://')
    # backend_api = create_api(base_url='https://knauf.facemetric.ru')
    logger.info('Backend API successfully initialized')

    # logger.info('Starting Segmentation Model initialization...')
    segmenter_model = None  # init_segmenter_yolact(args)
    # logger.info('Segmentation Model successfully initialized')

    mjpeg_server = None
    mjpeg_server = MjpegServer(port=int(args['mjpeg_port']))
    logger.info('MJPEG Server initialized successfully at port: ' + str(args['mjpeg_port']))

    cam = Camera(args['camera'])
    if not cam.isOpened():
        logger.error('Failed to open camera!')
        raise SystemExit('ERROR: failed to open camera!')
    else:
        logger.info('Camera successfully initialized')

    im_saver = utils.ImagesSaver(args['output_dir'] + args['camera']['ned']['serial_number'])

    logger.info('Detection System successfully initialized')
    try:
        with utils.GracefulInterruptHandler(cam, mjpeg_server, backend_api, logger, im_saver):
            loop_and_detect(cam, segmenter=segmenter_model, backend_api=backend_api, conf_th=0.8,
                            mjpeg_server=mjpeg_server,
                            im_saver=im_saver,
                            args=args,
                            logger=logger)
    except Exception as e:
        logger.exception(e)
    finally:
        logger.warning('The system shuts down gracefully')
        im_saver.release()
        cam.release()
        # mjpeg_server.shutdown()
        # backend_api.stop()
        logger.warning('The system has shut down gracefully')
        return 0


if __name__ == '__main__':
    main()
    sys.exit()
    exit()
