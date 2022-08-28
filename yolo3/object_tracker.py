# ================================================================
#
#   File name   : object_tracker.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : code to track detected object from video or webcam
#
# ================================================================
from deep_sort import generate_detections as gdet
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching
import time
from yolov3.configs import *
from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
import tensorflow as tf
import numpy as np
import cv2
import os
from openpyxl import Workbook
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

video_path = "../data/video/hackathon_5.mp4"
map_image_path = '../data/picture/map_hackathon_5.png'
save_pos_bool = False # True면 계산뒤에 poistion값을 저장합니다
position_excel_output_path = 'hackathon_5.xlsx'
output_tracking_video = '' #이름뒤에 .mp4를 붙이시길 바랍니다 ex -> 'output.mp4' 비우시면 저장이 따로되지 않습니다 
output_position_video = '' #이름뒤에 .mp4를 붙이시길 바랍니다 ex -> 'output.mp4' 비우시면 저장이 따로되지 않습니다 
show_frame = False #True면 frame당 이미지를 보여줍니다


position = []
def Object_tracking(Yolo, video_path, output_path, output_path_map, input_size=YOLO_INPUT_SIZE, show=False, save_pos = False,CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', Track_only=[]):
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None

    # initialize deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    times, times_2 = [], []

    if video_path:
        vid = cv2.VideoCapture(video_path)  # detect on video
    else:
        vid = cv2.VideoCapture(0)  # detect from webcam

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    # output_path must be .mp4
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))
    out_map = cv2.VideoWriter(output_path_map, codec, fps, (width, height))

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys())
    val_list = list(NUM_CLASS.values())
    while True:
        _, frame = vid.read()

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break

        image_data = image_preprocess(np.copy(original_frame), [
                                      input_size, input_size])
        #image_data = tf.expand_dims(image_data, 0)
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)

        #t1 = time.time()
        #pred_bbox = Yolo.predict(image_data)
        t2 = time.time()

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(
            pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(Track_only) != 0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(
                    int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(NUM_CLASS[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes)
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox,
                      score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []

        for track in tracker.tracks:
            # print(track)
            one_position = []
            if not track.is_confirmed() or track.time_since_update > 5:
                continue
            bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
            class_name = track.get_class()  # Get the class name of particular object
            tracking_id = track.track_id  # Get the ID for the particular track
            # Get predicted object index by object name
            index = key_list[val_list.index(class_name)]
            # Structure data, that we could use it with our draw_bbox function
            x = (bbox.tolist()[0]+bbox.tolist()[2])/2
            # y = (bbox.tolist()[1]+bbox.tolist()[3])/2
            y = (bbox.tolist()[3]-bbox.tolist()[1])*(5/6)+bbox.tolist()[1]
            # (y2-y1)*(5/6)+y1)
            a = np.array([x, y, 1])
            ipm_matrix = np.array(ipm_matrix)
            at = np.matmul(ipm_matrix, a)
            new_x = at[0]/at[2]
            new_y = at[1]/at[2]
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index])
            one_position.append(new_x)
            one_position.append(new_y)
            one_position.append(tracking_id)
            position.append(one_position)
        # print("전체 포지션 : ", position)

        # draw detection on frame
        map_image = cv2.imread(map_image_path)
        
        #hackathon5
        pts = np.array([[0, 510], [1036, 264], [1198, 719],
                [1280, 317]], dtype=np.float32)
        ipm_pts = np.array([[100, 100], [1080, 100], [100, 620],
                            [1080, 620]], dtype=np.float32)

        ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)

        ipm_image = cv2.warpPerspective(
            map_image, ipm_matrix, map_image.shape[:2][::-1])

        image, ipm = draw_bbox(original_frame, ipm_image, ipm_matrix, tracked_bboxes,
                               CLASSES=CLASSES, tracking=True)

        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)

        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)

        image = cv2.putText(image, "Time: {:.1f} FPS".format(
            fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        # draw original yolo detection
        #image = draw_bbox(image, bboxes, CLASSES=CLASSES, show_label=False, rectangle_colors=rectangle_colors, tracking=True)

        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(
            ms, fps, fps2))
        if output_path_map != '':
            out.write(image)
            out_map.write(ipm)
        if show:
            image = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_LINEAR)
            cv2.imshow('output', image)
            ipm = cv2.resize(ipm, dsize=(0, 0), fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_LINEAR)
            cv2.imshow('ipm', ipm)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
    if save_pos:

        df = pd.DataFrame(position)
        df.to_excel(position_excel_output_path, index=False)
        cv2.destroyAllWindows()


yolo = Load_Yolo_model()

Object_tracking(yolo, video_path, output_tracking_video,output_position_video , input_size=YOLO_INPUT_SIZE,
                show=show_frame, save_pos = save_pos_bool,iou_threshold=0.1, rectangle_colors=(255, 0, 0), Track_only=["person"])
