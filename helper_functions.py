import cv2
import numpy as np
from vector import distance

def convert_output(labels, n):
    output = np.zeros((len(labels), n), dtype='int')
    labels_list = np.array(list(enumerate(labels)))
    output[labels_list[:, 0], labels_list[:, 1]] = 1

    return output

def find_line(frame):
    lower = np.array([230, 0, 0])
    upper = np.array([255, 255, 255])

    new_frame = cv2.inRange(frame, lower, upper)

    lines = cv2.HoughLinesP(new_frame, 1, np.pi / 180, 100, minLineLength=5, maxLineGap=20)

    lower_x = 0
    lower_y = 0
    upper_x = 0
    upper_y = 0

    for x1, y1, x2, y2 in lines[0]:
        lower_x = x1
        lower_y = y1
        upper_x = x2
        upper_y = y2

    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            if x1 < lower_x:
                lower_x = x1
                lower_y = y1
            if x2 > upper_x:
                upper_x = x2
                upper_y = y2

    return lower_x, lower_y, upper_x, upper_y

def in_range(r, item, items):
    ret_val = []
    for obj in items:
        min_distance = distance(item['center'], obj['center'])
        if min_distance < r:
            ret_val.append(obj)
    return ret_val