import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

from skimage.io import imread
from scipy import ndimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from keras.models import load_model

from vector import pnt2line
import helper_functions as hf
import ann

kernel = np.ones((2, 2), np.uint8)
lower = np.array([230, 230, 230])
upper = np.array([255, 255, 255])

counter = 0
elements = []
crossing = []
moments = []

if __name__ == "__main__":
    file_path = 'videos/video-0.avi'
    video = cv2.VideoCapture(file_path)

    videoOn = True
    t = 0
    while videoOn:
        videoOn, img = video.read()

        if t == 0:
            x1, y1, x2, y2 = hf.find_line(img)
            line = [(x1, y1), (x2, y2)]

        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(img, lower, upper)
        img0 = 1.0 * mask

        img0 = cv2.dilate(img0, kernel)
        img0 = cv2.dilate(img0, kernel)

        file_name = 'images/frame-' + str(t) + '.png'
        cv2.imwrite(file_name, img0)

        labeled, nr_objects = ndimage.label(img0)
        objects = ndimage.find_objects(labeled)

        for i in range(nr_objects):
            loc = objects[i]
            (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                        (loc[0].stop + loc[0].start) / 2)
            (dxc, dyc) = ((loc[1].stop - loc[1].start),
                          (loc[0].stop - loc[0].start))

            if dxc > 11 or dyc > 11:
                cv2.circle(img, (xc, yc), 16, (25, 25, 255), 1)
                elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}

                lst = hf.in_range(20, elem, elements)
                nn = len(lst)
                if nn == 0:
                    elem['t'] = t
                    elem['pass'] = False
                    elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]
                    elem['future'] = []

                    elements.append(elem)
                elif nn == 1:
                    lst[0]['center'] = elem['center']
                    lst[0]['t'] = t
                    lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t})
                    lst[0]['future'] = []

        for el in elements:
            tt = t - el['t']
            if tt < 3:
                dist, pnt, r = pnt2line(el['center'], line[0], line[1])
                c = (25, 25, 255)
                if r > 0:
                    cv2.line(img, pnt, el['center'], (0, 255, 25), 1)
                    c = (25, 25, 255)
                    if dist < 9:
                        c = (0, 255, 160)
                        if el['pass'] is False:
                            el['pass'] = True
                            crossing.append(el)
                            moments.append(t)
                            counter += 1

                cv2.circle(img, el['center'], 16, c, 2)

                for history in el['history']:
                    ttt = t - history['t']
                    if ttt < 100:
                        cv2.circle(img, history['center'], 1, (0, 255, 255), 1)

                for future in el['future']:
                    ttt = future[0] - t
                    if ttt < 100:
                        cv2.circle(img, (future[1], future[2]), 1, (255, 255, 0), 1)

        cv2.putText(img, 'Counter: ' + str(counter), (450, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)
        t += 1

        cv2.imshow('frame', img)
        file_name = 'images/frame-' + str(t) + '.png'
        cv2.imwrite(file_name, img)
        k = cv2.waitKey(30) & 0xff

        if t == 1199:
            break

    video.release()
    cv2.destroyAllWindows()

    z = 0
    extracted = []
    for el in crossing:
        for history in el['history']:
            if history['t'] + 4 < t:
                if history['t'] + 4 == moments[z]:
                    element = {'center': history['center'], 't': history['t']}
        z += 1
        extracted.append(element)

    path = 'images/frame-'
    o = 0
    for el in extracted:
        new_path = path + str(el['t']) + '.png'
        print new_path
        xi, yi = el['center']
        xi1 = xi - 14
        yi1 = yi - 14
        xi2 = xi + 14
        yi2 = yi + 14
        picture = imread(new_path)
        new_picture = picture[yi1: yi2, xi1: xi2]
        put_path = 'numbers/num-' + str(o) + '.png'
        cv2.imwrite(put_path, new_picture)
        # plt.imshow(new_picture)
        # plt.show()
        o += 1

    # ann.initialize_ann()
    ann = load_model('ann.obj')

    sum = 0
    for index in range(0, len(crossing)):
        path = 'numbers/num-' + str(index) + '.png'
        num_image = imread(path)
        prediction = ann.predict(num_image.reshape(1, 784), verbose=1)
        value = np.argmax(prediction)
        print "Value: " + str(value)
        sum += value

    print "Sum: " + str(sum)