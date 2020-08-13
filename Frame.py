import time

import SupportFunction as sf

import cv2
import numpy as np
import os

marker_folder = os.path.join(os.getcwd(), "Marker")
file_name_marker = [
    '1.jpg', '2.jpg', '3.jpg', '4.jpg'
]
SCALE_PERCENT = 75 / 100  # percent of original size
MARKER_RESIZE = 504  # 24 * 21 | 20 * 21

MARK_IMAGE_1 = cv2.imread(
    marker_folder.join('/' + file_name_marker[0]), 1)
MARK_IMAGE_2 = cv2.imread(
    marker_folder.join('/' + file_name_marker[1]), 1)
MARK_IMAGE_3 = cv2.imread(
    marker_folder.join('/' + file_name_marker[2]), 1)
MARK_IMAGE_4 = cv2.imread(
    marker_folder.join('/' + file_name_marker[3]), 1)

ID_0 = [
    39, 219, 265, 334
]

ID = {
    0: MARK_IMAGE_1,
    1: MARK_IMAGE_2,
    2: MARK_IMAGE_3,
    3: MARK_IMAGE_4
}

IDX = {
    '34': (1, 2),
    '43': (1, 2),
    '40': (2, 3),
    '04': (2, 3),
    '01': (3, 4),
    '10': (3, 4),
    '12': (4, 0),
    '21': (4, 0),
    '23': (0, 1),
    '32': (0, 1)
}

POSITION = [
    'LTRB', 'BLTR', 'RBLT', 'TRBL'
]

COUNT = [0 for i in range(6)]


def kill_small_contour(contour, area):
    x, y, w, h = cv2.boundingRect(contour)
    return True if w * h > area else False


def process(frame):
    time_start_1 = time.time()

    gray = cv2.cvtColor(src=frame,
                        code=cv2.COLOR_BGR2GRAY)

    gray = cv2.morphologyEx(src=gray,
                            op=cv2.MORPH_OPEN,
                            kernel=cv2.getStructuringElement(
                                shape=cv2.MORPH_RECT,
                                ksize=(3, 3)))

    threshold = cv2.adaptiveThreshold(src=gray, maxValue=255,
                                      adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                      thresholdType=cv2.THRESH_BINARY_INV,
                                      blockSize=11, C=7)

    threshold = cv2.medianBlur(src=threshold, ksize=3)

    contours, hierarchy = cv2.findContours(image=threshold,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)

    contours = [contour for contour in contours if
                kill_small_contour(contour, (
                        frame.shape[0] * frame.shape[1]) / 35) is True]

    time_end_1 = time.time() - time_start_1
    print(f"step 1: {time_end_1}")
    # BUG-1
    # cv2.imshow("barrrr", threshold)

    time_start_2 = time.time()
    cv2.drawContours(frame, contours, -1, (0, 255, 128), 2, 8)

    list_approxs = []
    for i in range(len(contours)):
        contour = contours[i]
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        hull = cv2.convexHull(approx)

        isPoly = False
        points = [list(_contour[0]) for _contour in hull]
        if len(hull) == 5:
            Edges = [(sf.euclid_distance(points[0], points[1]), 0),
                     (sf.euclid_distance(points[1], points[2]), 1),
                     (sf.euclid_distance(points[2], points[3]), 2),
                     (sf.euclid_distance(points[3], points[4]), 3),
                     (sf.euclid_distance(points[4], points[0]), 4)]
            # cv2.drawContours(frame, [hull], -1, (0, 0, 255), 3, 8)

            Edges.sort(key=lambda x: x[0])
            if abs(Edges[-1][1] - Edges[-2][1]) != 1 or not (
                    max(Edges[-1][1], Edges[-2][1]) != 4 and min(Edges[-1][1],
                                                                 Edges[-2][
                                                                     1]) != 0):
                continue

            key = str(Edges[-1][1]) + str(Edges[-2][1])
            point_A = points[IDX[key][0]]
            point_B = points[IDX[key][1]]
            point_C = points[IDX[key][0] - 1]
            point_D = points[IDX[key][1] + 1 if IDX[key][
                                                    1] + 1 < 5
                                                    else 0]
            key_point = sf.get_key_point(point_A, point_B, point_C, point_D)
            if key_point is None:
                continue
            if max(Edges[-1][1], Edges[-2][1]) == 4 and min(Edges[-1][1],
                                                            Edges[-2][1]) == 0:
                points.pop(0)
                points.pop(-1)
                points.append(list(key_point))
            else:
                points.pop(IDX[key][0])
                if IDX[key][0] == len(points):
                    points.pop(0)
                else:
                    points.pop(IDX[key][0])
                points.insert(IDX[key][0], list(key_point))

            hull = cv2.convexHull(np.array(
                [points[0], points[1], points[2], points[3]]))
            # print(hull)
            # list_approxs.append((hull, points))
            isPoly = True
            # print(points)
            # cv2.circle(frame, tuple(key_point), 5, (0, 0, 255), cv2.FILLED)
            # cv2.drawContours(frame, [hull], -1, (0, 0, 0), 2, 8)

        if len(hull) == 4 or isPoly:
            if isPoly and len(hull) != 4:
                continue
            # cv2.drawContours(frame, [hull], -1, (0, 0, 255), 3, 8)
            avg_length = (sf.euclid_distance(points[0], points[1])
                          + sf.euclid_distance(points[1], points[2])
                          + sf.euclid_distance(points[2], points[3])
                          + sf.euclid_distance(points[0], points[3])
                          ) / 4
            min_threshold, max_threshold = avg_length * 0.7, avg_length * 1.3
            square_edges = [sf.euclid_distance(points[0], points[1]),
                            sf.euclid_distance(points[1], points[2]),
                            sf.euclid_distance(points[2], points[3]),
                            sf.euclid_distance(points[0], points[3])]
            # print(
            #     f"{i} got {square_edges} {min_threshold}, {max_threshold}, "
            #     f"{avg_length}")

            bool_edges = np.logical_not(
                np.bitwise_and(min_threshold < np.array(square_edges),
                               np.array(square_edges) < max_threshold))
            fl = not (np.any(bool_edges))

            if fl:
                # cv2.drawContours(frame, [hull], -1, (255, 0, 0), 3, 8)
                list_approxs.append((hull, points))
            else:
                dai = max(square_edges[0], square_edges[1])
                rong = min(square_edges[0], square_edges[1])
                if dai > rong * 1.7:
                    continue
                theta_1 = sf.get_theta(points[0], points[1], points[0],
                                        points[-1])
                theta_2 = sf.get_theta(points[2], points[1], points[2],
                                        points[-1])
                if theta_1 is None or theta_2 is None:
                    continue
                theta = theta_2 + theta_1
                if (abs(square_edges[0] - square_edges[2]) <= 10.0 or abs(
                        square_edges[1] - square_edges[
                            -1]) <= 10.0) and avg_length >= 50.0 \
                        and (theta < 175.0 or theta > 185.0):
                    list_approxs.append((hull, points))

        del hull

    time_end_2 = (time.time() - time_start_2)
    print(f"step 2 : {time_end_2}")
    warped_images = []
    return warped_images


def find_markers(frame):
    time_start_o = time.time()
    warped_images = process(frame)
    time_elapsed_o = (time.time() - time_start_o)
    print(f"detect object took {time_elapsed_o}")

    return frame


def imgProcess():
    image = cv2.imread("Data/147.jpg")

    width = int(image.shape[1] * SCALE_PERCENT)
    height = int(image.shape[0] * SCALE_PERCENT)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    time_start = time.time()
    image = find_markers(image)
    time_elapsed = (time.time() - time_start)

    print(f"process {image.shape} took {time_elapsed}")

    cv2.imshow("result.jpg", image)
    cv2.waitKey(0)


def videoProcess():
    # name = 'camera_1.avi'
    # name = 'Data\\Video\\camerac2.avi'
    # cap = cv2.VideoCapture(name)
    cap = cv2.VideoCapture(0)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    # frame_width, frame_height = 960, 540
    # out = cv2.VideoWriter('2nrs_out.avi', cv2.VideoWriter_fourcc('M',
    #                                                                'J', 'P',
    #                                                                'G'), 10,
    #                       (frame_width, frame_height))
    frame_count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break

        x = frame.copy()
        frame_count += 1
        time_start = time.clock()

        frames = find_markers(frame)

        time_elapsed = (time.clock() - time_start)
        print(f"frame {frame_count} {frame.shape} took : {time_elapsed}")

        # result = np.hstack((x, frames))
        cv2.imshow("frame", frames)
        # # out.write(frames)
        cv2.imwrite("frames\\" + str(frame_count) + '.jpg', x)
        cv2.imwrite("framesout\\" + str(frame_count) + '.jpg', frames)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame_count += 1
    print(frame_count)
    # cv2.waitKey(0)
    # When everything done, release the capture
    # out.release()
    cap.release()


if __name__ == '__main__':
    # imgProcess()
    videoProcess()
    cv2.destroyAllWindows()
