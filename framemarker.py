import math
import operator
import sys
import time
from operator import itemgetter
from shapely.geometry import Polygon

import cv2
import numpy as np

SCALE_PERCENT = 50 / 100  # percent of original size
MARKER_RESIZE = 504  # 24 * 21 | 20 * 21
MARK_IMAGE_1 = cv2.imread(
    "C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\image\\guy.jpg", 1)
MARK_IMAGE_2 = cv2.imread(
    "C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\image\\stewie.jpg",
    1)
MARK_IMAGE_3 = cv2.imread(
    "C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\image\\guy.jpg", 1)
MARK_IMAGE_4 = cv2.imread(
    "C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\image\\haha.jpg",
    1)
MARK_IMAGE_5 = cv2.imread(
    "C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\image\\tinyguy"
    ".jpg",
    1)
ID_0 = [
    39, 219, 265, 334
]

ID = {
    18: MARK_IMAGE_1,
    0: MARK_IMAGE_2,
    1: MARK_IMAGE_3,
    2: MARK_IMAGE_4,
    3: MARK_IMAGE_5
}

INDEX = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0)
]

POSITION = [
    'LTRB', 'BLTR', 'RBLT', 'TRBL'
    # ['BLTR', 'LTRB', 'TRBL', 'RBLT'],
    # ['RBLT', 'BLTR', 'LTRB', 'TRBL'],
    # ['TRBL', 'RBLT', 'BLTR', 'LTRB']
]

COUNT = [0 for i in range(6)]


def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


def bit_xor(a_list, b_list):
    return [] if len(a_list) != len(b_list) \
        else [a_list[index] ^ b_list[index] for index in range(len(a_list))]


def EuclidDistance(point_1, point_2):
    return math.sqrt(
        (point_2[1] - point_1[1]) ** 2 + (point_2[0] - point_1[0]) ** 2)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1],
                            flags=cv2.INTER_LINEAR)
    return result


def binaryToDecimal(binary):
    decimal, i, n = 0, 0, 0
    while binary != 0:
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary // 10
        i += 1
    return decimal


def get_linear_equations(pointA, pointB):
    x1, y1, x2, y2 = pointA[0], pointA[1], pointB[0], pointB[1]
    if x2 - x1 == 0:
        return None, None
    A = (y2 - y1) / (x2 - x1)
    B = y1 - A * x1
    return A, B


def get_key_point(point_A, point_B, point_C, point_D):
    A1, B1 = get_linear_equations(point_A, point_C)
    A2, B2 = get_linear_equations(point_B, point_D)
    if A1 is None or B1 is None or A2 is None or B2 is None:
        return None
    X = (B1 - B2) / (A2 - A1)
    Y = A1 * X + B1
    return int(X), int(Y)


def process(frame):
    time_start_1 = time.time()

    gray = cv2.cvtColor(src=frame,
                        code=cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT,
                                       ksize=(5, 5))

    gray = cv2.morphologyEx(src=gray,
                            op=cv2.MORPH_OPEN,
                            kernel=kernel)

    threshold = cv2.adaptiveThreshold(src=gray, maxValue=255,
                                      adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                      thresholdType=cv2.THRESH_BINARY_INV,
                                      blockSize=29, C=11)

    threshold = cv2.medianBlur(src=threshold, ksize=5)
    # edges = cv2.Canny(image=threshold,
    #                   threshold1=50,
    #                   threshold2=200)
    # cv2.imshow("barrrr", threshold)
    contours, hierarchy = cv2.findContours(image=threshold,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)

    time_end_1 = time.time() - time_start_1
    # print(f"step 1: {time_end_1}")

    time_start_2 = time.time()

    contours = [contour for contour in contours if
                cv2.contourArea(contour) > 750.0]

    list_approxs = []
    for i in range(len(contours)):
        contour = contours[i]
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        hull = cv2.convexHull(approx)
        points = []
        if len(hull) == 5:
            points = [list(_contour[0]) for _contour in hull]
            Edges = [(EuclidDistance(points[0], points[1]), 0),
                     (EuclidDistance(points[1], points[2]), 1),
                     (EuclidDistance(points[2], points[3]), 2),
                     (EuclidDistance(points[3], points[4]), 3),
                     (EuclidDistance(points[4], points[0]), 4)]
            Edges.sort(key=lambda x: x[0])
            if abs(Edges[-1][1] - Edges[-2][1]) != 1 or (
                    max(Edges[-1][1], Edges[-2][1]) != 4 and min(Edges[-1][1],
                                                                 Edges[-2][
                                                                     1]) != 0):
                continue
            point_A = points[INDEX[Edges[0][1]][0]]
            point_B = points[INDEX[Edges[0][1]][1]]
            point_C = points[INDEX[Edges[0][1]][0] - 1]
            point_D = points[INDEX[Edges[0][1]][1] + 1 if INDEX[Edges[0][1]][
                                                              1] + 1 < 5
            else 0]
            key_point = get_key_point(point_A, point_B, point_C, point_D)
            if key_point is None:
                continue
            if max(Edges[-1][1], Edges[-2][1]) == 4 and min(Edges[-1][1],
                                                            Edges[-2][1]) == 0:
                points.pop(0)
                points.pop(-1)
                points.append(list(key_point))
            else:
                points.pop(INDEX[Edges[0][1]][0])
                if INDEX[Edges[0][1]][0] == len(points):
                    points.pop(0)
                else:
                    points.pop(INDEX[Edges[0][1]][0])
                points.insert(INDEX[Edges[0][1]][0], list(key_point))
            new_hull = cv2.convexHull(np.array(
                    [points[0], points[1], points[2], points[3]]))
            list_approxs.append((new_hullq, points))
            # print(points)
            # cv2.circle(frame, tuple(key_point), 5, (0, 0, 255), cv2.FILLED)
            cv2.drawContours(frame, [points], -1, (0, 0, 0), 2, 8)

        if len(hull) == 4 and (
                cv2.contourArea(hull) - cv2.contourArea(approx) <= 100.0):
            points = [list(_contour[0]) for _contour in hull]

            avg_length = (EuclidDistance(points[0], points[1])
                          + EuclidDistance(points[1], points[2])
                          + EuclidDistance(points[2], points[3])
                          + EuclidDistance(points[0], points[3])
                          ) / 4
            min_threshold, max_threshold = avg_length * 0.8, avg_length * 1.2
            square_edges = [EuclidDistance(points[0], points[1]),
                            EuclidDistance(points[1], points[2]),
                            EuclidDistance(points[2], points[3]),
                            EuclidDistance(points[0], points[3])]
            bool_edges = np.logical_not(
                np.bitwise_and(min_threshold < np.array(square_edges),
                               np.array(square_edges) < max_threshold))
            fl = not (np.any(bool_edges))

            if fl:
                list_approxs.append((hull, points))

        del points
        del hull

    approx_list = {}
    rm = []
    if len(list_approxs) == 1:
        approx_list[0] = list_approxs[0]
    for i in range(len(list_approxs)):
        for j in range(i + 1, len(list_approxs)):
            iou = calculate_iou(list_approxs[j][1], list_approxs[i][1])
            area_1, area_2 = cv2.contourArea(
                list_approxs[i][0]), cv2.contourArea(list_approxs[j][0])
            # print(iou, area_1, area_2)
            if iou > 0.00 and abs(area_1 - area_2) <= max(area_2, area_1) / 3:
                index = i if area_1 > area_2 else j
                if index in rm:
                    break
                else:
                    idx = i if i != index else j
                    rm.append(idx)

                approx_list[index] = list_approxs[index]
    del rm

    list_approxs.clear()
    for key, value in approx_list.items():
        # cv2.drawContours(frame, [value[0]], -1, (0, 0, 255), 3, 8)
        list_approxs.append(value)

    del approx_list
    time_end_2 = (time.time() - time_start_2)
    # print(f"{time_end_2}")

    time_start_3 = time.time()
    warped_images = []
    # print("End", len(list_approxs))
    for contour, points in list_approxs:
        points = np.float32(points)

        warped_points = np.float32(
            [[MARKER_RESIZE, 0],
             [MARKER_RESIZE, MARKER_RESIZE],
             [0, MARKER_RESIZE],
             [0, 0]]
        )
        M = cv2.getPerspectiveTransform(points, warped_points)
        warped = cv2.warpPerspective(src=frame, M=M,
                                     dsize=(MARKER_RESIZE, MARKER_RESIZE),
                                     flags=cv2.INTER_LINEAR)
        # cv2.drawContours(frame, [contour], -1, (0, 0, 0), cv2.FILLED)

        warped_images.append((warped, points, contour))
    time_end_3 = time.time() - time_start_3
    # print(f"{time_end_3}")
    return warped_images


def edge_process(image):
    ret, border = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    code = ""
    area = 25 * 20
    if border is None or border.shape[1] < 300:
        return "0"

    index = 0
    for k in range(round(border.shape[1] / 20), border.shape[1],
                   round(border.shape[1] / 20) * 2):
        # cv2.rectangle(image, (k, 0), (k + 24, border.shape[1]), (128, 128,
        # 128), cv2.FILLED)
        x = "0" if np.sum(border[0:k, k:min(k + 24, border.shape[
            1])]) / 255 > area * 0.4 else "1"
        code += x
        # if np.sum(border[0:k, k:min(k + 24, border.shape[1])]) / 255 >
        # area * 0.3:
        #     code += "0"
        # else:
        #     code += "1"
        index += 1
        if index == 9:
            break
    return code if len(code) == 9 else "0"


def marker_process(image, index):
    size = image.shape[0]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean, std = cv2.meanStdDev(gray)
    ret, threshold = cv2.threshold(gray, int(mean), 255, cv2.THRESH_BINARY)

    # edges = cv2.Canny(image=threshold,
    #                   threshold1=50,
    #                   threshold2=200)
    # cv2.imshow("haizz", threshold)
    contours, hierarchy = cv2.findContours(image=threshold,
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)

    m, box = 0, (0, 0, size, size)
    for contour in contours:
        _box = cv2.boundingRect(contour)
        area = _box[2] * _box[3]
        if area > m:
            m = area
            box = _box

    border = 3
    box = list(box)
    box[2], box[3] = max(box[2], box[3]), max(box[2], box[3])

    img = threshold[box[1] + border:box[1] + box[3] - border,
          box[0] + border:box[0] + box[2] - border]
    # if index == 2:
    #     cv2.imshow(str(index) + "!2321312thr.jpg", threshold)

    # img = threshold[int(size / 12):size - int(size / 20),
    #       int(size / 12):size - int(size / 20)]
    size = img.shape[0]

    edges_image = [
        cv2.rotate(img[0:size, 0:20], cv2.ROTATE_90_CLOCKWISE),
        img[0:20, 0:size],
        cv2.rotate(img[0:size, size - 20:size],
                   cv2.ROTATE_90_COUNTERCLOCKWISE),
        cv2.rotate(img[size - 20:size, 0:size], cv2.ROTATE_180)
    ]
    mats = [binaryToDecimal(int(edge_process(edges_image[i]))) for i in
            range(4)]
    # if index == 2:
    #     print(mats)
    #     cv2.imshow('asdas', img)
    #     [cv2.imshow(str(i), edges_image[i]) for i in range(4)]

    # [cv2.imshow(str(i), edges_image[i]) for i in range(4)]
    # cv2.imshow(str(index) + "!2321312", img)
    # cv2.waitKey(0)
    # exit(0)
    key = ""
    position = ""

    print(mats)
    if sum(mats) == 511 * 4 or sum(mats) == 0:
        return key, position, threshold

    mm = 1
    for idx in range(4):
        xor_list = bit_xor(mats, ID_0)
        counts = {}
        for i in xor_list:
            counts[i] = counts.get(i, 0) + 1
        _key = max(counts, key=counts.get)
        if _key in ID and counts[_key] > mm:
            mm = counts[_key]
            key, position = _key, POSITION[idx]

        mats.append(mats.pop(0))
        del counts

    del mats
    return key, position, threshold


def post_processing(frame, warped_images):
    if not warped_images:
        COUNT[0] += 1
        return frame
    COUNT[5] += len(warped_images)
    # print("bug", len(warped_images))
    result = frame.copy()
    # print("len", len(warped_images))
    flag = True
    ssmm = 0.0
    for i, (warped, warped_points, contour) in enumerate(warped_images):
        time_start_1 = time.time()
        key, position, threshold = marker_process(warped, i)
        time_end_1 = (time.time() - time_start_1)
        # print(f"{time_end_1}")
        ssmm += time_end_1
        # print(f"{i} got key {key}")
        if key == "":
            # cv2.waitKey(0)
            # exit(0)
            COUNT[2] += 1
            flag = False
            continue
        if key in ID:
            COUNT[1] += 1
            replace_image = ID[key]
            replace_image_size = replace_image.shape[0]
            if position == "LTRB":
                pass
            elif position == "BLTR":
                replace_image = rotate_image(replace_image, 270)
            elif position == "RBLT":
                replace_image = rotate_image(replace_image, 180)
            elif position == "TRBL":
                replace_image = rotate_image(replace_image, 90)
            else:
                print("ERROR", key, position)
                exit(0)
            points = np.float32(
                [[replace_image_size, 0],
                 [replace_image_size, replace_image_size],
                 [0, replace_image_size],
                 [0, 0]]
            )
            M = cv2.getPerspectiveTransform(points, warped_points)
            warped_image = cv2.warpPerspective(replace_image, M, (
                frame.shape[1], frame.shape[0]))

            cv2.drawContours(result, [contour], -1, (0, 0, 0), cv2.FILLED)
            result = cv2.bitwise_xor(result, warped_image, mask=None)

        else:
            flag = False
            print("bug")
            exit(0)
    if flag:
        COUNT[3] += 1
    else:
        COUNT[4] += 1
    ssmm /= len(warped_images)
    del warped_images
    return result


def find_frames(frame):
    # width = int(frame.shape[1] * SCALE_PERCENT / 100)
    # height = int(frame.shape[0] * SCALE_PERCENT / 100)
    # dim = (width, height)
    # # resize image
    # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)

    time_start_o = time.time()
    warped_images = process(frame)
    time_elapsed_o = (time.time() - time_start_o)
    # print(f"detect object took {time_elapsed_o}")

    time_start = time.time()
    frame = post_processing(frame, warped_images)
    time_elapsed = (time.time() - time_start)

    # print(f"post_processing took {time_elapsed}")

    return frame


def video():
    # cap = cv2.VideoCapture("Data\\video_1_marker_cp.mp4")
    cap = cv2.VideoCapture(0)
    frame_width, frame_heightq = int(cap.get(3)), int(cap.get(4))
    # frame_width, frame_heighqt = 480, 640
    # out = cv2.VideoWriter('data.avi', cv2.VideoWriter_fourcc('M', 'J',
    # 'P', 'G'), 10, (frame_width,frame_height))
    frame_count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break
        # width = int(frame.shape[1] * SCALE_PERCENT)
        # height = int(frame.shape[0] * SCALE_PERCENT)
        # dim = (width, height)
        # # resize image
        # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)

        x = frame.copy()
        frame_count += 1
        time_start = time.clock()

        frames = find_frames(frame)

        time_elapsed = (time.clock() - time_start)
        print(f"frame {frame_count} {frame.shape} took : {time_elapsed}")

        result = np.hstack((x, frames))
        cv2.imshow("frame", result)
        # out.write(x)
        # ~!~!FD
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame_count += 1
    print(frame_count)
    print(f"% ID True / frame_counts: {COUNT[3] / frame_count}")
    print(f"% ID True / object True: {COUNT[3] / (frame_count - COUNT[0])}")
    print(f"% ID False / frame_counts: {COUNT[2] / frame_count}")
    print(f"% ID False / object True: {COUNT[2] / (frame_count - COUNT[0])}")
    print(
        f"% Object True / frame_counts: "
        f"{(frame_count - COUNT[0]) / frame_count}")
    # cv2.waitKey(0)
    # When everything done, release the capture
    # out.release()
    cap.release()


def imgg():
    # cap = cv2.VideoCapture("Data\\112961628984733575.mp4")
    # cap.set(1, 121)
    # ret, frame = cap.read()
    # cv2.imwrite("buggs.jpg", frame)

    image = cv2.imread("Untitled.png")
    # width = 480
    # height = 640
    # dim = (width, height)
    # # resize image
    # image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

    time_start = time.time()
    image = find_frames(image)
    time_elapsed = (time.time() - time_start)

    print(f"process {image.shape} took {time_elapsed}")

    cv2.imwrite("result.jpg", image)
    cv2.imshow("result.jpg", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    # imgg()
    video()
    print(COUNT)
    # cv2.destroyAllWindows()
