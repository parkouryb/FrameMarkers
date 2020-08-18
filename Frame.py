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
MARKER_RESIZE = 210  # 10 * 21

MARK_IMAGE_1 = cv2.imread(
    marker_folder + '\\' + file_name_marker[0], 1)
MARK_IMAGE_2 = cv2.imread(
    marker_folder + '\\' + file_name_marker[1], 1)
MARK_IMAGE_3 = cv2.imread(
    marker_folder + '\\' + file_name_marker[2], 1)
MARK_IMAGE_4 = cv2.imread(
    marker_folder + '\\' + file_name_marker[3], 1)

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

COUNT = [0 for i in range(3)]
IC = [0 for i in range(4)]


def kill_small_contour(contour, area):
    x, y, w, h = cv2.boundingRect(contour)
    return True if w * h > area else False


def contour_process(contours, *ars):
    frame = ars[0]
    list_approxs = []
    for i in range(len(contours)):
        contour = contours[i]
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        hull = cv2.convexHull(approx)

        isPoly = False
        points = [list(_contour[0]) for _contour in hull]
        edges = [(sf.euclid_distance(points[i],
                                     points[(i + 1) % len(hull)]), i)
                 for i in range(len(hull))]

        if len(hull) == 5:
            # cv2.drawContours(frame, [hull], -1, (0, 0, 255), 3, 8)
            edges.sort(key=lambda x: x[0])
            edges_min = min(edges[-1][1], edges[-2][1])
            edges_max = max(edges[-1][1], edges[-2][1])
            if abs(edges[-1][1] - edges[-2][1]) != 1 \
                    or (edges_max == 4 and edges_min == 0):
                continue
            key = str(edges[-1][1]) + str(edges[-2][1])

            point_A = points[IDX[key][0]]
            point_B = points[IDX[key][1]]
            point_C = points[IDX[key][0] - 1]
            point_D = points[(IDX[key][1] + 1) % len(hull)]
            key_point = sf.get_key_point(point_A, point_B, point_C, point_D)
            if key_point is None:
                continue
            if edges_max == 4 and edges_min == 0:
                points.pop(0), points.pop(-1)
                hull = np.delete(hull, [0, -1], axis=0)
                hull = np.insert(hull, len(hull), list(key_point), axis=0)
                points.append(list(key_point))
            else:
                key0 = IDX[key][0]
                points.pop(key0)
                hull = np.delete(hull, [key0], axis=0)
                rmi = 0 if key0 == len(points) else key0
                points.pop(rmi)
                hull = np.delete(hull, [rmi], axis=0)
                # insert
                points.insert(key0, list(key_point))
                hull = np.insert(hull, min(key0, len(hull)), list(key_point),
                                 axis=0)
            isPoly = True
        #     # cv2.circle(frame, tuple(key_point), 5, (0, 0, 255), cv2.FILLED)
        #     # cv2.drawContours(frame, [hull], -1, (0, 0, 0), 2, 8)

        if len(hull) == 4 or isPoly:
            if isPoly and len(hull) != 4:
                continue
            # cv2.drawContours(frame, [hull], -1, (0, 0, 255), 1, 8)
            edges.clear()
            edges = [sf.euclid_distance(points[i], points[(i + 1) % len(hull)])
                     for i in range(len(hull))]
            average = np.average(edges)
            threshold_min, threshold_max = average * 0.7, average * 1.3
            # print(
            #     f"{i} got {edges} {threshold_min}, {threshold_max},
            #     "f"{avg_length}")

            bool_edges = np.logical_not(
                np.bitwise_and(threshold_min < np.array(edges),
                               np.array(edges) < threshold_max))
            fl = not (np.any(bool_edges))

            if fl:
                # cv2.drawContours(frame, [hull], -1, (255, 0, 0), 3, 8)
                list_approxs.append((hull, points))
            else:
                obj_width = max(edges[0], edges[1])
                obj_height = min(edges[0], edges[1])
                if obj_width > obj_height * 1.7:
                    continue
                theta_1 = sf.get_theta(points[0], points[1], points[0],
                                       points[-1])
                theta_2 = sf.get_theta(points[2], points[1], points[2],
                                       points[-1])
                if theta_1 is None or theta_2 is None:
                    continue
                theta = theta_2 + theta_1
                if (abs(edges[0] - edges[2]) <= 10.0 or abs(
                        edges[1] - edges[
                            -1]) <= 10.0) and average >= 50.0 * SCALE_PERCENT \
                        and (theta < 175.0 or theta > 185.0):
                    list_approxs.append((hull, points))
        del hull
    return list_approxs


def pre_process(frame):
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
                                      blockSize=29, C=15)

    threshold = cv2.medianBlur(src=threshold, ksize=3)

    contours, hierarchy = cv2.findContours(image=threshold,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)

    # print(f"small range: {(frame.shape[0] * frame.shape[1]) / 25}")
    contours = [contour for contour in contours if
                kill_small_contour(contour, (
                        frame.shape[0] * frame.shape[1]) / 30) is True]

    time_end_1 = time.time() - time_start_1
    # print(f"step 1: {time_end_1}")
    # BUG-1
    # cv2.imshow("barrier", threshold)
    # cv2.drawContours(frame, contours, -1, (255, 0, 128), 3, 8)
    time_start_2 = time.time()
    list_approxs = contour_process(contours, frame)
    approx_list = {}
    remap = np.array([], dtype=int)
    # for _ in list_approxs:
    #     cv2.drawContours(frame, [_[0]], -1, (0, 0, 255), 3, 8)

    for i in range(len(list_approxs)):
        isIouZero = True
        for j in range(i + 1, len(list_approxs)):
            iou = sf.calculate_iou(list_approxs[j][1], list_approxs[i][1])
            area_1, area_2 = cv2.contourArea(
                list_approxs[i][0]), cv2.contourArea(list_approxs[j][0])
            if iou > 0.00 and abs(area_1 - area_2) <= max(area_2, area_1) / 3:
                index = i if area_1 > area_2 else j
                if index in remap:
                    break
                else:
                    idx = i if i != index else j
                    remap = np.append(remap, idx)
                approx_list[index] = list_approxs[index]
        if isIouZero:
            if i in remap:
                continue
            approx_list[i] = list_approxs[i]
    del remap

    list_approxs.clear()

    for key, value in approx_list.items():
        # cv2.drawContours(frame, [value[0]], -1, (0, 0, 255), 1, 8)
        list_approxs.append(value)

    del approx_list

    time_end_2 = (time.time() - time_start_2)
    # print(f"step 2 : {time_end_2}")
    warped_images = []
    time_start_3 = time.time()
    for contour, points in list_approxs:
        # cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3, 8)
        points = np.float32(points)

        warped_points = np.float32(
            [[MARKER_RESIZE, 0],
             [MARKER_RESIZE, MARKER_RESIZE],
             [0, MARKER_RESIZE],
             [0, 0]]
        )
        points = np.float32(points)
        M = cv2.getPerspectiveTransform(points, warped_points)
        warped = cv2.warpPerspective(src=frame, M=M,
                                     dsize=(MARKER_RESIZE, MARKER_RESIZE),
                                     flags=cv2.INTER_LINEAR)
        # cv2.drawContours(frame, [contour], -1, (0, 0, 0), 2, 8)
        warped_images.append((warped, points, contour))

    time_end_3 = time.time() - time_start_3
    # print(f"step 3 : {time_end_3}")
    return warped_images


def edge_process(image):
    ret, border = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    code = ""
    area = 10 * 20
    if border is None or border.shape[1] < 100 * SCALE_PERCENT:
        return "0"

    index = 0
    for k in range(round(border.shape[1] / 21), border.shape[1],
                   round((border.shape[1] * 2) / 20)):
        # cv2.rectangle(image, (k, 0), (k + 10, border.shape[1]), (128, 128,
        #                                                          128),
        #               cv2.FILLED)
        # print(index, np.sum(border[0:k, k:min(k + 10, border.shape[
        #     1])]) / 255)
        x = "0" if np.count_nonzero(border[0:k, k:min(k + 10, border.shape[
            1])]) > area * 0.25 else "1"
        code += x
        index += 1
        if index == 9:
            break

    return code if len(code) == 9 else "0"


def marker_process(image, index):
    size = image.shape[0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean, std = cv2.meanStdDev(gray)
    ret, threshold = cv2.threshold(gray, int(mean), 255, cv2.THRESH_BINARY)

    cv2.rectangle(threshold, (0, 0), (threshold.shape[0], threshold.shape[1]),
                  (0, 0, 0), 10, 8)

    # cv2.imshow("haizz" + str(index), gray)
    # cv2.imshow("haizzp2" + str(index), threshold)

    contours, hierarchy = cv2.findContours(image=threshold,
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_NONE)
    m, box = 0, (0, 0, size, size)
    for contour in contours:
        _box = cv2.boundingRect(contour)
        area = _box[2] * _box[3]
        if area > m:
            m = area
            box = _box

    border = 0
    box = list(box)
    if min(box[2], box[3]) <= 150.0 * SCALE_PERCENT:
        mx = max(box[2], box[3])
        box[2], box[3] = mx, mx

    img = threshold[box[1] + border:box[1] + box[3] - border,
          box[0] + border:box[0] + box[2] - border]
    # cv2.imshow("haizz2" + str(index), img)

    size = img.shape[0]

    edges_image = [
        cv2.rotate(img[0:size, 0:10], cv2.ROTATE_90_CLOCKWISE),
        img[0:10, 0:size],
        cv2.rotate(img[0:size, size - 10:size],
                   cv2.ROTATE_90_COUNTERCLOCKWISE),
        cv2.rotate(img[size - 10:size, 0:size], cv2.ROTATE_180)
    ]
    # [cv2.imshow(str(i) + str(index), edges_image[i]) for i in
    #  range(4)]

    mats = [sf.binary_to_decimal(int(edge_process(edges_image[i]))) for i in
            range(4)]
    key = ""
    position = ""
    # print(mats)
    # [cv2.imshow(str(index) + str(i), edges_image[i]) for i in range(4)]
    if sum(mats) == 511 * 4 or sum(mats) == 0:
        return key, position
    mm = 1
    for idx in range(4):
        xor_list = sf.bit_xor(mats, ID_0)
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
    return key, position


def post_processing(frame, warped_images):
    if not warped_images:
        return frame

    COUNT[0] += len(warped_images)
    result = frame.copy()
    flag = True
    ssmm = 0.0
    # print(warped_images[4])
    for i, (warped, warped_points, contour) in enumerate(warped_images):
        key, position = marker_process(warped, i)
        if key == "":
            COUNT[1] += 1
            flag = False
            continue
        if key in ID:
            IC[key] += 1
            COUNT[2] += 1
            replace_image = ID[key]
            replace_image_size = replace_image.shape[0]
            if position == "LTRB":
                pass
            elif position == "BLTR":
                replace_image = sf.rotate_image(replace_image, 270)
            elif position == "RBLT":
                replace_image = sf.rotate_image(replace_image, 180)
            elif position == "TRBL":
                replace_image = sf.rotate_image(replace_image, 90)
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
            print("bug")
            exit(0)
    return result


def find_markers(frame):
    time_start_o = time.time()
    warped_images = pre_process(frame)
    time_elapsed_o = (time.time() - time_start_o)
    # print(f"detect object took {time_elapsed_o}")

    time_start = time.time()
    frame = post_processing(frame, warped_images)
    time_elapsed = (time.time() - time_start)

    # print(f"post_processing took {time_elapsed}")
    return frame


def imgProcess():
    image = cv2.imread("frames/206.jpg")
    # image = cv2.imread("Data/data.jpg")

    width = int(image.shape[1] * SCALE_PERCENT)
    height = int(image.shape[0] * SCALE_PERCENT)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    time_start = time.time()
    image = find_markers(image)
    time_elapsed = (time.time() - time_start)

    print(f"process {image.shape} took {time_elapsed}")

    cv2.namedWindow("result", cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("result", image)
    cv2.waitKey(0)


def videoProcess(name):
    # name = 'camera_1.avi'
    cap = cv2.VideoCapture(name)
    # cap = cv2.VideoCapture(0)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    # frame_width, frame_height = 480, 270
    n = name.replace("Data\\Video\\", "")
    n = n.replace(".mp4", ".avi")
    out = cv2.VideoWriter("Data\\Video Output\\" + n,
                          cv2.VideoWriter_fourcc('M',
                                                 'J', 'P',
                                                 'G'), 10,
                          (int(frame_width * SCALE_PERCENT),
                           int(frame_height * SCALE_PERCENT)))
    frame_count = 0
    sum = 0
    max_time = 0
    min_time = 1.0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break

        x = frame.copy()
        frame_count += 1
        time_start = time.clock()

        frame = cv2.resize(frame, (int(frame.shape[1] * SCALE_PERCENT),
                                   int(frame.shape[0] * SCALE_PERCENT)),
                           interpolation=cv2.INTER_CUBIC)
        frames = find_markers(frame)

        time_elapsed = (time.clock() - time_start)
        # print(f"frame {frame_count} {frame.shape} took : {time_elapsed}")
        sum += time_elapsed
        max_time = max(max_time, time_elapsed)
        min_time = min(min_time, time_elapsed)
        cv2.imshow("frame", frames)
        out.write(frames)
        # cv2.imwrite("frames\\" + str(frame_count) + '.jpg', x)
        # cv2.imwrite("framesout\\" + str(frame_count) + '.jpg', frames)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame_count += 1
    sum /= frame_count
    # print(f"time avg: {sum} with frame: {frame_count} and size "
    #       f"{frame_width * SCALE_PERCENT, frame_height * SCALE_PERCENT}")
    # print(f"max time: {max_time}\nmin time: {min_time}")
    if name == 0:
        print(COUNT)
        print(IC)
    # cv2.waitKey(0)
    # When everything done, release the capture
    out.release()
    cap.release()


def manyVideo():
    for i in range(1, 13):
        COUNT = [0 for i in range(3)]
        IC = [0 for i in range(4)]
        print(f"video {i}.mp4")
        videoProcess("Data\\Video\\" + str(i) + ".mp4")
        # print(COUNT)
        # print(IC)


if __name__ == '__main__':
    # imgProcess()
    name = 'Data\\Video\\6.mp4'
    # videoProcess(0)
    for i in range(1, 13):
        COUNT = [0 for i in range(3)]
        IC = [0 for i in range(4)]
        print(f"video {i}.mp4")
        videoProcess("Data\\Video Input\\" + str(i) + ".mp4")
        # print(COUNT)
        # print(IC)
    cv2.destroyAllWindows()
