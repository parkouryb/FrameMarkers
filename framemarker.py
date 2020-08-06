import math
import operator
import time
from operator import itemgetter

import cv2
import numpy as np

MARKER_RESIZE = 504  # 24 * 21 | 20 * 21
MARK_IMAGE_1 = cv2.imread("C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\image\\guy.jpg", 1)
MARK_IMAGE_2 = cv2.imread("C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\image\\stewie.jpg", 1)
MARK_IMAGE_3 = cv2.imread("C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\image\\guy.jpg", 1)
MARK_IMAGE_4 = cv2.imread("C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\image\\haha.jpg", 1)
MARK_IMAGE_5 = cv2.imread("C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\image\\tinyguy.jpg", 1)
ID_0 = [
    39, 219, 265, 334
]

POSITION = [
    ['LTRB', 'TRBL', 'RBLT', 'BLTR'],
    ['BLTR', 'LTRB', 'TRBL', 'RBLT'],
    ['RBLT', 'BLTR', 'LTRB', 'TRBL'],
    ['TRBL', 'RBLT', 'BLTR', 'LTRB']
]

COUNT = [0 for i in range(3)]

def EuclidDistance(point_1, point_2):
    return math.sqrt((point_2[1] - point_1[1]) ** 2 + (point_2[0] - point_1[0]) ** 2)

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def binaryToDecimal(binary):
    decimal, i, n = 0, 0, 0
    while binary != 0:
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary // 10
        i += 1
    return decimal

def process(frame):
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

    edges = cv2.Canny(image=threshold,
                      threshold1=50,
                      threshold2=200)

    contours, hierarchy = cv2.findContours(image=edges,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)

    contours = [contours[i] for i in range(len(contours)) if cv2.contourArea(contours[i]) > 3000.0]

    list_approxs = []
    for i in range(len(contours)):
        epsilon = 0.05 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)
        hull = cv2.convexHull(approx)

        if len(hull) == 4 and (cv2.contourArea(hull) - cv2.contourArea(approx) <= 300.0):
            points = []

            for cnt in hull:
                points.append(list(cnt[0]))

            avg_length = (EuclidDistance(points[0], points[1]) + EuclidDistance(points[1], points[2]) +
                          EuclidDistance(points[2], points[3]) + EuclidDistance(points[0], points[3])) / 4
            min_threshold, max_threshold = avg_length * 0.8, avg_length * 1.2
            square_edges = [EuclidDistance(points[0], points[1]), EuclidDistance(points[1], points[2]),
                            EuclidDistance(points[2], points[3]), EuclidDistance(points[0], points[3])]

            bool_edges = np.logical_not(np.bitwise_and(min_threshold < np.array(square_edges),
                                                       np.array(square_edges) < max_threshold))
            fl = not (np.any(bool_edges))

            if fl is False:
                continue

            list_approxs.append((hull, points))
            # cv2.drawContours(frame, [hull], -1, (255, 0, 0), 3, 8)

    warped_images = []

    for _ in list_approxs:
        contour, points = _
        pts = np.float32(points)
        # print(pts)
        dst_pts = np.float32([[MARKER_RESIZE, 0], [MARKER_RESIZE, MARKER_RESIZE], [0, MARKER_RESIZE], [0, 0]])
        M = cv2.getPerspectiveTransform(pts, dst_pts)
        warped = cv2.warpPerspective(src=frame, M=M,
                                     dsize=(MARKER_RESIZE, MARKER_RESIZE),
                                     flags=cv2.INTER_LINEAR)
        # cv2.drawContours(frame, [contour], -1, (0, 0, 0), cv2.FILLED)

        warped_images.append((warped, pts, contour))

    return warped_images

def edge_process(image):
    ret, border = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    code = ""
    area = 25 * 20
    if border is None:
        return "*"
    # print(border.shape, border.shape[1] / 20)
    for index, k in enumerate(range(round(border.shape[1] / 20), border.shape[1], round(border.shape[1] / 20) * 2)):
        if index == 9:
            break
        # cv2.rectangle(image, (k, 0), (k + 24, border.shape[1]), (128, 128, 128), cv2.FILLED)
        if np.sum(border[0:k, k:min(k + 24, border.shape[1])]) / 255 > area * 0.3:
            code += "0"
        else:
            code += "1"

    return code if len(code) == 9 else "*"

def marker_process(image, index):
    size = image.shape[0]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean, std = cv2.meanStdDev(gray)
    ret, threshold = cv2.threshold(gray, int(mean), 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(image=threshold,
                      threshold1=50,
                      threshold2=200)

    contours, hierarchy = cv2.findContours(image=edges,
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)

    m, box = 0, (0, 0, size, size)
    for i in range(len(contours)):
        box_ = cv2.boundingRect(contours[i])
        area = box_[2] * box_[3]
        if area > m:
            m = area
            box = box_

    x = 3
    img = threshold[box[1] + x:box[1] + box[3] - x, box[0] + x:box[0] + box[2] - x]
    size = img.shape[0]

    edges_image = [
        cv2.rotate(img[0:size, 0:20], cv2.ROTATE_90_CLOCKWISE),
        img[0:20, 0:size],
        cv2.rotate(img[0:size, size - 20:size],
                  cv2.ROTATE_90_COUNTERCLOCKWISE),
        cv2.rotate(img[size - 20:size, 0:size], cv2.ROTATE_180)
    ]
    mats = [binaryToDecimal(int(edge_process(edges_image[i]))) for i in range(4)]
    # [cv2.imshow(str(i), edges_image[i]) for i in range(4)]

    key = ""
    position = ""

    if sum(mats) == 511 * 4 or sum(mats) == 0:
        return key, position, threshold

    print(mats)

    for index, value in enumerate(mats):
        print(index, value)

    #
    # lss = [val.split('-') for val, mat in dt.items()]
    # itms = {}
    # for i in range(len(mats)):
    #     value = mats[i]
    #     for j in range(len(lss)):
    #         ls = lss[j]
    #         if value in ls:
    #             # key = '-'.join(ls)
    #             # position = bb[i][ls.index(value)]
    #             if '-'.join(ls) + str(":") + bb[i][ls.index(value)] in itms:
    #                 itms['-'.join(ls) + str(":") + bb[i][ls.index(value)]] += 1
    #             else:
    #                 itms['-'.join(ls) + str(":") + bb[i][ls.index(value)]] = 1
    # if len(itms) != 0:
    #     key, position = max(itms.items(), key=operator.itemgetter(1))[0].split(":")
    #     if itms[key + str(":") + position] < 2:
    #         key, position = "", ""
    return key, position, threshold

def post_processing(frame, warped_images):
    if len(warped_images) == 0:
        COUNT[0] += 1
        return frame

    result = frame.copy()
    list_index = []

    for i in range(len(warped_images)):
        warped, pts_dst, contour = warped_images[i]
        mask_value, mask_key, threshold = marker_process(warped, i)
        # if mask_value == "":
        #     continue
        # print(i, ":", mask_value, mask_key)
        # if mask_value in dt:
        #     COUNT[2] += 1
        #     rpl_mark = dt[mask_value]
        #     size_out = rpl_mark.shape[0]
        #     if mask_key == "LTRB":
        #         pass
        #     elif mask_key == "BLTR":
        #         rpl_mark = rotate_image(rpl_mark, 270)
        #     elif mask_key == "RBLT":
        #         rpl_mark = rotate_image(rpl_mark, 180)
        #     elif mask_key == "TRBL":
        #         rpl_mark = rotate_image(rpl_mark, 90)
        #     else:
        #         print("bug", mask_value, mask_key)
        #         exit(0)
        #     pts = np.float32([[size_out, 0], [size_out, size_out], [0, size_out], [0, 0]])
        #     M = cv2.getPerspectiveTransform(pts, pts_dst)
        #     dst = cv2.warpPerspective(rpl_mark, M, (frame.shape[1], frame.shape[0]))
        #
        #     # draw process
        #     cv2.drawContours(result, [contour], -1, (0, 0, 0), cv2.FILLED)
        #     result = cv2.bitwise_xor(result, dst, mask=None)

    warped_images.clear()
    list_index.clear()
    return result

def find_frames(frame):
    warped_images = process(frame)

    frame = post_processing(frame, warped_images)

    return frame


if __name__ == '__main__':
    image = cv2.imread("a.png")

    image = find_frames(image)

    cv2.imwrite("as.jpg", image)
    cv2.waitKey(0)

    # cap = cv2.VideoCapture("Data\\video2.mp4")
    # frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    # out = cv2.VideoWriter('outvideo2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width,
    # frame_height))
    # frame_count = 0
    # while True:
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #     if frame is None:
    #         break
    #     x = frame.copy()
    #     frame_count += 1
    #     time_start = time.clock()
    #     frames = find_frames(frame)
    #
    #     time_elapsed = (time.clock() - time_start)
    #     print("frame :", frame_count, "-", time_elapsed)
    #
    #     result = np.hstack((x, frames))
    #     cv2.imshow("frame", result)
    #     out.write(frames)
    #     # if frame_count == 17:
    #     #     break
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.waitKey(0)
    # print(COUNT)
    # # When everything done, release the capture
    # out.release()
    # cap.release()
    cv2.destroyAllWindows()
