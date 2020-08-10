import math
import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

dirPath = os.path.dirname(os.path.realpath(__file__))
dir_marker_1 = "C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\markers\\1.png"
dir_image = "C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\marks"
dir_image1 = "C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\image\\1.jpg"
dir_image2 = "C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\image\\2.jpg"
dir_image3 = "C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\image\\3.jpg"
dir_image4 = "C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\image\\4.jpg"
mark_replace_1 = cv2.imread("C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\image\\guy.jpg", 1)
mark_replace_2 = cv2.imread("C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\image\\stewie.jpg", 1)
dt = {"10-246-292-355": mark_replace_1,
      "38-218-264-335": mark_replace_2}


def thresholdImage(gray):
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 29, 7)
    return threshold


def calculateMarker(image):
    size = image.shape[0]
    xx = 10
    crop_image = image[xx:size - xx, xx:size - xx]
    image = cv2.resize(crop_image, (size, size), interpolation=cv2.INTER_AREA)
    image = cv2.GaussianBlur(image, (9, 9), 0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean, std = cv2.meanStdDev(gray)
    ret, threshold = cv2.threshold(gray, int(mean - std / 2 + std / 5), 255, cv2.THRESH_BINARY)

    ls = []

    # top process
    iTop = 2 * 24 + 12
    top = ""
    while iTop < size - 60:
        if threshold[0:int((size * 2) / 21), iTop].sum() / 255 < 11.0:
            top += "0"
        else:
            top += "1"
        # print("top", threshold[0:int((size * 2) / 21), iTop].sum() / 255)
        cv2.line(threshold, (iTop, 0), (iTop, int((size * 2) / 21)), (128, 128, 128), 2)
        iTop += math.ceil((size - 60) / 10)

    # bot process
    iBot = int((size * 2) / 21) + 36
    sizeCount = size - int((size * 2) / 21)
    bot = ""
    while iBot < size - 24:
        if threshold[sizeCount:size, iBot].sum() / 255 < 11.0:
            bot += "0"
        else:
            bot += "1"
        # print("bot", threshold[sizeCount:size, iBot].sum() / 255)
        cv2.line(threshold, (iBot, sizeCount), (iBot, size), (128, 128, 128), 2)
        iBot += math.ceil((size - 60) / 10)
    # print(iBot, sizeCount)
    bot = bot[::-1]

    # left process
    iLeft = int((size * 2) / 21) + 36
    left = ""
    while iLeft < size - 24:
        if threshold[iLeft, 0:int((size * 2) / 21)].sum() / 255 < 11.0:
            left += "0"
        else:
            left += "1"
        # print("left", threshold[iLeft, 0:int((size * 2) / 21)].sum() / 255)
        cv2.line(threshold, (0, iLeft), (int((size * 2) / 21), iLeft), (128, 128, 128), 2)
        iLeft += math.ceil((size - 60) / 10)
    left = left[::-1]

    # right process
    iRight = 2 * 24 + 12
    right = ""
    while iRight < size - 60:
        if threshold[iRight, sizeCount:size].sum() / 255 < 11.0:
            right += "0"
        else:
            right += "1"
        # print("right", threshold[iRight, sizeCount:size].sum() / 255)
        cv2.line(threshold, (sizeCount, iRight), (size, iRight), (128, 128, 128), 2)
        iRight += math.ceil((size - 60) / 10)

    ls.append((binaryToDecimal(int(left)), "L"))
    ls.append((binaryToDecimal(int(top)), "T"))
    ls.append((binaryToDecimal(int(bot)), "B"))
    ls.append((binaryToDecimal(int(right)), "R"))
    ls.sort()

    haber = ""
    key = ""
    for i in range(4):
        key += ls[i][1]
        if i != 3:
            haber += str(ls[i][0]) + "-"
        else:
            haber += str(ls[i][0])

    cv2.imshow("threshold", threshold)
    return haber, key


def processImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold = thresholdImage(gray)

    edged = cv2.Canny(threshold, 50, 200)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,
                                       ksize=(3, 3))

    edged = cv2.morphologyEx(src=edged,
                             op=cv2.MORPH_DILATE,
                             kernel=kernel,
                             iterations=1)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    listContours = []
    for i in range(len(contours)):
        epsilon = 0.1 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)
        hull = cv2.convexHull(approx)
        if 3 < len(hull) < 5 and cv2.contourArea(approx) >= 4000.0:
            listContours.append(hull)
            # cv2.drawContours(image, [approx], -1, (0, 0, 255), 3, 8)

    sizeOut = 504  # 24 * 21
    list_result = []
    warped = image.copy()
    for contour in listContours:
        lC = []
        if len(contour) != 4:
            continue
        for cnt in contour:
            lC.append(list(cnt[0]))
        lC.sort(key=lambda x: x[1])
        if lC[1][0] > lC[2][0]:
            pass
        else:
            lC[1], lC[0] = lC[0], lC[1]
            lC[2], lC[3] = lC[3], lC[2]
        pts = np.float32(lC)
        dst_pts = np.float32([[0, 0], [sizeOut, 0], [0, sizeOut], [sizeOut, sizeOut]])
        M = cv2.getPerspectiveTransform(pts, dst_pts)
        warped = cv2.warpPerspective(image, M, dsize=(sizeOut, sizeOut), flags=cv2.INTER_LINEAR)
        list_result.append((warped, pts, sizeOut))

        lC.clear()
        print(type(pts), type(contour), contour, pts, sep="\n")
        cv2.drawContours(image, [contour], -1, (0, 0, 0), cv2.FILLED)

    return image, list_result


def project():
    image = cv2.imread("C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\Data\\marks\\6.jpg")
    x = image.copy()
    image, list_result = processImage(image)
    for i in range(len(list_result)):
        cv2.imshow("asd", list_result[i][0])
        cv2.waitKey(0)
        mask_value, mask_key = calculateMarker(list_result[i][0])
        # print(mask_value, mask_key, list_result[i][1], list_result[i][2], sep="\n")

        # if mask_value in dt:
        #     rpl_mark = dt[mask_value]
        #     sizeOut = rpl_mark.shape[0]
        #     pts1 = np.float32([[0, 0], [sizeOut, 0], [0, sizeOut], [sizeOut, sizeOut]])
        #     pts2 = list_result[i][1]
        #     M = cv2.getPerspectiveTransform(pts1, pts2)
        #     dst = cv2.warpPerspective(rpl_mark, M, (image.shape[1], image.shape[0]))
        #     dest_and = cv2.bitwise_xor(image, dst, mask=None)
        #     # cv2.imshow(str(i), dst)
        #     # cv2.imwrite("C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\" + str(9) + ".jpg", dest_and)

    # cv2.imwrite("C:\\Users\\Admin\\PycharmProjects\\FrameMarkers\\" + str(6) + ".jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def decimalToBinary(n):
    if n == 0:
        return ''
    else:
        return decimalToBinary(n / 2) + str(n % 2)


def binaryToDecimal(binary):
    decimal, i, n = 0, 0, 0
    while binary != 0:
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary // 10
        i += 1
    return decimal


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


if __name__ == '__main__':
    # project()
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        x = frame.copy()
        frame, list_result = processImage(image=frame)
        dest_and = frame.copy()
        for i in range(len(list_result)):
            mask_value, mask_key = calculateMarker(list_result[i][0])
            print(i, ":", mask_value, mask_key, sep="\n")

            if mask_value in dt:
                rpl_mark = dt[mask_value]
                sizeOut = rpl_mark.shape[0]
                if mask_key == "LTRB":
                    pass
                elif mask_key == "BLTR":
                    rpl_mark = rotate_image(rpl_mark, 90)
                elif mask_key == "TRBL":
                    rpl_mark = rotate_image(rpl_mark, 270)
                else:
                    rpl_mark = rotate_image(rpl_mark, 180)

                pts1 = np.float32([[0, 0], [sizeOut, 0], [0, sizeOut], [sizeOut, sizeOut]])
                pts2 = list_result[i][1]
                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(rpl_mark, M, (frame.shape[1], frame.shape[0]))
                dest_and = cv2.bitwise_xor(dest_and, dst, mask=None)

        cv2.imshow('frame', x)
        cv2.imshow('frame_1', dest_and)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
