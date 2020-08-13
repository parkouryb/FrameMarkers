import math

import cv2
import numpy as np
from shapely.geometry import Polygon


def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


def bit_xor(a_list, b_list):
    return [] if len(a_list) != len(b_list) \
        else [a_list[index] ^ b_list[index] for index in range(len(a_list))]


def euclid_distance(point_1, point_2):
    return math.sqrt(
        (point_2[1] - point_1[1]) ** 2 + (point_2[0] - point_1[0]) ** 2)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1],
                            flags=cv2.INTER_LINEAR)
    return result


def binary_to_decimal(binary):
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
    X = (B1 - B2) / (A2 - A1) if A2 - A1 != 0 else 1
    Y = A1 * X + B1
    if X == float('+inf') or Y == float('+inf') or X == float(
            '-inf') or Y == float('-inf'):
        return None
    return int(X), int(Y)


def get_theta(point_A, point_B, point_C, point_D):
    A1, B1 = get_linear_equations(point_A, point_B)
    A2, B2 = get_linear_equations(point_C, point_D)
    if A1 is None or B1 is None or A2 is None or B2 is None:
        return None
    if A1 == 0 or A2 == 0:
        return None
    tu = (B1 * B2) / (A1 * A2) + B1 * B2
    mau = (math.sqrt((B1 / A1) ** 2 + B1 * B1) * math.sqrt(
        (B2 / A2) ** 2 + B2 * B2))
    if mau == 0 or abs(tu / mau) > 1.0:
        return None
    theta = math.acos(tu / mau)
    # print(A1, B1)
    # print(A2, B2)
    return theta * 180 / 3.14

