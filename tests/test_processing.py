import pytest

import numpy as np
import matplotlib.pyplot as plt

from models.segments import Figure, Rect, Panel
from utils.processing import (transform_panel_coordinates_to_expanded_rect, binary_tag,
                              get_bounding_box, crop_rect, flatten_list, detect_headers,
                              transform_panel_coordinates_to_shrunken_region)

arr = np.zeros((100,100))
arr[20:35,45:60] = 1
fig = Figure(arr)
labelled = binary_tag(fig)
ccs = list(get_bounding_box(labelled))
main_rect = Rect(0, arr.shape[1], 0, arr.shape[0])
#print(ccs[0])


def test_transform_panel_coordinates_to_parent_single_crop():
    crop_rectangle = Rect(10, 90, 15, 95)
    cropped_img = crop_rect(arr, crop_rectangle)['img']
    # Find the same cc and transform back to the original cc, check if coordinates are the same
    labelled_crop = binary_tag(Figure(cropped_img))
    crop_ccs = get_bounding_box(labelled_crop)
    trans_crop = transform_panel_coordinates_to_expanded_rect(crop_rectangle, main_rect, crop_ccs)
    assert ccs[0] == trans_crop[0]


def test_transform_panel_coordinates_to_parent_crop_of_a_crop():
    crop_rectangle = Rect(10, 90, 10, 95)
    #print(f'crop:{crop_rectangle}')
    cropped_img = crop_rect(arr, crop_rectangle)['img']
    # Find the same cc and transform back to the original cc, check if coordinates are the same
    labelled_crop = binary_tag(Figure(cropped_img))
    crop_ccs = get_bounding_box(labelled_crop)
    #print(f'crop_ccs: {crop_ccs}')

    # Crop the crop now
    subcrop_rectangle = Rect(10, 70, 10, 60)
    cropped_img_2 = crop_rect(cropped_img, subcrop_rectangle)['img']
    labelled_subcrop = binary_tag(Figure(cropped_img_2))
    subcrop_ccs = get_bounding_box(labelled_subcrop)

    # Now do two transformations to go back to the main reference system
    trans_crop_1_step = transform_panel_coordinates_to_expanded_rect(subcrop_rectangle, crop_rectangle, subcrop_ccs)
    assert ccs[0] == transform_panel_coordinates_to_expanded_rect(crop_rectangle,main_rect,trans_crop_1_step)[0]


def test_transform_panel_coordinates_to_parent_crop_of_a_crop_absolute():
    crop_rectangle = Rect(10, 90, 10, 95)
    # print(f'crop:{crop_rectangle}')
    cropped_img = crop_rect(arr, crop_rectangle)['img']
    # Find the same cc and transform back to the original cc, check if coordinates are the same
    labelled_crop = binary_tag(Figure(cropped_img))
    crop_ccs = get_bounding_box(labelled_crop)
    # print(f'crop_ccs: {crop_ccs}')

    # Crop the crop now
    subcrop_rectangle = Rect(10, 70, 10, 60)
    cropped_img_2 = crop_rect(cropped_img, subcrop_rectangle)['img']
    labelled_subcrop = binary_tag(Figure(cropped_img_2))
    subcrop_ccs = get_bounding_box(labelled_subcrop)

    # Instead of doing two transformations, express the `subcrop_rectangle` in global coordinates to
    # perform a single tranformation back to the main frame
    subcrop_in_main_frame = Rect(crop_rectangle.left+subcrop_rectangle.left, crop_rectangle.right+subcrop_rectangle.right,
                                 crop_rectangle.top+subcrop_rectangle.top, crop_rectangle.bottom+subcrop_rectangle.bottom)

    assert ccs[0] == transform_panel_coordinates_to_expanded_rect(subcrop_in_main_frame, main_rect,
                                                                  subcrop_ccs,absolute=True)[0]


def test_flatten_list_ok():
    arr1 = [1, 2, 3, 4]
    arr2 = [6, 3 ,4, 5]
    arr3 = [7, 8, 9, 10]
    arr1.append(arr2)
    arr3.append(arr1)
    a = flatten_list(arr3)
    assert a == [7, 8, 9, 10, 1, 2, 3, 4, 6, 3, 4, 5]


def test_detect_headers_single_letter():
    img = np.zeros((300, 2000))
    img[5:30, 6:40] = 1  # header blob
    img[35:120, 5:300] = 1  # larger blob - will not be detected
    img[150:175, 7:40] = 1  # another header blob
    img[190:300, 40: 400] = 1  # larger blob again
    img[5:30, 600:800] = 1  # short blob in line with header (but not part of the header)
    header_ccs = [[Panel(left=6, right=40, top=5, bottom=30)], [Panel(left=7, right=40, top=150, bottom=175)]]
    detected_headers = detect_headers(Figure(img))

    assert {frozenset(elem) for elem in header_ccs} == {frozenset(elem) for elem in detected_headers}


def test_detect_headers_blocks():
    img = np.zeros((300, 2000))
    header_mask = np.linspace(0, 500, 21, dtype='int')

    img[5:30, 0:500] = 1  # header blob
    img[5:30, header_mask] = 0  # introduce spaces
    img[5:30, header_mask[10]:header_mask[11]] = 0  # introduce a space character
    img[35:120, 5:300] = 1  # larger blob - will not be detected
    img[150:175, 7:40] = 1  # another header blob
    img[190:300, 40: 400] = 1  # larger blob again
    img[5:30, 600:800] = 1  # short blob in line with header (but not part of the header)

    header_ccs = [[Panel(left=header_mask[idx] + 1, right=header_mask[idx + 1], top=5, bottom=30) for idx
                   in range(len(header_mask) - 1) if idx != 10], [Panel(left=7, right=40, top=150, bottom=175)]]

    detected_headers = detect_headers(Figure(img))

    assert {frozenset(elem) for elem in header_ccs} == {frozenset(elem) for elem in detected_headers}


# def test_detect_rectangles():


if __name__ == '__main__':
    test_transform_panel_coordinates_to_parent_single_crop()
    test_transform_panel_coordinates_to_parent_crop_of_a_crop()
    test_transform_panel_coordinates_to_parent_crop_of_a_crop_absolute()
    test_flatten_list_ok()
    test_detect_headers_single_letter()
    test_detect_headers_blocks()
