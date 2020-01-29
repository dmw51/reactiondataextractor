import pytest

import numpy as np

from models.segments import Figure, Rect, Panel
from utils.processing import transform_panel_coordinates_to_parent, binary_tag, get_bounding_box, crop_rect

arr = np.zeros((100,100))
arr[20:35,45:60] = 1
fig = Figure(arr)
labelled = binary_tag(fig)
ccs = list(get_bounding_box(labelled))
main_rect = Rect(0, arr.shape[1], 0, arr.shape[0])
#print(ccs[0])
def test_transform_panel_coordinates_to_parent_single_crop():
    crop_rectangle = Rect(10, 90, 15, 95)
    #print(f'crop:{crop_rectangle}')
    cropped_img = crop_rect(arr, crop_rectangle)['img']
    # Find the same cc and transform back to the original cc, check if coordinates are the same
    labelled_crop = binary_tag(Figure(cropped_img))
    crop_ccs = get_bounding_box(labelled_crop)
    #print(f'crop_ccs: {crop_ccs}')
    trans_crop = transform_panel_coordinates_to_parent(crop_rectangle, main_rect, crop_ccs)
    #print(f'after transforming: {trans_crop}')
    assert ccs[0] == transform_panel_coordinates_to_parent(crop_rectangle, main_rect, crop_ccs)[0]


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
    trans_crop_1_step = transform_panel_coordinates_to_parent(subcrop_rectangle, crop_rectangle, subcrop_ccs)
    assert ccs[0] == transform_panel_coordinates_to_parent(crop_rectangle,main_rect,trans_crop_1_step)[0]


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

    assert ccs[0] == transform_panel_coordinates_to_parent(subcrop_in_main_frame, main_rect, subcrop_ccs,absolute=True)[0]



if __name__ == 'main':
    test_transform_panel_coordinates_to_parent_single_crop()
    test_transform_panel_coordinates_to_parent_crop_of_a_crop()
    test_transform_panel_coordinates_to_parent_crop_of_a_crop_absolute()
