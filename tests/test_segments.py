import pytest

import numpy as np

from models.segments import Figure, Rect, Panel, TextLine
from models.utils import Point


def test_rect_overlaps_vertically():
    r1 = Rect(0, 20, 0, 40)
    r2 = Rect(0, 20, 0, 40)
    r3 = Rect(10, 20, 30, 60)
    o12 = r1.overlaps_vertically(r2)
    o13 = r1.overlaps_vertically(r3)
    o23 = r2.overlaps_vertically(r2)
    assert o12 == True
    assert o13 == True
    assert o23 == True


def test_textline_connected_components_setter():
    text_line = TextLine(0, 200, 0, 50)
    ccs = [Panel(0, 220, 0, 50)]
    text_line.connected_components = ccs
    assert text_line.left == 0
    assert text_line.right == 220
    assert text_line.top == 0
    assert text_line.bottom == 50


def test_textline_connected_components_append():
    text_line = TextLine(0, 200, 0, 50)
    cc = Panel(0, 220, 0, 50)
    text_line.append(cc)
    assert text_line.left == 0
    assert text_line.right == 220
    assert text_line.top == 0
    assert text_line.bottom == 50


def test_from_points_tuples():
    points = [(0, 200), (0, 400), (-100, 200), (-100, 400)]
    r = Rect.from_points(points)
    assert r == Rect(-100, 0, 200, 400)


def test_from_points_nparrays():
    points = np.array([(0, 200), (0, 400), (-100, 200), (-100, 400)])
    r = Rect.from_points(points)
    assert r == Rect(-100, 0, 200, 400)


def test_from_points_Points():
    points = [Point(200, 0), Point(400, 0), Point(200, -100), Point(400, -100)]
    r = Rect.from_points(points)
    assert r == Rect(-100, 0, 200, 400)


if __name__ == '__main__':
    test_textline_connected_components_append()
    test_textline_connected_components_setter()
    test_rect_overlaps_vertically()
    test_from_points_tuples()
    test_from_points_nparrays()
    test_from_points_Points()
