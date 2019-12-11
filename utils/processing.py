import copy
import math
import numpy as np

from models.utils import Line
from models.segments import Rect


def hide_arrows(fig, arrow):
    """

    :param fig: Figure object
    :param arrow: Arrow object
    :return: copy of the Figure object with arrows removed
    """
    fig= copy.deepcopy(fig)
    flattened = fig.img.flatten()
    np.put(flattened, [x * fig.img.shape[1] + y for x, y in zip(*arrow.pixels)], 0)
    img_no_arrow = flattened.reshape(fig.img.shape[0], fig.img.shape[1])
    fig.img = img_no_arrow
    return fig


def approximate_line(p1, p2):
    """
    This is a prototype and will not work for vertical lines.
    This algorithm operates in Cartesian space
    """
    #TODO: But both output and input are in the image space -
    #so reimplement this to match
    x1, y1 = p1.col, p1.row
    x2, y2 = p2.col, p2.row
    deltax = x2 - x1
    deltay = y2 - y1
    if x2 > x1:
        domain = range(x1, x2 + 1)
    else:
        domain = range(x2, x1 + 1)

    try:
        deltaerr = abs(deltay / deltax)
    except ZeroDivisionError:
        line = [(y1, x_value) for x_value in domain]
    else:
        error = 0
        y = y1
        line = []
        for x in domain:
            # Add row, then column
            line.append((y, x))
            error += deltaerr
            if error >= 0.5:
                deltay_sign = int(math.copysign(1, deltay))
                y += deltay_sign
                error -= 1

    return Line(line)


def create_megabox(boxes):
    """
    :param boxes: list of conditions bounding boxes
    :return: megabox
    """
    top = min(rect.top for rect in boxes)
    bottom = max(rect.bottom for rect in boxes)
    left = min(rect.left for rect in boxes)
    right = max(rect.right for rect in boxes)

    megabox = Rect(top=top, bottom=bottom, left=left, right=right)
    return megabox


