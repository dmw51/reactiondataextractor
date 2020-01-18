"""
This script will calculate all image-specific variables
"""

def get_area(fig):
    dim1, dim2 = fig.img.shape
    return dim1*dim2
