"""This file contains the main extraction routines"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging
import os


from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps

from actions import (detect_structures, find_arrows, get_conditions_smiles, complete_structures,
                     remove_redundant_characters, find_optimal_dilation_ksize, scan_form_reaction_step)
from conditions import get_conditions
from models.segments import FigureRoleEnum
from models.output import ReactionScheme
from utils.io_ import imread
from utils.processing import erase_elements
import chemschematicresolver as csr
import settings

log = logging.getLogger('extract')

def extract_image(filename, debug=False):
    """
    Extracts reaction schemes from a single file specified by ``filename``. ``debug`` enables more detailed logging and
    plotting.

    :param str filename: name of the image file
    :param bool debug: bool enabling debug mode
    :return Scheme: Reaction scheme object
    """

    # create console handler and set level
    level = 'DEBUG' if debug else 'INFO'
    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    log.addHandler(ch)

    MAIN_DIR = os.getcwd()
    path = os.path.join(MAIN_DIR, filename)

    fig = imread(path)
    settings.main_figure.append(fig)
    detect_structures(fig)  # This should be a Figure method; running during ''init''? - can't do, circular ref
    log.debug('Bond boundary length: %d' % fig.boundary_length)
    find_optimal_dilation_ksize(fig)

    # global_skel_pixel_ratio = skeletonize_area_ratio(fig, fig.get_bounding_box())
    arrows = find_arrows(fig, int(fig.boundary_length))
    log.info('Detected %d arrows' % len(arrows))

    structure_panels = complete_structures(fig)
    log.info('Found %d structure panels' % len(structure_panels))

    conditions = [get_conditions(fig, arrow) for arrow in arrows]
    for step_conditions in conditions:
        log.info('Conditions dictionary found: %s' % step_conditions.conditions_dct)

    fig_no_cond = erase_elements(fig, [cc for cc in fig.connected_components
                                       if cc.role == FigureRoleEnum.ARROW or cc.role == FigureRoleEnum.CONDITIONSCHAR])
    fig_clean = remove_redundant_characters(fig_no_cond, fig_no_cond.connected_components)
    processed = Image.fromarray(fig_clean.img).convert('RGB')
    processed = ImageOps.invert(processed)

    segmented_filepath = MAIN_DIR + '/processed' + '.tif'
    processed.save(segmented_filepath)

    log.debug('Cleaned image sent to the structure-label resolution model')
    csr_out = csr.extract_image_rde(MAIN_DIR + '/processed.tif', kernel_size=fig.kernel_size, debug=True,
                                    allow_wildcards=False)
    log.info('CSR found the following species: %s' % csr_out[0])
    conditions = get_conditions_smiles(csr_out, conditions)
    os.remove(segmented_filepath)

    steps = [scan_form_reaction_step(step_conditions, structure_panels) for step_conditions in conditions]
    [step.match_function_and_smiles(csr_out) for step in steps]
    log.info('SMILES-structure matching complete.')
    if debug:
        colors = ['r', 'g', 'y', 'm', 'b', 'c', 'k']

        f = plt.figure()
        ax = f.add_axes([0.1, 0.1, 0.9, .9])
        ax.imshow(fig.img)
        for panel in fig.connected_components:
            if panel.role:
                color = colors[panel.role.value]
            else:
                color = 'w'
            rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
                                  facecolor='none', edgecolor=color)
            ax.add_patch(rect_bbox)
        for step in steps:
            for group in step:
                for species in group:
                    panel = species.panel
                    rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
                                          facecolor='none', edgecolor='m')
                    ax.add_patch(rect_bbox)
        plt.show()

    scheme = ReactionScheme(steps)
    log.info('Scheme completed without errors')

    settings.main_figure = []
    return scheme