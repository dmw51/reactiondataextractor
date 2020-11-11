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

from actions import (detect_structures, find_arrows, complete_structures,
                      find_optimal_dilation_ksize, scan_form_reaction_step)
from conditions import get_conditions, clear_conditions_region
from models.segments import FigureRoleEnum, ReactionRoleEnum
from models.output import ReactionScheme
from utils.io_ import imread
from utils.processing import erase_elements
from resolve import LabelAssigner, RGroupResolver
from recognise import DiagramRecogniser
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
    log.info(f'Extracting file: {path}')

    fig = imread(path)
    settings.main_figure.append(fig)
    detect_structures(fig)  # This should be a Figure method; running during ''init''? - can't do, circular ref
    log.debug('Bond boundary length: %d' % fig.boundary_length)
    find_optimal_dilation_ksize(fig)

    # global_skel_pixel_ratio = skeletonize_area_ratio(fig, fig.get_bounding_box())
    arrows = find_arrows(fig, int(fig.boundary_length))
    log.info('Detected %d arrows' % len(arrows))

    structure_panels = complete_structures(fig)

    #### Hide the following inside a function
    conditions, conditions_structures = zip(*[get_conditions(fig, arrow) for arrow in arrows])
    conditions_structures = [panel for step_panels in conditions_structures for panel in step_panels]
    [setattr(structure, 'role', ReactionRoleEnum.CONDITIONS) for structure in structure_panels
     if structure in conditions_structures]
    react_prod_structures = [panel for panel in structure_panels if panel not in conditions_structures]
    ####
    for step_conditions in conditions:
        log.info('Conditions dictionary found: %s' % step_conditions.conditions_dct)

    fig_no_cond = clear_conditions_region(fig)


    log.info('Found %d structure panels' % len(structure_panels))


    # fig_clean = remove_redundant_characters(fig_no_cond, fig_no_cond.connected_components)
    # processed = Image.fromarray(fig_no_cond.img).convert('RGB')
    # processed = ImageOps.invert(processed)

    # segmented_filepath = MAIN_DIR + '/processed' + '.tif'
    # processed.save(segmented_filepath)
    assigner = LabelAssigner(fig_no_cond, react_prod_structures, conditions_structures)
    diags = assigner.create_diagrams()

    resolver = RGroupResolver(diags)
    resolver.analyse_labels()

    recogniser = DiagramRecogniser(diags)
    recogniser.recognise_diagrams()
    ### Hide the following inside a function
    for step_conditions in conditions:
        if step_conditions._structure_panels:
            cond_diags = [diag for diag in diags if diag.panel in step_conditions._structure_panels]
            step_conditions.diags = cond_diags
            step_conditions.conditions_dct['other species'].extend([diag.smiles for diag in cond_diags if diag.smiles])
    ####
    steps = [scan_form_reaction_step(step_conditions, diags) for step_conditions in conditions]
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
        ax.set_title('Roles')
        plt.show()

        f = plt.figure()
        ax = f.add_axes([0.1, 0.1, 0.9, .9])
        ax.imshow(fig.img)
        for diag in diags:
            panel = diag.panel
            rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
                                  facecolor='none', edgecolor='m')
            ax.add_patch(rect_bbox)
            if diag.label:
                panel = diag.label.panel
                rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
                                      facecolor='none', edgecolor='g')
                ax.add_patch(rect_bbox)
        ax.set_title('Diagrams and labels')
        plt.show()

    scheme = ReactionScheme(steps)
    log.info('Scheme completed without errors')

    settings.main_figure = []
    return scheme

print()