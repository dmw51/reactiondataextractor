# -*- coding: utf-8 -*-
"""
Extract
=======

Main extraction routines.

author: Damian Wilary
email: dmw51@cam.ac.uk

"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from matplotlib import pyplot as plt
import os

from actions import estimate_single_bond
from extractors import (ArrowExtractor, ConditionsExtractor, clear_conditions_region, DiagramExtractor, LabelExtractor,
                        RGroupResolver)
from models.output import ReactionScheme
from recognise import DiagramRecogniser
import settings
from utils.io_ import imread
from utils.processing import mark_tiny_ccs

log = logging.getLogger('extract')
file_handler = logging.FileHandler('extract.log')
log.addHandler(file_handler)


def extract_image(filename, debug=False):
    """
    Extracts reaction schemes from a single file specified by ``filename``. ``debug`` enables more detailed logging and
    plotting.

    :param str filename: name of the image file
    :param bool debug: bool enabling debug mode
    :return Scheme: Reaction scheme object
    """
    level = 'DEBUG' if debug else 'INFO'
    ch = logging.StreamHandler()
    log.setLevel(level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)

    MAIN_DIR = os.getcwd()
    path = os.path.join(MAIN_DIR, filename)
    log.info(f'Extraction started...')

    fig = imread(path)
    settings.main_figure.append(fig)
    fig.single_bond_length = estimate_single_bond(fig)
    mark_tiny_ccs(fig)

    arrow_extractor = ArrowExtractor()
    arrows = arrow_extractor.extract()
    log.info(f'Detected {len(arrows)} arrows')
    diag_extractor = DiagramExtractor()
    structure_panels = diag_extractor.extract()
    log.info(f'Found {len(structure_panels)} panels of chemical diagrams')
    conditions_extractor = ConditionsExtractor(arrows)
    conditions, conditions_structures = conditions_extractor.extract()
    for step_conditions in conditions:
        log.info('Conditions dictionary found: %s' % step_conditions.conditions_dct)

    react_prod_structures = [panel for panel in structure_panels if panel not in conditions_structures]
    fig_no_cond = clear_conditions_region(fig)

    label_extractor = LabelExtractor(fig_no_cond, react_prod_structures, conditions_structures)
    diags = label_extractor.extract()
    log.info('Label extraction process finished.')

    resolver = RGroupResolver(diags)
    resolver.analyse_labels()

    recogniser = DiagramRecogniser(diags)
    recogniser.recognise_diagrams()
    log.info('Diagrams have been optically recognised.')
    conditions_extractor.add_diags_to_dicts(diags)

    if debug:
        f = plt.figure()
        ax = f.add_axes([0, 0, 1, 1])
        ax.imshow(fig.img, cmap=plt.cm.binary)
        arrow_extractor.plot_extracted(ax)
        conditions_extractor.plot_extracted(ax)
        diag_extractor.plot_extracted(ax)
        label_extractor.plot_extracted(ax)
        ax.axis('off')
        ax.set_title('Segmented image')
        plt.show()

    scheme = ReactionScheme(conditions, diags)
    log.info('Scheme completed without errors.')

    print(scheme)
    print()
    print(scheme.long_str())
    settings.main_figure = []
    return scheme
