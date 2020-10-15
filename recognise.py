"""This module contains optical chemical structure recognition tools and routines"""
import os
import logging
import itertools
from skimage.util import pad

import utils.io_ as io_
from utils.processing import clean_output
import osra_rgroup
log = logging.getLogger()
parent_dir = os.path.dirname(os.path.abspath(__file__))
superatom_file = os.path.join(parent_dir, 'dict', 'superatom.txt')
spelling_file = os.path.join(parent_dir, 'dict', 'spelling.txt')


class DiagramRecogniser:
    """Used to optical chemical structure recognition of diagrams"""

    def __init__(self, diagrams, allow_wildcards=False):
        self.diagrams = diagrams
        self.allow_wildcards = allow_wildcards
        self._tag_multiple_r_groups()

    def recognise_diagrams(self):
        for diag in self.diagrams:
            if diag.r_groups:
                diag.smiles = self._get_rgroup_smiles(diag)
            else:
                diag.smiles = self._read_diagram_pyosra(diag)

    def _tag_multiple_r_groups(self):
        for diag in self.diagrams:
            if diag.label and diag.label.r_group and len(diag.label.r_group) > 1:
                diag.r_groups = True
            else:
                diag.r_groups = False

    def _get_rgroup_smiles(self, diag, extension='jpg', debug=False, superatom_path=superatom_file,
                          spelling_path=spelling_file):
        """ Extract SMILES from a chemical diagram (powered by pyosra)

        :param diag: Input Diagram
        :param extension: String indicating format of input file
        :param debug: Bool to indicate debugging

        :return labels_and_smiles: List of Tuple(List of label candidates, SMILES) objects
        """

        # Add some padding to image to help resolve characters on the edge
        padded_img = pad(diag.crop.raw_img, ((5, 5), (5, 5), (0, 0)), mode='constant', constant_values=1)

        # Save a temp image
        img_name = 'r_group_temp.' + extension
        io_.imsave(img_name, padded_img)

        osra_input = []
        # label_cands = []

        # Format the extracted rgroup
        for tokens in diag.label.r_group:
            token_dict = {}
            for token in tokens:
                token_dict[token[0].text] = token[1].text

            # Assigning var-var cases to true value if found (eg. R1=R2=H)
            for a, b in itertools.combinations(token_dict.keys(), 2):
                if token_dict[a] == b:
                    token_dict[a] = token_dict[b]

            osra_input.append(token_dict)
            # label_cands.append(tokens[0][2])

        # Run osra on temp image
        smiles = osra_rgroup.read_rgroup(osra_input, input_file=img_name, verbose=False, debug=debug,
                                         superatom_file=superatom_path, spelling_file=spelling_path)
        if not smiles:
            log.warning('No SMILES strings were extracted for diagram %s' % diag.tag)

        if not debug:
            io_.imdel(img_name)

        # smiles = [actions.clean_output(smile) for smile in smiles]
        return smiles
        # labels_and_smiles = []
        # for i, smile in enumerate(smiles):
        #     labels_and_smiles.append((label_cands[i], smile))
        #
        # return labels_and_smiles

    def _read_diagram_pyosra(self, diag, extension='jpg', debug=False, pad_val=1, superatom_path=superatom_file,
                            spelling_path=spelling_file):
        """ Converts a diagram to SMILES using pyosra

        :param diag: Diagram to be extracted
        :param extension: String file extension
        :param debug: Bool inicating debug mode

        :return smile: String of extracted chemical SMILE

        """
        diag_crop = diag.crop
        if hasattr(diag_crop, 'clean_raw_img'):  # Choose either the cleaned (de-noised) version or a raw crop
            img = diag_crop.clean_raw_img
        else:
            img = diag_crop.raw_img
        # label = transform_panel_coordinates_to_shrunken_region(diag.crop.cropped_rect, [diag.label])[0]
        # img = erase_elements(img, [label], 1)
        # Add some padding to image to help resolve characters on the edge
        if len(img.shape) == 3:
            padded_img = pad(img, ((20, 20), (20, 20), (0, 0)), mode='constant', constant_values=pad_val)
        else:
            padded_img = pad(img, ((20, 20), (20, 20)), mode='constant', constant_values=pad_val)

        # Save a temp image
        temp_img_fname = 'osra_temp.' + extension
        io_.imsave(temp_img_fname, padded_img)

        # Run osra on temp image
        try:
            smile = osra_rgroup.read_diagram(temp_img_fname, debug=debug, superatom_file=superatom_path,
                                             spelling_file=spelling_path)
        except Exception as e:
            print(str(e))

        if not smile:
            log.warning('No SMILES string was extracted for diagram %s' % diag.tag)

        if not debug:
            io_.imdel(temp_img_fname)

        smile = clean_output(smile)
        return smile


    def is_false_positive(self, diag, allow_wildcards=False):
        """ Identifies failures from incomplete / invalid smiles

        :rtype bool
        :returns : True if result is a false positive
        """

        # label_candidates, smiles = diag.label, diag.smiles
        smiles = diag.smiles
        # Remove results without a label
        # if len(label_candidates) == 0:
        #     return True

        # Remove results containing the wildcard character in the SMILE
        if '*' in smiles and not allow_wildcards:
            return True

        # Remove results where no SMILE was returned
        if smiles == '':
            return True

        return False
