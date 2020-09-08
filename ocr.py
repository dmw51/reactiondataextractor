# -*- coding: utf-8 -*-
"""
Optical Character Recognition
=============================

Extract text from images using Tesseract.

Module adapted by :-
author: Ed Beard
email: ejb207@cam.ac.uk

from FigureDataExtractor (<CITATION>) :-
author: Matthew Swain
email: m.swain@me.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import enum
import logging
import warnings

import numpy as np
import locale
locale.setlocale(locale.LC_ALL, 'C')
import tesserocr
from skimage.morphology import skeletonize
from skimage.transform import rescale
from skimage.filters import gaussian

from chemdataextractor.doc.text import Sentence
from chemdataextractor.parse.cem import BaseParser
from chemschematicresolver import decorators, io_, model
from utils.processing import convert_greyscale, crop, crop_rect, pad, normalize_image
from chemschematicresolver.parse import ChemSchematicResolverTokeniser, LabelParser

import matplotlib.pyplot as plt
log = logging.getLogger('extract.ocr')

# Whitelist for labels
ALPHABET_UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALPHABET_LOWER = ALPHABET_UPPER.lower()
DIGITS = '0123456789'
SUBSCRIPT = '₁₂₃₄₅₆₇₈₉ₓ₋'
SUPERSCRIPT = '⁰°'
ASSIGNMENT = ':=-'
CONCENTRATION = '%()'
BRACKETS ='[]{}'
SEPARATORS = ',. '
OTHER = r'\'`/@'
LABEL_WHITELIST = (ASSIGNMENT + DIGITS + ALPHABET_UPPER + ALPHABET_LOWER + CONCENTRATION +
                   OTHER + SUBSCRIPT + SUPERSCRIPT + SEPARATORS)
CONDITIONS_WHITELIST = DIGITS + ALPHABET_UPPER + ALPHABET_LOWER + CONCENTRATION + SEPARATORS + BRACKETS + SUPERSCRIPT\
                       + OTHER
CHAR_WHITELIST = DIGITS + '+' + ALPHABET_UPPER

OCR_CONFIDENCE = 0.7

def choose_best_ocr_configuration(img, psms=[], invert=True, x_offset=0, y_offset=0, whitelist=None,pad_val=1):
    """
    Runs the ocr process for the specifies page segmentation modes. if `invert` is True, checks if ocr is better when
    the image is inverted.
    :param np.ndarray img: analysed image
    :param [PSM,...] psms: list of all PSMs to try
    :param bool invert: flag to process inverted image
    :param int x_offset: the 'left' coordinate of bounding box of the 'img' in the main figure
    :param int y_offset: the 'top' coordinate of bounding box of the 'img' in the main figure
    :param str whitelist: whitelist for tesseract OCR process
    :return[TextBlock,...]: raw ocr output with the highest confidence score
    """

    raw_results = []
    for psm in psms:
        text = get_text(img, x_offset=x_offset, y_offset=y_offset, psm=psm, whitelist=whitelist, pad_val=pad_val)
        confidence = np.mean([ t.confidence for t in text]) if text else 0
        raw_results.append([text, confidence])

    if invert:
        pad_val = 1-pad_val
        img = img.astype(float)
        max_val = np.max(img)
        inverted_img = np.abs(max_val - img)

        for psm in psms:
            text = get_text(inverted_img, x_offset=x_offset, y_offset=y_offset, psm=psm, whitelist=whitelist, pad_val=pad_val)
            confidence = np.mean([t.confidence for t in text]) if text else 0
            raw_results.append([text, confidence])

    raw_results.sort(key=lambda result: result[1], reverse=True)

    # print(f'all ocr results: {raw_results}')
    text, avg_conf = raw_results[0]
    # print(f' ocr text: {text}')

    return raw_results[0]

def read_character(fig, diag, whitelist=CHAR_WHITELIST, pad_val=0):
    """ Recognises a single character"""

    img = convert_greyscale(fig.img)
    cropped_img = crop(img, left=diag.left, right=diag.right, top=diag.top, bottom=diag.bottom)
    # plt.imshow(cropped_img['img'])
    # plt.show()
    # cropped_img = skeletonize(cropped_img['img'])
    cropped_img = pad(cropped_img['img'], (10, 10))
    #cropped_img = pad(cropped_img, (10, 10))
    # plt.imshow(cropped_img)
    # plt.show()
    # # cropped_img = skeletonize(cropped_img)
    # modes = ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
    # f, ax = plt.subplots(5)
    # for i, mode in enumerate(modes):
    #     cropped_img = gaussian(cropped_img,sigma=0.5, mode='mirror')
    #     text = get_text(cropped_img, x_offset=diag.left, y_offset=diag.top, psm=PSM.SINGLE_CHAR, whitelist=whitelist)
    #     if text:
    #         char = get_words(text)[0]
    #     print(f' recognised: {char}, conf: {char.confidence}, mode: {mode}')
    #     ax[i].imshow(cropped_img)
    #     ax[i].set_title(mode)
    # plt.show()


    cropped_img = gaussian(cropped_img, sigma=0.5, mode='nearest')
    # print(f'whitelist :{whitelist}')
    # plt.imshow(cropped_img)
    # plt.title('blurred')
    # plt.show()
    text, _ = choose_best_ocr_configuration(cropped_img, x_offset=diag.left, y_offset=diag.top,
                                            psms=[PSM.SINGLE_CHAR, PSM.SPARSE_TEXT],
                                            whitelist=whitelist, pad_val=pad_val)
    if text:
        char = get_words(text)[0]

        if char.confidence > OCR_CONFIDENCE:
            return char


def read_diag_text(fig, diag, whitelist=LABEL_WHITELIST):
    """ Reads a diagram using OCR and returns the textual OCR objects"""
    img = convert_greyscale(fig.img)
    cropped_img = crop(img, left=diag.left, right=diag.right, top=diag.top, bottom=diag.bottom)
    text = get_text(cropped_img, x_offset=diag.left, y_offset=diag.top, psm=PSM.SINGLE_BLOCK, whitelist=whitelist)
    tokens = get_words(text)
    return text


def read_label(fig, label, whitelist=LABEL_WHITELIST):
    """ Reads a label paragraph objects using ocr

    :param numpy.ndarray img: Input unprocessedimage
    :param Label label: Label object containing appropriate bounding box

    :rtype List[List[str]]
    """

    size = 5
    img = convert_greyscale(fig.img)
    cropped_img = crop(img, label.left, label.right, label.top, label.bottom)
    padded_img = pad(cropped_img, size, mode='constant', constant_values=(1, 1))
    text = get_text(padded_img, x_offset=label.left, y_offset=label.top, psm=PSM.SINGLE_BLOCK, whitelist=whitelist)
    if not text:
        label.text = []
        return label, 0
    raw_sentences = get_sentences(text)

    if len(raw_sentences) is not 0:
        # Tag each paragraph
        tagged_sentences = [Sentence(sentence, word_tokenizer=ChemSchematicResolverTokeniser(),
                                           parsers=[BaseParser()]) for sentence in raw_sentences]
    else:
        tagged_sentences = []
    label.text = tagged_sentences

    # Calculating average confidence for the block
    confidences = [t.confidence for t in text]
    avg_conf = np.mean(confidences)
    log.info('Confidence in OCR: %s' % avg_conf)

    return label, avg_conf


def read_conditions(fig, textline, conf_threshold = OCR_CONFIDENCE, whitelist=CONDITIONS_WHITELIST, pad_val=0):
    """
    Reads conditions' text in `fig.img` detected inside `text_line` and returns the recognised OCR objects.
    :param Figure fig: figure containing unprocessed image
    :param TextLine textline:  TextLine delimiting position of a single line of conditions' text
    :param str whitelist: all characters that will be looked for in the text
    :return: ?
    """

    # img = convert_greyscale(fig.img)   # Should  not be needed (pre-processed)
    panel = textline.panel
    crop = panel.create_crop(fig)
    cropped_img = pad(crop.img, (10, 10))
    # cropped_img = gaussian(cropped_img, sigma=0.5, mode='nearest')
    # plt.imshow(cropped_img)
    # plt.title('cropped condition text line')
    # plt.show()
    # print(f'max of crop: {np.max(cropped_img)}')
    # print(f'min of crop: {np.min(cropped_img)}')
    # cropped_img = normalize_image(cropped_img)

    text, avg_conf = choose_best_ocr_configuration(cropped_img, x_offset=panel.left, y_offset=panel.top,
                                            psms=[PSM.SINGLE_LINE, PSM.SINGLE_WORD], whitelist=whitelist, pad_val=pad_val)
    raw_sentence = get_sentences(text)

    if len(raw_sentence) == 1:
        # Tag each paragraph
        tagged_sentence = Sentence(raw_sentence[0].strip(), word_tokenizer=ChemSchematicResolverTokeniser(),
                                   parsers=[BaseParser()])
    else:
        tagged_sentence = [] # TODO: Only one sentence (line) should be detected. Is this always the case?

    log.debug('Tagged sentence: %s' % tagged_sentence)
    log.debug('Confidence in OCR: %s' % avg_conf)
    return tagged_sentence if avg_conf > conf_threshold else []


def read_isolated_conditions(isolated_block):
    """
    Helper function to read conditions from isolated connected components segmented as conditions' text characters
    :param isolated_block:
    :return:
    """
    fig = isolated_block
    conditions_region = isolated_block.get_bounding_box()
    return read_conditions(fig, conditions_region)

# These enums just wrap tesserocr functionality, so we can return proper enum members instead of ints.

class Orientation(enum.IntEnum):
    """Text element orientations enum."""
    #: Up orientation.
    PAGE_UP = tesserocr.Orientation.PAGE_UP
    #: Right orientation.
    PAGE_RIGHT = tesserocr.Orientation.PAGE_RIGHT
    #: Down orientation.
    PAGE_DOWN = tesserocr.Orientation.PAGE_DOWN
    #: Left orientation.
    PAGE_LEFT = tesserocr.Orientation.PAGE_LEFT


class WritingDirection(enum.IntEnum):
    """Text element writing directions enum."""
    #: Left to right.
    LEFT_TO_RIGHT = tesserocr.WritingDirection.LEFT_TO_RIGHT
    #: Right to left.
    RIGHT_TO_LEFT = tesserocr.WritingDirection.RIGHT_TO_LEFT
    #: Top to bottom.
    TOP_TO_BOTTOM = tesserocr.WritingDirection.TOP_TO_BOTTOM


class TextlineOrder(enum.IntEnum):
    """Text line order enum."""
    #: Left to right.
    LEFT_TO_RIGHT = tesserocr.TextlineOrder.LEFT_TO_RIGHT
    #: Right to left.
    RIGHT_TO_LEFT = tesserocr.TextlineOrder.RIGHT_TO_LEFT
    #: Top to bottom.
    TOP_TO_BOTTOM = tesserocr.TextlineOrder.TOP_TO_BOTTOM


class Justification(enum.IntEnum):
    """Justification enum."""
    #: Unknown justification.
    UNKNOWN = tesserocr.Justification.UNKNOWN
    #: Left justified
    LEFT = tesserocr.Justification.LEFT
    #: Center justified
    CENTER = tesserocr.Justification.CENTER
    #: Right justified
    RIGHT = tesserocr.Justification.RIGHT


class PSM(enum.IntEnum):
    """Page Segmentation Mode enum."""
    OSD_ONLY = tesserocr.PSM.OSD_ONLY
    AUTO_OSD = tesserocr.PSM.AUTO_OSD
    AUTO_ONLY = tesserocr.PSM.AUTO_ONLY
    AUTO = tesserocr.PSM.AUTO
    SINGLE_COLUMN = tesserocr.PSM.SINGLE_COLUMN
    SINGLE_BLOCK_VERT_TEXT = tesserocr.PSM.SINGLE_BLOCK_VERT_TEXT
    SINGLE_BLOCK = tesserocr.PSM.SINGLE_BLOCK
    SINGLE_LINE = tesserocr.PSM.SINGLE_LINE
    SINGLE_WORD = tesserocr.PSM.SINGLE_WORD
    CIRCLE_WORD = tesserocr.PSM.CIRCLE_WORD
    SINGLE_CHAR = tesserocr.PSM.SINGLE_CHAR
    SPARSE_TEXT = tesserocr.PSM.SPARSE_TEXT
    SPARSE_TEXT_OSD = tesserocr.PSM.SPARSE_TEXT_OSD
    RAW_LINE = tesserocr.PSM.RAW_LINE
    COUNT = tesserocr.PSM.COUNT


class RIL(enum.IntEnum):
    """Page Iterator Level enum."""
    BLOCK = tesserocr.RIL.BLOCK
    PARA = tesserocr.RIL.PARA
    SYMBOL = tesserocr.RIL.SYMBOL
    TEXTLINE = tesserocr.RIL.TEXTLINE
    WORD = tesserocr.RIL.WORD


def get_words(blocks):
    """Convert list of text blocks into a flat list of the contained words.

    :param list[TextBlock] blocks: List of text blocks.
    :return: Flat list of text words.
    :rtype: list[TextWord]
    """
    words = []
    for block in blocks:
        for para in block:
            for line in para:
                for word in line:
                    words.append(word)
    return words


def get_lines(blocks):
    """Convert list of text blocks into a nested list of lines, each of which contains a list of words.

    :param list[TextBlock] blocks: List of text blocks.
    :return: List of sentences
    :rtype: list[list[TextWord]]
    """
    lines = []
    for block in blocks:
        for para in block:
            for line in para:
                words = []
                for word in line:
                    words.append(word)
                lines.append(words)
    return lines


def get_sentences(blocks):
    """Convert list of text blocks into a nested list of lines, each of which contains a list of words.

    :param list[TextBlock] blocks: List of text blocks.
    :return: List of sentences
    :rtype: list[list[TextWord]]
    """
    sentences = []
    for block in blocks:
        for para in block:
            for line in para:
                # sentences.append(line.text.replace(',', ' ')) # NB - commas switched for spaces to improve tokenization
                sentences.append(line.text)
    return sentences


def get_text(img, x_offset=0, y_offset=0, psm=PSM.SINGLE_LINE, padding=20, whitelist=None, img_orientation=None, pad_val=1):
    """Get text elements in image.

    When passing a cropped image to this function, use ``x_offset`` and ``y_offset`` to ensure the coordinate positions
    of the extracted text elements are relative to the original uncropped image.

    :param numpy.ndarray img: Input image.
    :param int x_offset: Offset to add to the horizontal coordinates of the returned text elements.
    :param int y_offset: Offset to add to the vertical coordinates of the returned text elements.
    :param PSM psm: Page segmentation mode.
    :param int padding: Padding to add to text element bounding boxes.
    :param string whitelist: String containing allowed characters. e.g. Use '0123456789' for digits.
    :param Orientation img_orientation: Main orientation of text in image, if known.
    :return: List of text blocks.
    :rtype: list[TextBlock]
    """
    log.debug(
        'get_text: %s x_offset=%s, y_offset=%s, padding=%s, whitelist=%s',
        img.shape, x_offset, y_offset, padding, whitelist
    )
    # Rescale image - make it larger
    img = rescale(img, 2.0, order=1, mode='reflect', anti_aliasing=True, multichannel=False)

    # Add a buffer around the entire input image to ensure no text is too close to edges
    img_padding = 10
    if img.ndim == 3:
        npad = ((img_padding, img_padding), (img_padding, img_padding), (0, 0))
    elif img.ndim == 2:
        npad = ((img_padding, img_padding), (img_padding, img_padding))
    else:
        raise ValueError('Unexpected image dimensions')
    img = np.pad(img, pad_width=npad, mode='constant', constant_values=pad_val)
    shape = img.shape
    # plt.imshow(img)
    # plt.title('padded, resized rde')
    # plt.show()

    # Rotate img before sending to tesseract if an img_orientation has been given
    if img_orientation == Orientation.PAGE_LEFT:
        img = np.rot90(img, k=3, axes=(0, 1))
    elif img_orientation == Orientation.PAGE_RIGHT:
        img = np.rot90(img, k=1, axes=(0, 1))
    elif img_orientation is not None:
        raise NotImplementedError('Unsupported img_orientation')

    def _get_common_props(it, ril):
        """Get the properties that apply to all text elements."""
        # Important: Call GetUTF8Text() before Orientation(). Former raises RuntimeError if no text, latter Segfaults.
        text = it.GetUTF8Text(ril)
        orientation, writing_direction, textline_order, deskew_angle = it.Orientation()
        bb = it.BoundingBox(ril, padding=padding)

        # Translate bounding box and orientation if img was previously rotated
        if img_orientation == Orientation.PAGE_LEFT:
            orientation = {
                Orientation.PAGE_UP: Orientation.PAGE_LEFT,
                Orientation.PAGE_LEFT: Orientation.PAGE_DOWN,
                Orientation.PAGE_DOWN: Orientation.PAGE_RIGHT,
                Orientation.PAGE_RIGHT: Orientation.PAGE_UP
            }[orientation]
            left, right, top, bottom = bb[1], bb[3], shape[0] - bb[2], shape[0] - bb[0]
        elif img_orientation == Orientation.PAGE_RIGHT:
            orientation = {
                Orientation.PAGE_UP: Orientation.PAGE_RIGHT,
                Orientation.PAGE_LEFT: Orientation.PAGE_UP,
                Orientation.PAGE_DOWN: Orientation.PAGE_LEFT,
                Orientation.PAGE_RIGHT: Orientation.PAGE_DOWN
            }[orientation]
            left, right, top, bottom = shape[1] - bb[3], shape[1] - bb[1], bb[0], bb[2]
        else:
            left, right, top, bottom = bb[0], bb[2], bb[1], bb[3]

        common_props = {
            'text': text,
            'left': left + x_offset - img_padding,
            'right': right + x_offset - img_padding,
            'top': top + y_offset - img_padding,
            'bottom': bottom + y_offset - img_padding,
            'confidence': it.Confidence(ril),
            'orientation': Orientation(orientation),
            'writing_direction': WritingDirection(writing_direction),
            'textline_order': TextlineOrder(textline_order),
            'deskew_angle': deskew_angle
        }
        return common_props

    blocks = []

    with tesserocr.PyTessBaseAPI(psm=psm) as api:
        # api.SetVariable("segment_penalty_dict_nonword", "0")
        # api.SetVariable("segment_penalty_dict_frequent_word", "0")
        # api.SetVariable("segment_penalty_dict_case_ok", "0")
        # api.SetVariable("segment_penalty_dict_case_bad", "0")
        # Convert image to PIL to load into tesseract (suppress precision loss warning)
        with warnings.catch_warnings(record=True) as ws:
            pil_img = io_.img_as_pil(img)
        api.SetImage(pil_img)
        if whitelist is not None:
            api.SetVariable('tessedit_char_whitelist', whitelist)
        # TODO: api.SetSourceResolution if we want correct pointsize on output?
        api.Recognize()
        it = api.GetIterator()
        block = None
        para = None
        line = None
        word = None
        it.Begin()

        while True:
            try:
                if it.IsAtBeginningOf(RIL.BLOCK):
                    common_props = _get_common_props(it, RIL.BLOCK)
                    block = TextBlock(**common_props)
                    blocks.append(block)

                if it.IsAtBeginningOf(RIL.PARA):
                    common_props = _get_common_props(it, RIL.PARA)
                    justification, is_list_item, is_crown, first_line_indent = it.ParagraphInfo()
                    para = TextParagraph(
                        is_ltr=it.ParagraphIsLtr(),
                        justification=Justification(justification),
                        is_list_item=is_list_item,
                        is_crown=is_crown,
                        first_line_indent=first_line_indent,
                        **common_props
                    )
                    if block is not None:
                        block.paragraphs.append(para)

                if it.IsAtBeginningOf(RIL.TEXTLINE):
                    common_props = _get_common_props(it, RIL.TEXTLINE)
                    line = TextLine(**common_props)
                    if para is not None:
                        para.lines.append(line)

                if it.IsAtBeginningOf(RIL.WORD):
                    common_props = _get_common_props(it, RIL.WORD)
                    wfa = it.WordFontAttributes()
                    if wfa:
                        common_props.update(wfa)
                    word = TextWord(
                        language=it.WordRecognitionLanguage(),
                        from_dictionary=it.WordIsFromDictionary(),
                        numeric=it.WordIsNumeric(),
                        **common_props
                    )
                    if line is not None:
                        line.words.append(word)

                # Beware: Character level coordinates do not seem to be accurate in Tesseact 4!!
                common_props = _get_common_props(it, RIL.SYMBOL)
                symbol = TextSymbol(
                    is_dropcap=it.SymbolIsDropcap(),
                    is_subscript=it.SymbolIsSubscript(),
                    is_superscript=it.SymbolIsSuperscript(),
                    **common_props
                )
                word.symbols.append(symbol)
            except RuntimeError as e:
                # Happens if no text was detected
                log.debug(e)

            if not it.Next(RIL.SYMBOL):
                break
    return blocks


@decorators.python_2_unicode_compatible
class TextElement(model.Rect):
    """Abstract base class for all text elements."""

    def __init__(self, text, left, right, top, bottom, orientation, writing_direction, textline_order, deskew_angle,
                 confidence):
        """

        :param string text: Recognized text content.
        :param int left: Left edge of bounding box.
        :param int right: Right edge of bounding box.
        :param int top: Top edge of bounding box.
        :param int bottom: Bottom edge of bounding box.
        :param Orientation orientation: Orientation of this element.
        :param WritingDirection writing_direction: Writing direction of this element.
        :param TextlineOrder textline_order: Text line order of this element.
        :param float deskew_angle: Angle required to make text upright in radians.
        :param float confidence: Mean confidence for the text in this element. Probability 0-100%.
        """
        super(TextElement, self).__init__(left, right, top, bottom)
        self.text = text
        self.orientation = orientation
        self.writing_direction = writing_direction
        self.textline_order = textline_order
        self.deskew_angle = deskew_angle
        self.confidence = confidence

    def __repr__(self):
        return '<%s: %r>' % (self.__class__.__name__, self.text)

    def __str__(self):
        return '<%s: %r>' % (self.__class__.__name__, self.text)


class TextBlock(TextElement, collections.MutableSequence):
    """Text block."""

    def __init__(self, text, left, right, top, bottom, orientation, writing_direction, textline_order, deskew_angle,
                 confidence):
        """

        :param string text: Recognized text content.
        :param int left: Left edge of bounding box.
        :param int right: Right edge of bounding box.
        :param int top: Top edge of bounding box.
        :param int bottom: Bottom edge of bounding box.
        :param Orientation orientation: Orientation of this element.
        :param WritingDirection writing_direction: Writing direction of this element.
        :param TextlineOrder textline_order: Text line order of this element.
        :param float deskew_angle: Angle required to make text upright in radians.
        :param float confidence: Mean confidence for the text in this element. Probability 0-100%.
        """
        super(TextBlock, self).__init__(text, left, right, top, bottom, orientation, writing_direction, textline_order,
                                        deskew_angle, confidence)
        self.paragraphs = []

    def __getitem__(self, index):
        return self.paragraphs[index]

    def __setitem__(self, index, value):
        self.paragraphs[index] = value

    def __delitem__(self, index):
        del self.paragraphs[index]

    def __len__(self):
        return len(self.paragraphs)

    def insert(self, index, value):
        self.paragraphs.insert(index, value)


class TextParagraph(TextElement, collections.MutableSequence):
    """Text paragraph.

    :param string text: Recognized text content.
    :param int left: Left edge of bounding box.
    :param int right: Right edge of bounding box.
    :param int top: Top edge of bounding box.
    :param int bottom: Bottom edge of bounding box.
    :param Orientation orientation: Orientation of this element.
    :param WritingDirection writing_direction: Writing direction of this element.
    :param TextlineOrder textline_order: Text line order of this element.
    :param float deskew_angle: Angle required to make text upright in radians.
    :param float confidence: Mean confidence for the text in this element. Probability 0-100%.
    :param bool is_ltr: Whether this paragraph text is left to right.
    :param Justification justification: Paragraph justification.
    :param bool is_list_item: Whether this paragraph is part of a list.
    :param bool is_crown: Whether the first line is aligned with the subsequent lines yet other paragraphs are indented.
    :param int first_line_indent: Indent of first line in pixels.
    """

    def __init__(self, text, left, right, top, bottom, orientation, writing_direction, textline_order, deskew_angle,
                 confidence, is_ltr, justification, is_list_item, is_crown, first_line_indent):
        super(TextParagraph, self).__init__(text, left, right, top, bottom, orientation, writing_direction,
                                            textline_order, deskew_angle,  confidence)
        self.lines = []
        self.is_ltr = is_ltr
        self.justification = justification
        self.is_list_item = is_list_item
        self.is_crown = is_crown
        self.first_line_indent = first_line_indent

    def __getitem__(self, index):
        return self.lines[index]

    def __setitem__(self, index, value):
        self.lines[index] = value

    def __delitem__(self, index):
        del self.lines[index]

    def __len__(self):
        return len(self.lines)

    def insert(self, index, value):
        self.lines.insert(index, value)


class TextLine(TextElement, collections.MutableSequence):
    """Text line."""

    def __init__(self, text, left, right, top, bottom, orientation, writing_direction, textline_order,  deskew_angle,
                 confidence):
        """

        :param string text: Recognized text content.
        :param int left: Left edge of bounding box.
        :param int right: Right edge of bounding box.
        :param int top: Top edge of bounding box.
        :param int bottom: Bottom edge of bounding box.
        :param Orientation orientation: Orientation of this element.
        :param WritingDirection writing_direction: Writing direction of this element.
        :param TextlineOrder textline_order: Text line order of this element.
        :param float deskew_angle: Angle required to make text upright in radians.
        :param float confidence: Mean confidence for the text in this element. Probability 0-100%.
        """
        super(TextLine, self).__init__(text, left, right, top, bottom, orientation, writing_direction, textline_order,
                                       deskew_angle, confidence)
        self.words = []

    def __getitem__(self, index):
        return self.words[index]

    def __setitem__(self, index, value):
        self.words[index] = value

    def __delitem__(self, index):
        del self.words[index]

    def __len__(self):
        return len(self.words)

    def insert(self, index, value):
        self.words.insert(index, value)


class TextWord(TextElement, collections.MutableSequence):
    """Text word."""

    def __init__(self, text, left, right, top, bottom, orientation, writing_direction, textline_order, deskew_angle,
                 confidence, language, from_dictionary, numeric, font_name=None, bold=None, italic=None,
                 underlined=None, monospace=None, serif=None, smallcaps=None, pointsize=None, font_id=None):
        """

        :param string text: Recognized text content.
        :param int left: Left edge of bounding box.
        :param int right: Right edge of bounding box.
        :param int top: Top edge of bounding box.
        :param int bottom: Bottom edge of bounding box.
        :param Orientation orientation: Orientation of this element.
        :param WritingDirection writing_direction: Writing direction of this element.
        :param TextlineOrder textline_order: Text line order of this element.
        :param float deskew_angle: Angle required to make text upright in radians.
        :param float confidence: Mean confidence for the text in this element. Probability 0-100%.
        :param language: Language used to recognize this word.
        :param from_dictionary: Whether this word was found in a dictionary.
        :param numeric: Whether this word is numeric.
        :param string font_name: Font name.
        :param bool bold: Whether this word is bold.
        :param bool italic: Whether this word is italic.
        :param underlined: Whether this word is underlined.
        :param monospace: Whether this word is in a monospace font.
        :param serif: Whethet this word is in a serif font.
        :param smallcaps: Whether this word is in small caps.
        :param pointsize: Font size in points (1/72 inch).
        :param font_id: Font ID.
        """
        super(TextWord, self).__init__(text, left, right, top, bottom, orientation, writing_direction, textline_order,
                                       deskew_angle, confidence)
        self.symbols = []
        self.font_name = font_name
        self.bold = bold
        self.italic = italic
        self.underlined = underlined
        self.monospace = monospace
        self.serif = serif
        self.smallcaps = smallcaps
        self.pointsize = pointsize
        self.font_id = font_id
        self.language = language
        self.from_dictionary = from_dictionary
        self.numeric = numeric

    def __getitem__(self, index):
        return self.symbols[index]

    def __setitem__(self, index, value):
        self.symbols[index] = value

    def __delitem__(self, index):
        del self.symbols[index]

    def __len__(self):
        return len(self.symbols)

    def insert(self, index, value):
        self.symbols.insert(index, value)


class TextSymbol(TextElement):
    """Text symbol."""

    def __init__(self, text, left, right, top, bottom, orientation, writing_direction, textline_order, deskew_angle,
                 confidence, is_dropcap, is_subscript, is_superscript):
        """

        :param string text: Recognized text content.
        :param int left: Left edge of bounding box.
        :param int right: Right edge of bounding box.
        :param int top: Top edge of bounding box.
        :param int bottom: Bottom edge of bounding box.
        :param Orientation orientation: Orientation of this element.
        :param WritingDirection writing_direction: Writing direction of this element.
        :param TextlineOrder textline_order: Text line order of this element.
        :param float deskew_angle: Angle required to make text upright in radians.
        :param float confidence: Mean confidence for the text in this element. Probability 0-100%.
        :param bool is_dropcap: Whether this symbol is a dropcap.
        :param bool is_subscript: Whether this symbol is subscript.
        :param bool is_superscript: Whether this symbol is superscript.
        """
        super(TextSymbol, self).__init__(text, left, right, top, bottom, orientation, writing_direction, textline_order,
                                         deskew_angle, confidence)
        self.is_dropcap = is_dropcap
        self.is_subscript = is_subscript
        self.is_superscript = is_superscript


