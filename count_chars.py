import os
from collections import Counter

PATH = os.path.join(os.getcwd(),  'images/tess_train_images/high_res/crops/text/all_text.txt')

with open(PATH) as file:
    text = file.read()

count = Counter(text)
print(count)

ALPHABET_UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALPHABET_LOWER = ALPHABET_UPPER.lower()
DIGITS = '0123456789'
SUBSCRIPT = '₁₂₃₄₅₆₇₈₉ₓ₋'
SUPERSCRIPT = '⁰°'
ASSIGNMENT = ':=-'
CONCENTRATION = '%()<>'
BRACKETS ='[]{}'
SEPARATORS = ',. '
OTHER = r'\'`/@'

CONDITIONS_WHITELIST = DIGITS + ALPHABET_UPPER + ALPHABET_LOWER + CONCENTRATION + SEPARATORS + BRACKETS

for char in CONDITIONS_WHITELIST:
    if char not in count:
        print(char)
