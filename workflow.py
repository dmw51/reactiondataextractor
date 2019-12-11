



import os

from matplotlib import pyplot as plt
from utils.io import imread
from actions import find_solid_arrows, find_reaction_conditions
PATH = os.path.join('/', 'home', 'by256', 'PycharmProjects', 'RDE', 'images')
filename = 'Sample2.jpg'

fig = imread(os.path.join(PATH,filename))

plt.imshow(fig.img,cmap=plt.cm.gray)
plt.show()





# Function to find arrows
arrows = find_solid_arrows(fig)

#Function to find conditions - single arrow only?
#Should take in figure and operate on rectangles?
conditions = find_reaction_conditions(arrows[0],)
