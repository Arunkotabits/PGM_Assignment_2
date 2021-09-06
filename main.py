import pixellib
from pixellib.semantic import semantic_segmentation
import tensorflow as tf
from skimage import data, io, filters
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import chan_vese

path_to_image = 'elephant1.jpg'
path_to_output_image = 'PGM_Assignment_2/output'

# Segmentation Using Pascalvoc algo.(remove comments to run the code)
# segment_image = semantic_segmentation()
# segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
# segment_image.segmentAsPascalvoc(path_to_image, output_image_name='new_elephant1.jpg')

# Segmentation Using ade20k algo.(remove comments to run the code)
# segment_video = semantic_segmentation()
# segment_video.load_ade20k_model("deeplabv3_xception65_ade20k.h5")
# segment_video.segmentAsAde20k(path_to_image, output_image_name='new_elephant1.jpg')

# Segmentation Using Numpy algo.(remove comments to run the code)
# load the image and convert into
# numpy array
# img = Image.open('elephant1.jpg')
# numpydata = asarray(img)
#
# # data
# print(numpydata.shape)
# print(numpydata)
#
# image = numpydata/255
# edges = filters.sobel(image)
# io.imshow(edges)
# io.show()


# Segmentation Using chanvase algo.(remove comments to run the code)
temp=asarray(Image.open('elephant1.jpg'))
x=temp.shape[0]
y=temp.shape[1]*temp.shape[2]

temp.resize((x, y)) # a 2D array
print(temp)

image = img_as_float(temp)
# Feel free to play around with the parameters to see how they impact the result
cv = chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200,
                dt=0.5, init_level_set="checkerboard",
                extended_output=True)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

print("pass one completed")

ax[0].imshow(image, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

print('pass two completed')

ax[1].imshow(cv[0], cmap="gray")
ax[1].set_axis_off()
title = f'Chan-Vese segmentation - {len(cv[2])} iterations'
ax[1].set_title(title, fontsize=12)

print('pass three completed')

ax[2].imshow(cv[1], cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Final Level Set", fontsize=12)

print('pass four completed')

ax[3].plot(cv[2])
ax[3].set_title("Evolution of energy over iterations", fontsize=12)

print('pass 5 completed')

fig.tight_layout()
plt.show()