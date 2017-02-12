import os, cv2, matplotlib.pyplot as plt, numpy as np
from numpy import matlib as ml


# automatically set threshold using technique from 
# http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
# just saw URL, and have seen it before, so that's re-assuring that I like it
def auto_canny(img_to_canny, auto_canny_sigma):
    img_to_canny_median = np.median(img_to_canny)
    lower_canny_thresh = int(max(0, (1 - auto_canny_sigma) * img_to_canny_median ))
    upper_canny_thresh = int(max(255, (1 + auto_canny_sigma) * img_to_canny_median ))
    return cv2.Canny(img_to_canny,lower_canny_thresh,upper_canny_thresh)

plt.ion() # needed for my setup, not sure how you're running this.

dir_path = 'imgs'
img_paths = [ os.path.join(dir_path, iname) for iname in os.listdir(dir_path) ]

good_imgs = [
    'R3_12.jpg',
    'R4_12.jpg',
    'R5_12.jpg',
    'R1_11.jpg',
    'R2_11.jpg',
    'R3_11.jpg',
]
img_paths = [ os.path.join(dir_path, iname) for iname in good_imgs ]
##
num_imgs_columns = 3
num_imgs_rows = 3
plt.figure(1)

# load img
img = cv2.cvtColor(cv2.imread(img_paths[3]), cv2.COLOR_BGR2RGB)[420:,:]

plt.subplot(num_imgs_rows,num_imgs_columns,1)
plt.imshow(img)

# bw img
bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.subplot(num_imgs_rows,num_imgs_columns,2)
plt.imshow(bw_img, cmap='gray')

# histogram equalization

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
claheimg = clahe.apply(bw_img)

plt.subplot(num_imgs_rows,num_imgs_columns,3)
plt.imshow(claheimg, cmap='gray')

# canny edge detection

bw_edged = auto_canny(bw_img, 0.33)


plt.subplot(num_imgs_rows,num_imgs_columns,5)
plt.imshow(bw_edged, cmap='gray')

cont2, contours, hierarchy = cv2.findContours(bw_edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
countoured_img = cv2.drawContours(img.copy(), contours, -1, (0,255,0), 1)
plt.subplot(num_imgs_rows,num_imgs_columns,8)
plt.imshow(countoured_img[:,:])

# the histogram equalization definitely makes everythign super sharp, but possibly
# too sharp 
clahe_edged = auto_canny(claheimg, 0.33)

plt.subplot(num_imgs_rows,num_imgs_columns,6)
plt.imshow(clahe_edged, cmap='gray')

blurred_edges = cv2.GaussianBlur(clahe_edged, (21,21), 0)
plt.subplot(num_imgs_rows,num_imgs_columns,9)
plt.imshow(blurred_edges, cmap='gray')

#cv2.bitwise_not(edges.copy())
find_corners = auto_canny(clahe.apply(blurred_edges), 0.33)
# plt.subplot(num_imgs_rows,num_imgs_columns,8)
# plt.imshow(find_corners, cmap='gray')

plt.tight_layout()
