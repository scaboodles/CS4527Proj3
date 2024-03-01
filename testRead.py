import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor

def midMask(points):
    avg_x = np.mean(points[:, 0, 0])
    avg_y = np.mean(points[:, 0, 1])

    # middle point based on the mean of x and y coordinates
    return int(avg_x), int(avg_y)

def find_sky(image, grid_rows, grid_cols):
    lower_blue = np.array([20, 110, 145])
    upper_blue = np.array([135, 206, 235])

    lower_white = np.array([192, 192, 192])
    upper_white = np.array([255, 255, 255])

    coordinates = []
    
    img_height, img_width, _ = image.shape
    
    # size of each bounding box
    box_height = img_height // grid_rows
    box_width = img_width // grid_cols
    
    labels = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            # bounding box
            start_row, start_col = row * box_height, col * box_width
            end_row, end_col = start_row + box_height, start_col + box_width
            bounding_box = image[start_row:end_row, start_col:end_col]
            
            # blue points within the bounding box
            mask = cv2.inRange(bounding_box, lower_blue, upper_blue)
            points = cv2.findNonZero(mask)
            
            print(row, col, np.sum(mask))
            if points is not None and np.sum(mask) > 31365:
                x, y = midMask(points)
                adjusted_point = (x + start_col, y + start_row)
                coordinates.append(adjusted_point)
                labels.append(1)
            
            #find white/gray points now
            mask = cv2.inRange(bounding_box, lower_white, upper_white)
            points = cv2.findNonZero(mask)
            
            if points is not None:
                x, y = midMask(points)
                adjusted_point = (x + start_col, y + start_row)
                coordinates.append(adjusted_point)
                labels.append(0)

            #now black 
            #mask = cv2.inRange(bounding_box, lower_black, upper_black)
            #points = cv2.findNonZero(mask)
            
            #if points is not None:
                ## Adjust coordinates to match the position in the original image
                #x, y = midMask(points)
                #adjusted_point = (x + start_col, y + start_row)
                #coordinates.append(adjusted_point)
                #labels.append(0)

    return coordinates, labels


#   STOLEN from https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

image = cv2.imread('./frames/1019.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)

coords, labels = find_sky(image, 2, 3)
print(coords, labels)
input_point = np.array(coords)
input_label = np.array(labels)

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
masked_area = np.sum(masks)
print("mask pixels?", masked_area)
height, width = image.shape[:2]
total_pixels = height * width
print("total all in all", total_pixels)
percentage = ( masked_area/ total_pixels) * 100
print("ratio", percentage)
plt.show()
