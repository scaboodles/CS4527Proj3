import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor

def midMask(points):
    avg_x = np.mean(points[:, 0, 0])
    avg_y = np.mean(points[:, 0, 1])

    return int(avg_x), int(avg_y)

def find_sky(image, grid_rows, grid_cols):
    lower_blue = np.array([45, 110, 145])
    upper_blue = np.array([135, 206, 235])

    lower_white = np.array([192, 192, 192])
    upper_white = np.array([255, 255, 255])

    lower_black = np.array([0,0,0])
    upper_black = np.array([25,25,25])
    
    coordinates = []
    
    img_height, img_width, _ = image.shape
    
    box_height = img_height // grid_rows
    box_width = img_width // grid_cols
    
    labels = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            start_row, start_col = row * box_height, col * box_width
            end_row, end_col = start_row + box_height, start_col + box_width
            bounding_box = image[start_row:end_row, start_col:end_col]
            
            mask = cv2.inRange(bounding_box, lower_blue, upper_blue)
            points = cv2.findNonZero(mask)
            
            if points is not None and np.sum(mask) > 31365:
                x, y = midMask(points)
                adjusted_point = (x + start_col, y + start_row)
                coordinates.append(adjusted_point)
                labels.append(1)
            
            mask = cv2.inRange(bounding_box, lower_white, upper_white)
            points = cv2.findNonZero(mask)
            
            if points is not None:
                x, y = midMask(points)
                adjusted_point = (x + start_col, y + start_row)
                coordinates.append(adjusted_point)
                labels.append(0)
    return coordinates, labels


def getBluePercent(imgPath, predictor):
    image = cv2.imread(imgPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    coords, labels = find_sky(image, 2, 3)
    if(len(labels) == 0):
        return 0

    input_point = np.array(coords)
    input_label = np.array(labels)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    masked_area = np.sum(masks)
    #print("mask pixels?", masked_area)
    height, width = image.shape[:2]
    total_pixels = height * width
    #print("total all in all", total_pixels)
    percentage = ( masked_area/ total_pixels) * 100
    #print("ratio", percentage)
    return percentage

