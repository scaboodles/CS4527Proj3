#   target vid is time lapse from 5:00 AM to 9:00pm
import cv2
import numpy as np

def extract_frames():
    cap = cv2.VideoCapture("./foggyTrimCrop2.mp4")
    
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("num frames expected: ", video_length)

    frame_num = 0
    while True:
        success, frame = cap.read()
        
        if not success:
            print("num frames actual:", frame_num)
            break
        
        img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    
        # Apply histogram equalization on the luminance channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        
        # Convert back to BGR color space
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        # Increase contrast
        alpha = 1.5 
        beta = 0    
        #img_output = cv2.convertScaleAbs(img_output, alpha=alpha, beta=beta)

        lower_bound = np.array([200, 200, 200], dtype="uint8")
        upper_bound = np.array([255, 255, 255], dtype="uint8")
        
        # Invert colors near gray/white
        #img_output = invert_near_gray_white(img_output, lower_bound, upper_bound)

        threshold_black = np.array([50, 50, 50], dtype="uint8")
    
        # Turn near-black colors to pure black
        #img_output = turn_near_black_to_black(img_output, threshold_black)

        frame_filename = f"./frames/{frame_num:04d}.jpg"
        cv2.imwrite(frame_filename, img_output)
        
        frame_num += 1
    
    cap.release()
    cv2.destroyAllWindows()

extract_frames()