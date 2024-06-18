import cv2
import numpy as np

def apply_low_pass_filter(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    LPF_kernel = np.array([ [1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]], dtype=np.float32) / 16
    print(LPF_kernel)
    
    filtered_image = cv2.filter2D(gray_image, -1, LPF_kernel)
    
    return filtered_image

def apply_low_pass_filter_5x5(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    LPF_kernel = np.array([ [1, 2, 4, 2, 1],
                            [2, 4, 8, 4, 2],
                            [4, 8, 16, 8, 4],
                            [2, 4, 8, 4, 2],
                            [1, 2, 4, 2, 1]], dtype=np.float32) / 100
    print(LPF_kernel)
    
    filtered_image = cv2.filter2D(gray_image, -1, LPF_kernel)
    
    return filtered_image

def apply_high_pass_filter(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    HPF_kernel = np.array([ [0, 1,  0],
                            [1, -4, 1],
                            [0, 1,  0]], dtype=np.float32) 
    filtered_image = cv2.filter2D(gray_image, -1, HPF_kernel)
    
    return filtered_image



def apply_edge_decection(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    edge_kernel = np.array([[-1, 0,  1],
                            [-1, 0, 1],
                            [-1, 0,  1]], dtype=np.float32)
    
    print(edge_kernel)
    filtered_image = cv2.filter2D(gray_image, -1, edge_kernel)
    
    return filtered_image


image_path = ''  
input_image = cv2.imread(image_path)

# Apply low-pass filter

LPF_filtered_image = apply_low_pass_filter(input_image)
LPF_filtered_image_5x5 = apply_low_pass_filter_5x5(input_image)
HPF_filtered_image= apply_high_pass_filter(input_image)
edge_detedt_image = apply_edge_decection(input_image)

# Display original and filtered images
#cv2.imshow('Original Image', input_image)
#cv2.imshow('Filtered Image Low Pass', LPF_filtered_image)
#cv2.imshow('Filtered Image Low Pass 5x5', LPF_filtered_image_5x5)
#cv2.imshow('Filtered Image High Pass', HPF_filtered_image)
#cv2.imshow('Filtered Image edge_detection', edge_detedt_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

