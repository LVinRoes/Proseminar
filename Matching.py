import cv2
import numpy as np

def match_template(image, template):
    # Convert images to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Get the dimensions of the template
    w, h = template_gray.shape[::-1]
    
    # Apply template Matching
    res = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # Draw a rectangle around the matched region.
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    matched_image = image.copy()
    cv2.rectangle(matched_image, top_left, bottom_right, (0, 255, 0), 2)
    
    return matched_image, max_val, top_left, bottom_right

# Load the large image and the template image
image_path = ''  # Replace with your large image path
template_path = ''  # Replace with your template image path

image = cv2.imread(image_path)
template = cv2.imread(template_path)

if image is None or template is None:
    print("Could not load one of the images. Please check the file paths.")
else:
    matched_image, max_val, top_left, bottom_right = match_template(image, template)
    
    print(f"Max correlation coefficient: {max_val}")
    print(f"Top left corner of the matched region: {top_left}")
    print(f"Bottom right corner of the matched region: {bottom_right}")

    # Display the result
    cv2.imshow('Matched Image', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

