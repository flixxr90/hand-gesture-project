import cv2
import pyautogui
import numpy as np
import time
import math

# Screen size 
screen_width, screen_height = pyautogui.size()

# Define skin color range in HSV
lower_skin = np.array([0, 15, 60], dtype=np.uint8)
upper_skin = np.array([40, 40, 200], dtype=np.uint8)











# Webcam
cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
smoothening = 7

print("Hand Gesture Mouse Control Active")
print("Move hand to move mouse, close fist to click")
print("Press ESC to exit")

try:
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (hand)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 500:  # Minimum size filter
                # Get center of contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Convert to screen coordinates
                    screen_x = np.interp(cx, (0, w), (0, screen_width))
                    screen_y = np.interp(cy, (0, h), (0, screen_height))
                    
                    # Smooth movement
                    curr_x = prev_x + (screen_x - prev_x) / smoothening
                    curr_y = prev_y + (screen_y - prev_y) / smoothening
                    
                    # Clamp coordinates to avoid fail-safe trigger at screen edges
                    # Add 10-pixel margin from edges
                    curr_x = max(10, min(curr_x, screen_width - 10))
                    curr_y = max(10, min(curr_y, screen_height - 10))
                    
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y
                    
                    # Get hand shape for click detection
                    # Close fist (small contour) = click
                    x, y, width, height = cv2.boundingRect(largest_contour)
                    aspect_ratio = float(width) / height if height > 0 else 0
                    
                    # Draw contour and center
                    cv2.drawContours(frame, [largest_contour], 0, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Closed fist detection: small area with circular shape
                    if area < 2000 and 0.7 < aspect_ratio < 1.3:
                        pyautogui.click()
                        print("Click!")
                        time.sleep(0.5)
        
        # Show mask for debugging
        cv2.imshow("Hand Gesture Mouse", frame)
        cv2.imshow("Skin Detection", skin_mask)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()