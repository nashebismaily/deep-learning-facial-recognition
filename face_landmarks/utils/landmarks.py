import cv2

def overlay(roi, overlay_image):
    resized_dimensions = (roi.shape[1], roi.shape[0])
    resized_overlay_dst = cv2.resize(overlay_image, resized_dimensions, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
    bgr_image = resized_overlay_dst[:, :, :3]
    foreground_alpha_mask = resized_overlay_dst[:, :, 3:]
    background_alpha_mask = cv2.bitwise_not(foreground_alpha_mask)
    foreground_alpha_mask_bgr = cv2.cvtColor(foreground_alpha_mask, cv2.COLOR_GRAY2BGR)
    background_alpha_mask_bgr = cv2.cvtColor(background_alpha_mask, cv2.COLOR_GRAY2BGR)
    background_roi = (roi * 1 / 255.0) * (background_alpha_mask_bgr * 1 / 255.0)
    foreground_roi = (bgr_image * 1 / 255.0) * (foreground_alpha_mask_bgr * 1 / 255.0)
    return cv2.addWeighted(background_roi, 255.0, foreground_roi, 255.0, 0.0)

def overlay_img_above_frame(facial_frame, x, w, y, h, overlay_t_img):
    top_of_frame_coord = max(0, y - h)
    rightmost_frame_coord = x + w
    roi_above_frame = facial_frame[top_of_frame_coord:y,x:rightmost_frame_coord]
    overlayed_roi = overlay(roi_above_frame, overlay_t_img)
    facial_frame[top_of_frame_coord:y, x:rightmost_frame_coord] = overlayed_roi

def get_gray_and_3_chan_colored_frames(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

def get_landmark_color(bgr, landmark):
    # Jawline
    if landmark == 0:
        bgr = (255, 0, 0)  # Jawline
    # Left Eyebrow
    if landmark == 17:
        bgr=(255,255,0)
    # Right Eyebrow
    elif landmark == 22:
        bgr = (0, 255, 0)
    # Nose
    elif landmark == 27:
        bgr = (0, 255, 255)
    # Upper Lip
    elif landmark == 31:
        bgr = (170, 0, 255)
    # Left Eye
    elif landmark == 36:
        bgr = (0, 128, 255)
    # Right Eye
    elif landmark == 42:
        bgr = (0, 0, 255)
    # Mouth
    elif landmark == 48:
        bgr = (255, 230, 255)
    return bgr