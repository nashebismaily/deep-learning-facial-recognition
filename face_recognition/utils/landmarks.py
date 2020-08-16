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