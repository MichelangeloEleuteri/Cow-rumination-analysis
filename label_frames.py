print("▶ label_frames.py is running")
import os
import cv2

# === Configuration ===
FRAME_DIR = r"C:\Users\eleut\OneDrive\Desktop\Cows\rumination_project\frames2"
LABEL_DIR = r"C:\Users\eleut\OneDrive\Desktop\Cows\rumination_project\labeled_frames"
WINDOW_NAME = "Label frames: 1=start, 2=middle, 3=end, 0=no_rum, ←=back, q=quit"
RESIZE_WIDTH = 800  # Set to None if you don’t want to resize images

# === Setup ===
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1000, 800)

os.makedirs(LABEL_DIR, exist_ok=True)
for label in ['start', 'middle', 'end', 'no_rum']:
    os.makedirs(os.path.join(LABEL_DIR, label), exist_ok=True)

# === Load frame list ===
frame_files = sorted([
    f for f in os.listdir(FRAME_DIR)
    if f.endswith(".jpg") or f.endswith(".png")
])

print(f"Found {len(frame_files)} frames in {FRAME_DIR}")
if not frame_files:
    exit()

index = 0
history = []  # stores (filename, label)

while 0 <= index < len(frame_files):
    fname = frame_files[index]
    fpath = os.path.join(FRAME_DIR, fname)
    frame = cv2.imread(fpath)

    if RESIZE_WIDTH:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # Display filename on top-left corner
    cv2.putText(frame, fname, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('1'):
        label = 'start'
    elif key == ord('2'):
        label = 'middle'
    elif key == ord('3'):
        label = 'end'
    elif key == ord('0'):
        label = 'no_rum'
    else:
        print("Invalid key. Use 1,2,3,0,or q.")
        continue

    # Save labeled frame
    dest_path = os.path.join(LABEL_DIR, label, fname)
    cv2.imwrite(dest_path, cv2.imread(fpath))  # original frame
    history.append((fname, label))
    print(f"Labeled {fname} as {label}")

    index += 1

cv2.destroyAllWindows()
print("Labeling complete.")
