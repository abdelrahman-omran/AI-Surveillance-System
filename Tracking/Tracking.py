import cv2
import torch
import numpy as np
import os
import csv
import json
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision
import torchvision.ops  # explicitly import ops
_ = torchvision.extension  # force the extension to load
from ultralytics import YOLO
from boxmot import BotSort
from boxmot.appearance.backbones.osnet import osnet_x0_25
# model = osnet_x0_25(num_classes=751)  # for example, Market-1501 has 751 classes

# # 2. Load the .pth checkpoint (adjust the path as needed)
# checkpoint_path = r"c:\Users\11abd\Downloads\osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth"
# state_dict = torch.load(checkpoint_path, map_location='cpu')
# model.load_state_dict(state_dict)
# model.eval()

# # 3. Create a dummy input tensor that matches the model's expected input shape.
# #    For OSNet used in person re-ID, a common size is [1, 3, 256, 128].
# dummy_input = torch.randn(1, 3, 256, 128)

# # 4. Trace the model with TorchScript
# traced_model = torch.jit.trace(model, dummy_input)

# # 5. Save the traced model as a .pt file
# torch.jit.save(traced_model, r"c:\Users\11abd\Downloads\osnet_converted.pt")

# ======== CUSTOM SETTINGS ========

# (1) Custom YOLOv8 model weights (update the path if needed)
model_weights_path = "best (5).pt"

model = YOLO(model_weights_path)
# Optionally move YOLO to GPU if available:
device = torch.device('cpu')
model.to(device)  # ensure the underlying model is on the proper device

# (2) Custom OSNet weights with .pth extension
custom_osnet_path = Path("osnet_x0_25_market1501_finetuned.pt")

# ======== INPUT/OUTPUT SETUP ========

image_dir = "surveillance-for-retail-stores\tracking\test\01\img1"
image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
if not image_files:
    raise ValueError("Error: No images found in the directory.")

output_video_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
first_frame = cv2.imread(os.path.join(image_dir, image_files[0]))
if first_frame is None:
    raise ValueError("Error: Could not load the first image.")
frame_height, frame_width = first_frame.shape[:2]
out = cv2.VideoWriter(output_video_path, fourcc, 25.0, (frame_width, frame_height))

print("Video writer dimensions:", (frame_width, frame_height))
output_csv = "tracking_output.csv"
csv_file = open(output_csv, "w", newline='')
fieldnames = ["ID", "frame", "objects", "objective"]
writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
writer.writeheader()

# ======== INITIALIZE TRACKER ========
# Use GPU if available:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tracker = BotSort(
    reid_weights=custom_osnet_path,
    device=device,
    half=False,
    frame_rate=25,

    )




# ======== PROCESS EACH IMAGE FRAME ========
frame_counter = 0
for frame_idx, image_file in enumerate(image_files):
    frame_counter += 1
    frame_path = os.path.join(image_dir, image_file)
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Error: Could not load {frame_path}. Skipping...")
        continue
    print("Processing frame:", frame_path, "with shape:", frame.shape)

    # --- Detection with custom YOLOv8 ---
    results = model(frame, verbose=False)
    if results and len(results) > 0 and results[0].boxes is not None:
        # Move detection results to CPU and convert to numpy arrays
        boxes = results[0].boxes.xyxy.cpu().numpy()  # shape: (N, 4)
        confidences = results[0].boxes.conf.cpu().numpy()  # shape: (N,)
    else:
        boxes = np.empty((0, 4))
        confidences = np.empty((0,))

    # Filter detections (confidence threshold >= 0.2)
    if confidences.size > 0:
        keep = confidences >= 0.0
        boxes = boxes[keep]
        confidences = confidences[keep]
    else:
        boxes = np.empty((0, 4))
        confidences = np.empty((0,))

    # --- Prepare detection array for BotSort ---
    if boxes.shape[0] > 0:
        dummy_class = np.zeros((boxes.shape[0], 1))
        scores = confidences.reshape(-1, 1)
        dets = np.hstack([boxes, scores, dummy_class])
    else:
        dets = np.empty((0, 6))

    # --- Update tracker ---
    res = tracker.update(dets, frame)

    # --- Annotate frame (tracker.plot_results modifies frame in-place) ---
    tracker.plot_results(frame, show_trajectories=True)
    annotated_frame = frame.copy()

    # --- Write tracking results to CSV ---
    objects_list = []
    if res is not None and res.shape[0] > 0:
        for det in res:
            x1, y1, x2, y2, track_id, conf, *_ = det
            width = x2 - x1
            height = y2 - y1
            obj = {
                'tracked_id': int(track_id),
                'x': float(x1),
                'y': float(y1),
                'w': float(width),
                'h': float(height),
                'confidence': conf  # BotSort does not output detection confidence
            }
            objects_list.append(obj)
    objects_str = json.dumps(objects_list)
    writer.writerow({
        "ID": frame_idx,
        "frame": float(frame_counter),
        "objects": objects_str,
        "objective": "tracking"
    })

    # --- Write the annotated frame to the video writer ---
    out.write(annotated_frame)

    # --- Display one sample frame inline using matplotlib ---
    # if frame_idx == 1:
    #     display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    #     plt.figure(figsize=(10, 6))
    #     plt.imshow(display_frame)
    #     plt.title("Sample Annotated Frame")
    #     plt.axis("off")
    #     plt.show()

# ======== RELEASE RESOURCES ========
csv_file.close()
out.release()
print(f"Tracking results saved to {output_csv}")
print(f"Output video saved to {output_video_path}")
