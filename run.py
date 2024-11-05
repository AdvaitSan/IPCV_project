import cv2
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict

# Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize GUI
def select_video_source():
    def start_detection(source):
        root.destroy()
        detect_and_track_objects(source)
    
    root = tk.Tk()
    root.title("Select Video Source")
    tk.Label(root, text="Choose source for object detection:").pack()
    tk.Button(root, text="Video File", command=lambda: start_detection(filedialog.askopenfilename())).pack()
    tk.Button(root, text="Webcam", command=lambda: start_detection(0)).pack()
    root.mainloop()

# Object Detection and Tracking Function
def detect_and_track_objects(source):
    cap = cv2.VideoCapture(source)
    prev_frame_time = 0
    fps_list = []
    
    # Create the main window
    main_window = tk.Tk()
    main_window.title("Real-Time Object Detection and Data Display")
    
    # Create frames for video and data display
    video_frame = tk.Frame(main_window)
    video_frame.grid(row=0, column=0)
    data_frame = tk.Frame(main_window)
    data_frame.grid(row=0, column=1, padx=10, sticky='n')
    
    # Label for video display
    video_label = tk.Label(video_frame)
    video_label.pack()
    
    # Labels for real-time data display
    fps_label = tk.Label(data_frame, text="FPS: 0")
    fps_label.pack(anchor='nw')
    object_count_label = tk.Label(data_frame, text="Objects Detected:")
    object_count_label.pack(anchor='nw')
    
    object_counts = defaultdict(int)

    def update_data_window(fps, object_counts):
        fps_label.config(text=f"FPS: {int(fps)}")
        
        # Update object count display
        object_count_text = "\n".join([f"{label}: {count}" for label, count in object_counts.items()])
        object_count_label.config(text=f"Objects Detected:\n{object_count_text}")
        main_window.update_idletasks()

    def update_video_frame(frame):
        # Convert frame to RGB and then to ImageTk format
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # Update the video label with the new frame
        video_label.img_tk = img_tk  # Keep a reference to avoid garbage collection
        video_label.config(image=img_tk)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame, size=640)
        boxes = results.pandas().xyxy[0]  # Extract bounding boxes
        
        # Reset object counts for each frame
        object_counts.clear()
        
        # Iterate over detected objects
        for index, row in boxes.iterrows():
            label = row['name']
            confidence = row['confidence']
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            object_counts[label] += 1
        
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time else 0
        prev_frame_time = new_frame_time
        fps_list.append(fps)
        
        # Update data display
        update_data_window(fps, object_counts)
        
        # Update video display
        update_video_frame(frame)
        
        # Close on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    main_window.destroy()
    
    # Performance Analysis
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"Average FPS: {avg_fps:.2f}")

    # Plot FPS over time
    plt.figure()
    plt.plot(fps_list, label='FPS over Time')
    plt.xlabel("Frame")
    plt.ylabel("FPS")
    plt.title("Frame Rate Performance")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    select_video_source()
