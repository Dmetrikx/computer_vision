# YOLOv5 Window Capture and Object Detection

This repository contains a Python script that captures frames from a specified window, processes them using the YOLOv5 model for object detection, and displays the results. We use multi-threading to achieve a result closer to real-time than waiting for the model to process a frame before moving on to the next frame.

![image](https://github.com/user-attachments/assets/7d6aa469-f95d-4856-9301-0057ed591bf4)
EXAMPLE VIDEO CREDIT: https://www.youtube.com/watch?v=6y5CqAHxGX0

## Requirements

- Python 3.x
- OpenCV
- PyTorch
- MSS
- Numpy
- PyGetWindow

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Dmetrikx/computer_vision
    cd computer_vision
    ```

2. Install the required packages:
    ```sh
    pip install opencv-python-headless torch mss numpy pygetwindow
    ```

## Usage

1. Run the script:
    ```sh
    python computer_vision_picture_in_picture.py
    ```

2. Ensure that the window you want to capture is titled "Picture in Picture". You can change the window title by modifying the `get_window_bbox` function's `window_title` parameter.

## Script Overview

The script performs the following tasks:

1. **Load YOLOv5 Model**: The YOLOv5 model is loaded using PyTorch Hub.
    ```python
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    ```

2. **Get Window Bounding Box**: The bounding box of the specified window is retrieved using the `pygetwindow` library.
    ```python
    bbox = get_window_bbox()
    ```

3. **Capture Frames**: Frames are captured from the specified window using the `mss` library (30 FPS).
    ```python
    def capture_frames():
        sct = mss.mss()
        while True:
            sct_img = sct.grab({
                'left': bbox[0],
                'top': bbox[1],
                'width': bbox[2],
                'height': bbox[3]
            })
            frame = np.array(sct_img)
            frame_queue.put(frame)
            time.sleep(.03)
    ```

4. **Process Frames**: Captured frames are processed using the YOLOv5 model for object detection. Detected objects are drawn on the frames.
    ```python
    def process_frame(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()
        for x1, y1, x2, y2, conf, cls in detections:
            class_name = model.names[int(cls)]
            if class_name not in class_colors:
                class_colors[class_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            label = f"{class_name} {conf:.2f}"
            color = class_colors[class_name]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    ```

5. **Display Frames**: Processed frames are displayed using OpenCV (20 FPS).
    ```python
    def display_frames():
        display_event.wait()
        while True:
            future = result_queue.get()
            frame = future.result()
            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)
    ```

6. **Multithreading**: The script uses multiple threads to capture, process, and display frames concurrently. It is recommended we add some sort of time delay between starting the capture and processing threads and the display threads to build up a set of frames we can render.
    ```python
    capture_thread = threading.Thread(target=capture_frames)
    process_thread = threading.Thread(target=process_frames)
    display_thread = threading.Thread(target=display_frames)
    
    capture_thread.start()
    process_thread.start()
    
    time.sleep(10)
    display_event.set()
    display_thread.start()

    capture_thread.join()
    process_thread.join()
    display_thread.join()
    ```
    processing frames specifically will have 16 threads dedicated to it.
   ```python
   def process_frames():
    with ThreadPoolExecutor(max_workers=16) as executor:
        while True:
            frame = frame_queue.get()
            future = executor.submit(process_frame, frame)
            result_queue.put(future)
   ```

## License

This project is licensed under the MIT License.
