import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


class Yolov8:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)  
        self.class_name = self.model.names

    def detect(self, frame, confidence_threshold=0.7):
        labels = []
        confidences = []
        
        results = self.model(frame)
        
        if results[0].boxes is not None:
            for box in results[0].boxes.data:
                confidence = float(box[4].item())
                if confidence >= confidence_threshold:
                    class_id = int(box[-1].item())
                    label = self.class_name[class_id]
                    labels.append(label)
                    confidences.append(confidence)
        
        return results, labels, confidences


if __name__ == "__main__":
    detector = Yolov8("yolov8n.pt")  

    # Video file path
    video_path = r"bangalore-ki-traffic-trafic-bangalore-koramangala-view-1440-publer.io.mp4"
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results, labels, confidences = detector.detect(frame)

        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (700, 700))

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        clear_output(wait=True)
        plt.imshow(annotated_frame_rgb)
        plt.axis("off")
        display(plt.gcf())

        if cv2.waitKey(1) & 0xFF == ord('r'):
            break

    cap.release()
    cv2.destroyAllWindows()
