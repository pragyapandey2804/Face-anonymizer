import cv2
import mediapipe as mp

# Read the image
img_path = './test_img.jpeg'
img = cv2.imread(img_path)

H, W, _ = img.shape  # Get the dimensions of the image

# Detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB as mediapipe expects RGB images
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            # Convert relative bounding box to absolute pixel coordinates
            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            x2 = int((bbox.xmin + bbox.width) * W)
            y2 = int((bbox.ymin + bbox.height) * H)

            # Draw the bounding box
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
