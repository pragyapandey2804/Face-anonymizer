import cv2
import argparse
import mediapipe as mp
import os

def process_img(img, face_detection):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = face_detection.process(img_rgb)

        H, W, _ = img.shape

        if out.detections is not None :
            for detection in out.detections:
              location_data = detection.location_data
              bbox = location_data.relative_bounding_box
              x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

              x1 = int(bbox.xmin * W)
              y1 = int(bbox.ymin * H)
              x2 = int((bbox.xmin + bbox.width) * W)
              y2 = int((bbox.ymin + bbox.height) * H)

              # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  to draw rectangle around the face
              #blur face
              img[y1:y2, x1:x2, :] = cv2.blur(img[y1:y2, x1:x2, :], (30, 30)) 

        return img   

args = argparse.ArgumentParser()  

# args.add_argument("--mode", default = 'video')
# args.add_argument("--filePath", default = './test_video.mp4')
# args.add_argument("--mode", default = 'image')
# args.add_argument("--filePath", default = './test_img.jpeg')
args.add_argument("--mode", default = 'webcam')
args.add_argument("--filePath", default = None)

args = args.parse_args()

output_dir = "./output"
if not os.path.exists(output_dir):
     os.makedirs(output_dir)

#detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection :
     
    if args.mode in ['image']:
        #read image
        img = cv2.imread(args.filePath)
        
        img = process_img(img, face_detection)
        #save image
        cv2.imwrite(os.path.join(output_dir, "output.png"), img)
     
    elif args.mode in ['video']: 

        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                       25,
                                       (frame.shape[1], frame.shape[0]))
        while ret:
            frame = process_img(frame, face_detection)

            output_video.write(frame)

            ret, frame = cap.read()

        cap.release()
        output_video.release()

    elif args.mode in ['webcam']: 

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        while ret:
            frame = process_img(frame, face_detection)

            cv2.imshow("frame", frame)
            cv2.waitKey(25)

            ret, frame = cap.read()

        cap.release()




             
            


        