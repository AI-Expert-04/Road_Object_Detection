import cv2
import os
import yolov3


def video_processing(video_path, background):
    model = yolov3.YOLO_V3()
    model.build()
    model.load()

    if not os.path.exists('../outputs'):
        os.mkdir('../outputs')

    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('../outputs/output.wmv', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    now_frame = 1

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        h, w = image.shape[:2]

        result_image = model.predict(image)
        out.write(result_image)

        print('(' + str(now_frame) + '/' + str(frame_count) + '): ' + str(now_frame * 100 // frame_count) + '%')
        now_frame += 1

        if not background:
            cv2.imshow('result', result_image)
            if cv2.waitKey(1) == ord('q'):
                break

    out.release()
    cap.release()


if __name__ == '__main__':
    video_path = '../data/videos/cabc9045-581f64de.mov'
    video_processing(video_path, False)
