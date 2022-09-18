import json
import cv2

from datetime import datetime

video_directory = 'word_videos/'
video_name = "word"
video_extension = ".mp4"

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.current_video_frames = []

    def __del__(self):
        self.video.release()

    def get_frame_with_detection(self):
        check, frame = self.video.read()
        image = frame
        if(check == True):
            _, outputImagetoReturn = cv2.imencode('.jpg', image)
        self.current_video_frames.append(frame)
        print(len(self.current_video_frames))
        return outputImagetoReturn.tobytes(), frame

    def createVideoSnippet(self):
        now = datetime.now()
        datetime_clicked = now.strftime("%d-%m-%Y_%H-%M-%S")
        print("Video snippet button clicked at time: ", datetime_clicked)

        full_video_name = video_directory + video_name + '-' + datetime_clicked + video_extension
        copied_frames = self.current_video_frames.copy()
        self.current_video_frames = []

        width  = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
        fps =  self.video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(full_video_name, fourcc, fps, (width,height))

        for frame in copied_frames:
            print(frame)
            video.write(frame)

        video.release()
        print("Video Released at " + full_video_name)
        return json.dumps({'videoName': full_video_name})

def generate_frames(camera):
    try:
        while True:
            frame, img = camera.get_frame_with_detection()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    except Exception as e:
        print(e)

    finally:
        print("Reached finally, detection stopped")
        cv2.destroyAllWindows()
