from __future__ import print_function
# import the necessary packages
import datetime
class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0
    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self
    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()
    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1
    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()
    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()

# import the necessary packages
from threading import Thread
import cv2
class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        self.stream.set(cv2.CAP_PROP_PAN, 0)
        self.stream.set(cv2.CAP_PROP_TILT, 36000)
        self.stream.set(cv2.CAP_PROP_ZOOM, 0)
        self.stream.set(cv2.CAP_PROP_FOCUS, 0)
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        (self.grabbed, self.frame) = self.stream.read()
        self.count = 0
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            self.count += 1
    def read(self):
        # return the frame most recently read
        return self.count, self.frame
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

# import the necessary packages
import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=300,
    help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=1,
    help="Whether or not frames should be displayed")
args = vars(ap.parse_args())


if 0:
    # grab a pointer to the video stream and initialize the FPS counter
    print("[INFO] sampling frames from webcam...")
    stream = cv2.VideoCapture(0)
    stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    stream.set(cv2.CAP_PROP_PAN, 0)
    stream.set(cv2.CAP_PROP_TILT, 36000)
    stream.set(cv2.CAP_PROP_ZOOM, 0)
    stream.set(cv2.CAP_PROP_FOCUS, 0)
    stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    stream.set(cv2.CAP_PROP_FPS, 30)
    fps = FPS().start()
    # loop over some frames
    while fps._numFrames < args["num_frames"]:
        # grab the frame from the stream and resize it to have a maximum
        # width of 400 pixels
        (grabbed, frame) = stream.read()
        print(frame.shape)
        #frame = imutils.resize(frame, width=400)
        # check to see if the frame should be displayed to our screen
        if args["display"] > 0:
            cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
        # update the FPS counter
        fps.update()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    stream.release()
    cv2.destroyAllWindows()
else:
    # created a *threaded* video stream, allow the camera sensor to warmup,
    # and start the FPS counter
    print("[INFO] sampling THREADED frames from webcam...")
    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    # loop over some frames...this time using the threaded stream
    old_index = -1
    while fps._numFrames < args["num_frames"]:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        index, frame = vs.read()
        #print(frame.shape)
        #frame = imutils.resize(frame, width=400)
        # check to see if the frame should be displayed to our screen
        if args["display"] > 0:
            cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
        # update the FPS counter
        if old_index != index:
            fps.update()
            print("A ")
        else:
            print("B ", end='')
        old_index = index
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()