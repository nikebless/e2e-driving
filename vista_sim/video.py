import shutil, os, subprocess
import socket
import cv2
import uuid

HOSTNAME = socket.gethostname()
FFMPEG_BIN = '/usr/local/bin/ffmpeg' if HOSTNAME == 'neuron' else 'ffmpeg'
NVIDIA_ACCEL = True if HOSTNAME == 'neuron' else False

class VideoStream:
    '''Class to write images to a video stream. Uses GPU if running on neuron (ffmpeg is not built with CUDA on HPC).

    Some useful links:
    - choosing best options for h264_nvenc: 
        https://superuser.com/questions/1296374/best-settings-for-ffmpeg-with-nvenc
    - h264_nvenc encoding options (note the lossless compression "-preset 10"): 
        https://gist.github.com/nico-lab/e1ba48c33bf2c7e1d9ffdd9c1b8d0493
    '''
    def __init__(self, fps=30):
        # self.tmp = "./tmp"
        self.tmp = os.path.join('.', str(uuid.uuid4().hex)[:8])
        self.fps = fps
        if os.path.exists(self.tmp) and os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp)
        os.mkdir(self.tmp)
        self.index = 0

    def write(self, image):
        cv2.imwrite(os.path.join(self.tmp, f"{self.index:04}.png"), image)
        self.index += 1

    def save(self, fname):
        encoding = '-c:v mpeg4 -q:v 10'
        if NVIDIA_ACCEL:
            # -preset 10 is lossless compression
            # -cq 0 is best quality given compression
            encoding = '-c:v h264_nvenc -preset hq -profile:v high -rc-lookahead 8 -bf 2 -rc vbr -cq 30 -b:v 0 -maxrate 120M -bufsize 240M'
        cmd = f"{FFMPEG_BIN} -f image2 -framerate {self.fps} -i {self.tmp}/%04d.png {encoding} -y {fname}"
        subprocess.call(cmd, shell=True)
        shutil.rmtree(self.tmp)
