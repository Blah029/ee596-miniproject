"""EE596 Miniporject - Video Encoder
E/17/371
"""
import json
import logging
import numpy as np
from moviepy.editor import VideoFileClip

import ee596_mp_image_encoder as imenc
import ee596_mp_image_decoder as imdec
import ee596_mp_image_utils as imutils
import ee596_mp_video_utils as vidutils


class VideoEncoder:
    def __init__(self, video:VideoFileClip, workingdir:str, label:str="video", 
                 blocksize:int=8, qlevels:int=8):
        """Initialise VideoEncoder class"""
        self.video = video
        self.workingdir = workingdir
        self.label = label
        self.process()
        self.write_data()

    def process(self):
        """Separate video into frames, zero-pad if necessary, and split into
        macroblocks
        """
        self.data = {}
        frames = vidutils.get_frames(self.video)
        for i,frame in enumerate(frames):
            frame_encoder = imenc.ImageEncoder(frame,self.workingdir,
                                               f"frame {i}")
            self.data[i] = frame_encoder.data
        
    def write_data(self):
        """Write encoded frames to a file"""
        with open(f"{workingdir}\\Encoded\\{self.label}.json","w") as file:
            json.dump(self.data,file, indent=0)
        logger.info(f"data written: EE506 Miniproject\\Encoded\\{self.label}.json")

## Set up th logger
logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger("ee596-mp-videnc")
## Main sequence
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    # imutils.logger.setLevel(logging.DEBUG)
    vidutils.logger.setLevel(logging.DEBUG)
    ## Turn off numpy scientific notation
    np.set_printoptions(suppress=True)
    ## Set working directory and read image
    workingdir = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Semester 7\\EE596\\EE506 Miniproject"
    video = VideoFileClip(f"{workingdir}\\Videos\\GI_trimmed.mp4", audio=False)
    label = "Test Video"
    # logger.debug(f"video type: {type(video)}")
    logger.debug(f"frames: {video.fps*video.duration}")
    ## Test video encoder
    testencoder = VideoEncoder(video,workingdir,label)
