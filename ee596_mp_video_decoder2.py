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


class VideoDecoder:
    def __init__(self, path:str, label:str="video"):
        """Initialise VideoDecoder class"""
        self.path = path
        self.label = label
        self.read_json()
        self.decodeframes()
        self.write_video()

    def read_json(self):
        """Read encoded frame data"""
        with open(f"{self.path}","r") as file:
            self.data = json.load(file)
            logger.debug(f"video keys: {self.data.keys()}")
        logger.info(f"data read: {file.name[71:]}")
        logger.debug(f"data length: {len(self.data)}")

    def decodeframes(self):
        self.frames = []
        for i in range(len(self.data)):
            frame_decoder = imdec.ImageDecoder(label=self.label)
            # logger.debug(f"{self.data[f'{i}'].keys()}")
            # frame_decoder.data = self.data[f"{i}"]
            frame_decoder.read_json(self.data[f"{i}"])
            frame_decoder.decode()
            self.frames.append(frame_decoder.image)
        self.frames = np.array(self.frames)
        logger.debug(f"sequence shape: {self.frames.shape}")

    def write_video(self):
        self.video = vidutils.get_video(self.frames)
        # self.video.write_videofile(f"{self.workingdir}\\Decoded\\{self.label}.mp4",30)
        self.video.write_videofile(f"D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Semester 7\\EE596\\EE506 Miniproject\\Decoded\\Test Video.mp4",30)
 

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
    ## Set working directory and read data
    workingdir = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Semester 7\\EE596\\EE506 Miniproject"
    label = "Test Video"
    ## Test video decoder
    test_decoder = VideoDecoder(f"{workingdir}\\Encoded\\{label}.json",label)
    
