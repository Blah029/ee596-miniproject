"""EE596 Miniporject - Video Encoder
E/17/371
"""
import logging
import matplotlib.pyplot as plt
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
        self.blocksize = blocksize
        self.qlevels = qlevels
        self.preporcess()
        self.get_motionvectors()

    def preporcess(self):
        """Separate video into frames, zero-pad if necessary, and split into
        macroblocks
        """
        frames = vidutils.get_frames(self.video)
        self.macroblocks = []
        self.height, self.width, _ = frames[0].shape
        logger.debug(f"frame shape: {self.height,self.width}")
        for frame in frames:
            frame,self.pad_rows,self.pad_cols = \
                imutils.zeropad(frame,self.height,self.width,self.blocksize)
            self.macroblocks.append(imutils.split_macroblocks(frame,
                                                              self.height,
                                                              self.width,
                                                              self.blocksize))
        self.macroblocks = np.array(self.macroblocks)
        logger.debug(f"macroblocks shape: {self.macroblocks.shape}, row pad: {self.pad_rows}, column pad: {self.pad_cols}")            
        logger.debug(f"macroblock: \n{self.macroblocks[0,0,0,:,:,0]}")

    def get_motionvectors(self):
        """Calculate motion vectors for the macroblocks of 
        each subsequent frame
        """
        motionvectors_shape = np.hstack([self.macroblocks.shape[:3],2])
        self.motionvectors = np.zeros(motionvectors_shape, dtype=int)
        # motionvectors_shape = np.hstack([self.macroblocks.shape[:3]])
        self.motionvectors = np.zeros(motionvectors_shape, dtype=object)
        for i,frame in enumerate(self.macroblocks):
            if i > 0:
                for j,block_row in enumerate(frame):
                    for k,block in enumerate(block_row):
                        window = np.array([i,j,8,8])
                        # logger.debug(f"shape block: {block.shape}, frame: {frame.shape}, window: {window.shape}")
                        self.motionvectors[i,j,k] \
                            = vidutils.get_motionvector(block,
                                                        self.macroblocks[i-1],
                                                        window)
                        ## logging control
                    #     logger.debug(f"end of block")
                    #     break
                    # logger.debug(f"end of block row")
                    # break
                break
        logger.debug(f"motionvectors shape: {self.motionvectors.shape}")
        logger.debug(f"motionvectors: \n{self.motionvectors[0,0]}")
        log_y = self.motionvectors[0,0,0,0]
        log_x = self.motionvectors[0,0,0,1]
        logger.debug(f"block at match: \n{self.macroblocks[0,log_y,log_x,:,:,0]}")

    # def find_motion_vector(macroblock, frame, window):
    #     min_sad = float('inf')
    #     motion_vector = (0, 0)
    #     win_y,win_x,win_height,win_width = window
    #     for i in range(win_y, min(win_y+win_height, frame.shape[0]-macroblock.shape[0]+1)):
    #         for j in range(win_x, min(win_x+win_width, frame.shape[1]-macroblock.shape[1]+1)):
    #             sad = np.sum(np.abs(frame[i:i+macroblock.shape[0], j:j+macroblock.shape[1]] - macroblock))
    #             if sad < min_sad:
    #                 min_sad = sad
    #                 motion_vector = (i, j)
    #     return motion_vector

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
