"""EE596 Miniporject - Video Functions
E/17/371
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip

import ee596_mp_image_encoder as imenc
import ee596_mp_image_decoder as imdec
import ee596_mp_image_utils as imutils


def get_frames(video:VideoFileClip):
    """Return a numpy array containing frames of the video"""
    frames = []
    for frame in video.iter_frames():
        frames.append(frame)
    return np.array(frames)


def get_motionvector(block:np.ndarray, frame:np.ndarray, window:np.array):
    """Return the motion vector, given macroblock and frame"""
    # block_height,block_width,_ = block.shape
    # win_y,win_x,win_height,win_width = window
    # for y in range(win_y, min(win_y+win_height, frame.shape[0]-block_height+1)):
    #     for x in range(win_x, min(win_x+win_width, frame.shape[1]-block_width+1)):
    #         if np.array_equal(block, frame[y:y+block_height, x:x+block_width]):
    #             return np.array([y,x])
    min_sad = float('inf')
    motion_vector = (0, 0)
    win_y,win_x,win_height,win_width = window
    for i in range(win_y, min(win_y+win_height, frame.shape[0]-block.shape[0]+1)):
        for j in range(win_x, min(win_x+win_width, frame.shape[1]-block.shape[1]+1)):
            # sad = np.sum(np.abs(frame[i:i+block.shape[0], j:j+block.shape[1]] - block))
            sad = np.sum(np.abs(frame[i,j,:,:] - block))
            # logger.debug(f"block shape: {block.shape}")
            # logger.debug(f"iterating block: \n{frame[i,j,:,:,0]}")
            if sad < min_sad:
                min_sad = sad
                motion_vector = np.array([i, j])
                # logger.debug(f"new min. SAD at: {motion_vector}, SAD: {sad}")
    return motion_vector


def get_video(frames:np.ndarray, fps:int=30):
    """Convert an array of images into a video"""
    video = ImageSequenceClip(frames,fps)
    return video



## Set up th logger
logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger("ee596-mp-vidutils")
## Main sequence
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    imutils.logger.setLevel(logging.DEBUG)
    ## Turn off numpy scientific notation
    np.set_printoptions(suppress=True)
    ## Set working directory and read image
    workingdir = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Semester 7\\EE596\\EE506 Miniproject"
    video = VideoFileClip(f"{workingdir}\\Videos\\GI.mp4", audio=False)
    ## Trim
    # frame_start = video.get_frame(3)
    # imutils.plot_rgb(frame_start)
    # video_trimmed = video.subclip(3,3.33)
    # video_trimmed.write_videofile(f"{workingdir}\\Videos\\GI_trimmed.mp4")
    # logger.debug(f"trimmed frames: {video_trimmed.fps*video_trimmed.duration}")