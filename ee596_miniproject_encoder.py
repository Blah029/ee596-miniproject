"""EE506 Miniproject - Image Encoder
E/17/371

Encoding procedure:
    - Break into 8x8 macro blocks
    - Apply DCT to each macroblock
    - Quantise
    - Read in zigzag
    - Apply RLC to AC values, DPCM to DC values
    - Apply Huffman encoding

References:
    - https://docs.python.org/3/library/logging.html
    - https://stackoverflow.com/questions/15978468/using-the-scipy-dct-function-to-create-a-2d-dct-ii
    - https://www.geeksforgeeks.org/print-matrix-in-zig-zag-fashion/
    - https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
    - https://stackoverflow.com/questions/1987694/how-do-i-print-the-full-numpy-array-without-truncation
"""
import cv2
import huffman
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct

import ee596_miniproject_decoder as decoder
import ee596_miniproject_utils as utils


class ImageEncoder:
    def __init__(self, image:np.ndarray, workingdir:str, label:str="image", 
                 blocksize:int=8, qlevels:int=8):
        """Initialise ImageEncoder object"""
        ## Strip alpha channel and convert to YUV
        self.image = image[:,:,:3]
        self.image = cv2.cvtColor(self.image,cv2.COLOR_RGB2YUV)
        self.workingdir = workingdir
        self.label = label
        self.blocksize = blocksize
        self.qlevels = qlevels
        self.height,self.width,_ = image.shape
        self.pad_rows,self.pad_cols = 0,0
        ## Scale to 0-255 range
        if int(np.max(self.image)) == 1:
            self.image = np.int32(self.image*255)
        if self.height%self.blocksize != 0 or self.width%self.blocksize != 0:
            self.zeropad()
            logger.debug(f"image height or width is not divisible by {self.blocksize}. row padding: {self.pad_rows}, column padding: {self.pad_cols}")
        self.blocksegmentation()
        self.applydct()
        self.quantise_table()
        self.blockstozigzag()
        self.code_runlength()
        self.dc_dpcm()
        self.encode_huffman()
        self.write_json()

    def zeropad(self):
        """Pad image with zeros before block segmentation"""
        self.pad_rows = self.blocksize - self.height%self.blocksize
        self.pad_cols = self.blocksize - self.width%self.blocksize
        self.image = np.pad(self.image,
                            ((0, self.pad_rows), (0, self.pad_cols), (0, 0)),
                            mode="constant", constant_values=0)
        self.height,self.width,_ = image.shape
        logger.debug(f"pad rows: {self.pad_rows}, pad columnss: {self.pad_cols}")

    def blocksegmentation(self):
        """Segment image into 8x8 macroblocks"""
        macroblocks = []
        for i in range(0,int(self.height),self.blocksize):
            macroblocks_row = []
            for j in range(0,int(self.width),self.blocksize):
                macroblocks_row.append(self.image[i:i+self.blocksize,
                                             j:j+self.blocksize,:])
            macroblocks.append(macroblocks_row)
        self.macroblocks = np.array(macroblocks).squeeze()
        logger.debug(f"segemented shape: {self.macroblocks.shape}, max: {np.max(self.macroblocks)}")
        logger.debug(f"segmeted: \n{self.macroblocks[0,0,:,:,0]}")

    def applydct(self):
        """Apply discrete cosine transform to segmented blocks"""
        self.macroblocks_dct = dct(dct(self.macroblocks,axis=2),axis=3)
        # logger.debug(f"dct row shape: {self.macroblocks_dct[0].shape}")
        logger.debug(f"dct: \n{self.macroblocks_dct[0,0,:,:,0]}")
        # utils.plot_rgb(self.macroblocks_dct,"DCT")

    def quantise_table(self, qtable_luma:np.ndarray=None, 
                       qtable_chroma:np.ndarray=None):
        """Quantise the transformed macroblocks using the given quantisation
        table, or the standard JPEG quantisation table
        """
        ## Set default quantisation tables
        if qtable_luma == None:
            qtable_luma = utils.qtable_luma
        if qtable_chroma == None:
            qtable_chroma = utils.qtable_chroma
        ## Resize quantisation tables to match macrobloack
        ## Truncate
        # qtable_luma = qtable_luma[:self.blocksize,:self.blocksize]
        # qtable_chroma = qtable_chroma[:self.blocksize,:self.blocksize]
        ## Resize quantisation tables if necessary
        qtable_luma = cv2.resize(qtable_luma.astype(np.int16),
                                 (self.blocksize,self.blocksize))
        qtable_chroma = cv2.resize(qtable_chroma.astype(np.int16),
                                 (self.blocksize,self.blocksize))
        logger.debug(f"luma table shape: {qtable_luma.shape}, chroma table shape: {qtable_chroma.shape}")
        self.qtable = np.stack([qtable_luma,qtable_chroma,qtable_chroma],axis=2)
        logger.debug(f"yuv table shape: {self.qtable.shape}")
        self.macroblocks_quantised = np.empty_like(self.macroblocks_dct, 
                                                   dtype=int)
        ## For RGB image
        # for i,block_row in enumerate(self.macroblocks_dct):
        #     for j,block in enumerate(block_row):
        #         self.macroblocks_quantised[i,j] = np.round(block/qtable_rgb)
        ## For YUV image
        for i,block_row in enumerate(self.macroblocks_dct):
            for j,block in enumerate(block_row):
                self.macroblocks_quantised[i,j] = np.round(block/self.qtable)
        # logger.debug(f"dct: \n{np.around(self.macroblocks_dct[0,0,:,:,0],2)}")
        logger.debug(f"quantised: \n{self.macroblocks_quantised[0,0,:,:,0]}")
        # logger.debug(f"reconstructed: \n{self.macroblocks_quantised[0,0,:,:,0]*self.qtable[:,:,0]}")
        logger.debug(f"ymax: {np.max(self.macroblocks_quantised[:,:,0])}, cbmax: {np.max(self.macroblocks_quantised[:,:,1])}, crmax: {np.max(self.macroblocks_quantised[:,:,2])}")
        # utils.plot_yuv_layers(self.macroblocks_quantised,"Table-quantised")

    def blockstozigzag(self):
        """Acess macroblocks line by line, read the elements in 
        diagonal zigzag, and convert each 8x8 block to a 1x64 array. 
        Each block will be a new row.
        """
        zigzag_rows = self.macroblocks.shape[0] * self.macroblocks.shape[1]
        zigzag_cols = self.macroblocks.shape[2] * self.macroblocks.shape[3]
        self.macroblocks_zigzag = np.empty((zigzag_rows, zigzag_cols, 3),dtype=int)
        for i in range(3):
            j = 0
            for block_row in self.macroblocks_quantised:
                for block in block_row:
                    self.macroblocks_zigzag[j,:,i] = \
                        utils.tozigzag(block[:,:,i])
                    j += 1
        logger.debug(f"macroblocks zigzag shape: {self.macroblocks_zigzag.shape}")
        logger.debug(f"macroblocks zigzag: \n{self.macroblocks_zigzag[0,:,0]}")

    def code_runlength(self):
        """Perform rlc encoding on block values"""
        rlc_rows = self.macroblocks.shape[0] * self.macroblocks.shape[1]
        self.macroblocks_rlc = np.empty((rlc_rows,3), dtype=object)
        for i in range(3):
            for j,blockvalues in enumerate(self.macroblocks_zigzag):
                self.macroblocks_rlc[j,i] = \
                    np.hstack(utils.encode_rlc(blockvalues[:,i]))
        logger.debug(f"macroblocks rlc shape: {self.macroblocks_rlc.shape}")
        logger.debug(f"macroblocks rlc: \n{self.macroblocks_rlc[0,0]}")

    def dc_dpcm(self):
        """Perform differential pulse code modulation on the DC values of 
        each macroblock
        """
        self.macroblocks_dpcm = self.macroblocks_rlc.copy()
        logger.debug(f"dc values input: \n{[self.macroblocks_dpcm[j,0][0]  for j in range(min(self.macroblocks_dpcm.shape[0],5))]}")
        for i in range(3):
            ## Extract the DC values
            dc = np.array([self.macroblocks_dpcm[j, i][0] \
                           for j in range(self.macroblocks_dpcm.shape[0])])
            # logger.debug(f"dpcm input shape: {dc.shape}")
            ## Apply the dpcm to DC values
            dc_encoded = utils.encode_dpcm(dc)
            ## Replace the first elements of each sub-array with the encoded value
            for j in range(self.macroblocks_dpcm.shape[0]):
                self.macroblocks_dpcm[j, i][0] = dc_encoded[j]
            # if i == 0:
            #     logger.debug(f"dpcm input: \n{dc[:5]}")
            #     logger.debug(f"dpcm output: \n{dc_encoded[:5]}")
        logger.debug(f"new dc values: \n{[self.macroblocks_dpcm[j,0][0] for j in range(min(self.macroblocks_dpcm.shape[0],5))]}")
        ## Control logging
        for i in range(1):
            logger.debug(f"coded block {i} shape: {self.macroblocks_dpcm[i,0].shape}")

    def encode_huffman(self):
        """Perform Huffman encoding on the YUV layers of the image"""
        ## Merge each layer into one symbol stream
        symbolstreams = np.empty(3, dtype=object)
        for i in range(3):
            symbolstreams[i] = np.hstack(self.macroblocks_dpcm[:,i])
        logger.debug(f"symbolstreams shape: {symbolstreams.shape}")
        ## Generate codebooks
        ## Vectorise functions to use with arrays
        vect_getprobabilites = np.vectorize(utils.getprobabilities, 
                                            otypes=[object])
        # vect_generatecodebook = np.vectorize(generateCodebook, otypes=[object])
        vect_generatecodebook = np.vectorize(huffman.codebook, otypes=[object])
        probabilities = vect_getprobabilites(symbolstreams)
        self.codebooks = vect_generatecodebook(probabilities)
        logger.debug(f"probabilities: \n{probabilities[0][:5]}")
        logger.debug(f"self.codebooks")
        ## Control logging
        for i,key in enumerate(self.codebooks[0]):
            if i < 5:
                logger.debug(f"{int(key):5d} {self.codebooks[0][key]:>5}")
        self.macroblocks_huffman = np.empty_like(self.macroblocks_dpcm)
        for i in range(3):
            vect_encodesymbols = np.vectorize(utils.encodesymbols, otypes=[object])
            ## Each array in self.macroblocks_dpcm[:,i] uses the same self.codebooks[i]
            ## not sure how that works with np.vectorise. Possible point of failure.
            repeated_codebook = np.repeat(self.codebooks[i],
                                           len(self.macroblocks_huffman[:,i]))
            self.macroblocks_huffman[:,i] = \
                vect_encodesymbols(self.macroblocks_dpcm[:,i],
                                   repeated_codebook)
        logger.debug(f"encoded shape: {self.macroblocks_huffman.shape}")

    def write_json(self):
        """Write header data and encoded data to a json file"""
        data = {
            "resolution": f"{self.height}x{self.width}",
            "padding": f"{self.pad_rows}x{self.pad_cols}",
            "qtable": self.qtable.tolist(),
            "codebooks": self.codebooks.tolist(),
            "bitstream": self.macroblocks_huffman.tolist()
        }
        with open(f"{workingdir}\\Encoded\\{self.label}.json","w") as file:
            json.dump(data,file, indent=0)
        logger.info(f"data written: EE506 Miniproject\\Encoded\\{self.label}.json")


## Set up th logger
logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger("ee596-miniproject-enc")
## Main sequence
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    decoder.logger.setLevel(logging.DEBUG)
    # utils.logger.setLevel(logging.DEBUG)
    ## Turn off numpy scientific notation
    np.set_printoptions(suppress=True)
    ## Set working directory and read image
    workingdir = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Semester 7\\EE596\\EE506 Miniproject"
    image = plt.imread(f"{workingdir}\\Images\\DITF2.png")
    # image = plt.imread(f"{workingdir}\\Images\\Parrots-680x680.jpg")
    label = "Test Image"
    ## Crop image
    # start = np.array([3*60, 71*4])
    # image = image[start[0]:start[0]+16, start[1]:start[1]+16]
    image = image[:360,:640]
    ## Test encoder
    logger.debug(f"input image shape: {image.shape}")
    test_encoder = ImageEncoder(image,workingdir,label)
    ## Test decoder
    logger.debug(f"----------------------------------------------------------")
    test_decoder = decoder.ImageDecoder(f"{workingdir}\\Encoded\\{label}.json",label)
    plt.imshow(test_decoder.image)
    plt.show()