"""EE506 Miniproject - Image Decoder
E/17/371
"""
import cv2
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import idct

import ee596_miniproject_utils as utils


class ImageDecoder:
    def __init__(self, path:str, label:str="image"):
        """Initialise ImageDecoder object"""
        self.path = path
        self.label = label
        self.read_json()
        self.decode_huffman()
        self.dc_decode()
        self.decode_runlength()
        self.zigzagtoblocks()
        self.reconstruct()
        self.invertdct()
        self.getfullimage()

    def read_json(self):
        """Read original image dimensions, 3D quantisation table, 
        codebooks of each channel, and encoded bits
        """
        with open(f"{self.path}","r") as file:
            data = json.load(file)
        logger.info(f"data read: {file.name[71:]}")
        ## Read input
        resolution = data["resolution"].split("x")
        padding = data["padding"].split("x")
        self.qtable = np.array(data["qtable"], dtype=int)
        self.codebooks = np.array(data["codebooks"], dtype=object)
        self.macroblocks_huffman = np.array(data["bitstream"], dtype=object)
        ## Pocess input
        # self.height = int(resolution[0])
        # self.width = int(resolution[1])
        self.pad_rows = int(padding[0])
        self.pad_cols = int(padding[1])
        self.height = int(resolution[0]) + self.pad_rows
        self.width = int(resolution[1]) + self.pad_cols
        self.blocksize = self.qtable.shape[0]
        for i,codebook in enumerate(self.codebooks):
            self.codebooks[i] = {int(float(k)):v for k,v in codebook.items()}
        logger.debug(f"height: {self.height}, width: {self.width}")
        logger.debug(f"row padding: {self.pad_rows}, column padding: {self.pad_cols}")
        # logger.debug(f"y channel qtable: \n{self.qtable[:,:,0]}")
        # logger.debug(f"codebooks:")
        # ## Control logging
        # for i,key in enumerate(self.codebooks[0]):
        #     if i < 5:
        #         logger.debug(f"{key:5d} {self.codebooks[0][key]:>5}")
        # logger.debug(f"encoded shape: {self.macroblocks_huffman.shape}")

    def decode_huffman(self):
        """Decode an image encoded using Huffman encoding"""
        self.macroblocks_dpcm = np.empty_like(self.macroblocks_huffman)
        for i in range(3):
            vect_decodesymbols = np.vectorize(utils.decodebitstream, otypes=[object])
            ## Each array in self.macroblocks_huffman[:,i] uses the same codebooks[i]
            ## not sure how that works with np.vectorise. Possible point of failure.
            # macroblocks_dpcm[:,i] = \
            #     vect_decodesymbols(self.macroblocks_huffman[:,i],
            #                        self.codebooks[i],(self.height/self.blocksize,3))
            repeated_codebook = np.repeat(self.codebooks[i],
                                          len(self.macroblocks_huffman[:,i]))
            # logger.debug(f"vecotise argument shapes: {self.macroblocks_huffman[:,i].shape}, {repeated_codebook.shape}")
            self.macroblocks_dpcm[:,i] = \
                vect_decodesymbols(self.macroblocks_huffman[:,i],
                                   repeated_codebook)
        logger.debug(f"huffman decoded shape: {self.macroblocks_dpcm.shape}")

    def dc_decode(self):
        """Decode the DPCM encoded DC values"""
        self.macroblocks_rlc = self.macroblocks_dpcm.copy()
        logger.debug(f"dpcm coded dc values: {[self.macroblocks_rlc[j,0][0]  for j in range(min(self.macroblocks_rlc.shape[0],5))]}")
        for i in range(3):
            ## Extract the DC values
            dc = np.array([self.macroblocks_rlc[j, i][0] \
                           for j in range(self.macroblocks_rlc.shape[0])])
            ## Decode DPCM
            dc_decoded = utils.decode_dpcm(dc)
            ## Replace the first elements of each sub-array with the encoded value
            for j in range(self.macroblocks_rlc.shape[0]):
                self.macroblocks_rlc[j, i][0] = dc_decoded[j]
        logger.debug(f"dpcm decoded dc values: {[self.macroblocks_rlc[j,0][0] for j in range(min(self.macroblocks_rlc.shape[0],5))]}")
        logger.debug(f"dpcm decoded: \n{self.macroblocks_rlc[0,0]}")

    def decode_runlength(self):
        """Decode run length encoded array"""
        vect_decoderlc = np.vectorize(utils.decode_rlc, otypes=[object])
        decoded = vect_decoderlc(self.macroblocks_rlc)
        self.macroblocks_zigzag = \
            np.empty((int(self.height*self.width/self.blocksize**2),
                      self.blocksize**2,
                      3),
                      dtype=int)
        for i in range(self.macroblocks_zigzag.shape[0]):
            for j in range(3):
                self.macroblocks_zigzag[i,:,j] = decoded[i,j]
        logger.debug(f"rlc decoded: \n{self.macroblocks_zigzag[0,:,0]}")

    def zigzagtoblocks(self):
        """Reconstruct macroblocks from 1D diagonal-zigzag arranged values"""
        self.macroblocks_quantised = np.empty((int(self.height/self.blocksize),
                                               int(self.width/self.blocksize),
                                               self.blocksize,
                                               self.blocksize,
                                               3), dtype=int)
        # logger.debug(f"un-zigzagged placeholder shape: {self.macroblocks_quantised.shape}")
        for i in range(3):
            row,col = 0,0
            for j,block in enumerate(self.macroblocks_zigzag):
                # logger.debug(f"iterating row: {row}, column: {col}")
                # logger.debug(f"iterating block shape: {block[:,i].shape}")
                # logger.debug(f"iterating row: {row}, column: {col}")
                self.macroblocks_quantised[row,col,:,:,i] = \
                    utils.fromzigzag(block[:,i],self.blocksize)
                col += 1
                if col == int(self.width/self.blocksize):
                    row +=1
                    col = 0
        logger.debug(f"un-zigzagged shape: {self.macroblocks_quantised.shape}")
        logger.debug(f"un-zigzagged: \n{self.macroblocks_quantised[0,0,:,:,0]}")

    def reconstruct(self):
        """Reconstruct quantised macroblobks using quantisation tables"""
        self.macroblocks_dct = np.empty_like(self.macroblocks_quantised)
        for i,block_row in enumerate(self.macroblocks_quantised):
            for j,block in enumerate(block_row):
                self.macroblocks_dct[i,j] = block*self.qtable
        logger.debug(f"reconstructed: \n{self.macroblocks_dct[0,0,:,:,0]}")

    def invertdct(self):
        """Apply inverse dct to each macroblock"""
        ## Create palceholder
        self.macroblocks = np.empty_like(self.macroblocks_dct)
        self.macroblocks = idct(idct(self.macroblocks_dct,axis=2),axis=3)
        self.macroblocks = np.int32(self.macroblocks/(self.blocksize**2*4))
        logger.debug(f"dct inverted: \n{self.macroblocks[0,0,:,:,0]}")

    def getfullimage(self):
        ## Merge macroblocks and remove padding
        self.image = utils.mergeblocks(self.macroblocks)[:self.height-self.pad_rows,:self.width-self.pad_cols,:]
        ## Preprocess for colourspace conversion
        self.image = np.float32(self.image/255)
        logger.debug(f"image max: {np.max(self.image)}, min: {np.min(self.image)}")
        ## Convert from YUV to RGB
        self.image = cv2.cvtColor(self.image,cv2.COLOR_YUV2RGB)


## Set up th logger
logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger("ee596-miniproject-dec")
## Main sequence
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    # utils.logger.setLevel(logging.DEBUG)
    ## Turn off numpy scientific notation
    np.set_printoptions(suppress=True)
    ## Set working directory and read image
    workingdir = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Semester 7\\EE596\\EE506 Miniproject"
    label = "Test Image"
    test_decoder = ImageDecoder(f"{workingdir}\\Encoded\\{label}.json",label)
    plt.imshow(test_decoder.image)
    plt.show()