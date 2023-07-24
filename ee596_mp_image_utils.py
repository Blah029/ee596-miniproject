"""EE596 Miniproject - Functions used in encoders and decoders
E/17/371
"""
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np

## Reset figure number
figno = 1
## Luminance quantisation table
qtable_luma = np.array([[16, 11, 10, 16,  24,  40,  51,  61],
                        [12, 12, 14, 19,  26,  58,  60,  55],
                        [14, 13, 16, 24,  40,  57,  69,  56],
                        [14, 17, 22, 29,  51,  87,  80,  62],
                        [18, 22, 37, 56,  68, 109, 103,  77],
                        [24, 35, 55, 64,  81, 104, 103,  92],
                        [49, 64, 78, 87, 103, 121, 120, 101],
                        [72, 92, 95, 98, 112, 100, 103,  99]])
## Chrominance quantisation table
qtable_chroma = np.array([[17, 18, 24, 47, 99, 99, 99, 99], 
                            [18, 21, 26, 66, 99, 99, 99, 99], 
                            [24, 26, 56, 99, 99, 99, 99, 99], 
                            [47, 66, 99, 99, 99, 99, 99, 99], 
                            [99, 99, 99, 99, 99, 99, 99, 99], 
                            [99, 99, 99, 99, 99, 99, 99, 99], 
                            [99, 99, 99, 99, 99, 99, 99, 99], 
                            [99, 99, 99, 99, 99, 99, 99, 99]])


def zeropad(image:np.ndarray,height:int,width:int, blocksize:int=8):
    """Pad image with zeros before block segmentation"""
    if height%blocksize != 0:
        pad_rows = blocksize - height%blocksize
    else:
        pad_rows = 0
    if width%blocksize != 0:
        pad_cols = blocksize - width%blocksize
    else:
        pad_cols = 0
    image = np.pad(image,
                        ((0, pad_rows), (0, pad_cols), (0, 0)),
                        mode="constant", constant_values=0)
    logger.debug(f"pad rows: {pad_rows}, pad columnss: {pad_cols}")
    return image,pad_rows,pad_cols


def split_macroblocks(image:np.ndarray,height:int,width:int, blocksize:int=8):
    """Segment image into 8x8 macroblocks"""
    macroblocks = []
    for i in range(0,int(height),blocksize):
        macroblocks_row = []
        for j in range(0,int(width),blocksize):
            macroblocks_row.append(image[i:i+blocksize,
                                            j:j+blocksize,:])
        macroblocks.append(macroblocks_row)
    macroblocks = np.array(macroblocks).squeeze()
    logger.debug(f"split shape: {macroblocks.shape}, max: {np.max(macroblocks)}")
    logger.debug(f"split y: \n{macroblocks[0,0,:,:,0]}")
    # logger.debug(f"split cb: \n{macroblocks[0,0,:,:,1]}")
    # logger.debug(f"split cr: \n{macroblocks[0,0,:,:,2]}")
    return macroblocks


def merge_macrblocks(macroblocks:np.ndarray):
    """Merge an array of macroblocks and return full image"""
    merged = []
    for row in macroblocks:
        merged.append(np.hstack(row))
    return (np.vstack(merged))


def plot_rgb(image:np.ndarray, figname:str=None):
    """Merge macroblocks if present, and plot RGB image"""
    global figno
    ## Merge macroblocks
    if len(image.shape) == 5:
        image = merge_macrblocks(image)
    if figname == None:
        plt.figure(f"Figure {figno}")
    else:
        plt.figure(f"Figure {figno} - {figname}")
    plt.imshow(image)
    plt.show()
    figno += 1


def plot_yuv(image:np.ndarray, figname:str=None):
    """Merge macroblocks if present, convert to RBG, 
    and plot given YUV image
    """
    ## Merge macroblocks
    if len(image.shape) == 5:
        image = merge_macrblocks(image)
    image = cv2.cvtColor(np.float32(image),cv2.COLOR_YUV2RGB)
    plot_rgb(image,figname)


def plot_yuv_layers(image:np.ndarray, figname:str=None):
    """Merge macroblocks if present, and plot layers of YUV image"""
    ## Merge macroblocks
    if len(image.shape) == 5:
        image = merge_macrblocks(image)
    fig,ax = plt.subplots(2,2)
    if figname == None:
        fig.suptitle(f"Figure {figno} - YUV layers")
    else:
        fig.suptitle(f"Figure {figno} - {figname} YUV layers")
    ax[0,0].imshow(np.stack([image[:,:,0,],
                         image[:,:,0,],
                         image[:,:,0,]],axis=2))
    ax[0,1].set_axis_off()
    ax[1,0].imshow(np.stack([np.zeros(image[:,:,0].shape),
                           np.zeros(image[:,:,0].shape),
                           image[:,:,1,]], axis=2))
    ax[1,1].imshow(np.stack([image[:,:,2,],
                           np.zeros(image[:,:,0].shape),
                           np.zeros(image[:,:,0].shape)], axis=2))
    plt.show()


def get_qtable(quality_factor:int=50, blocksize:int=8, qtable:np.ndarray=None):
    """Return 3-layer quatisation table for 
    given quality level and macroblock size
    """
    ## Use standard tables, resize if necessary
    global qtable_luma,qtable_chroma
    if qtable == None:
        qtable_luma = cv2.resize(qtable_luma.astype(np.int16),
                                 (blocksize,blocksize))
        qtable_chroma = cv2.resize(qtable_chroma.astype(np.int16),
                                 (blocksize,blocksize))
        qtable = np.stack([qtable_luma,qtable_chroma,qtable_chroma],axis=2)
    logger.debug(f"qtable y: \n{qtable[:,:,0]}")
    ## Adjust for quality level
    if quality_factor < 1 or quality_factor > 100:
        raise ValueError("Quality factor must be between 1 and 100")
    if quality_factor < 50:
        scaling_factor = 5000 / quality_factor
    else:
        scaling_factor = 200 - 2 * quality_factor
    qtable_new = np.zeros_like(qtable, dtype=int)
    logger.debug(f"new qtable y: \n{qtable_new[:,:,0]}")
    for i in range(blocksize):
        for j in range(blocksize):
            for k in range(3):
                qtable_new[i,j,k] = int((scaling_factor * qtable[i,j,k] + 50) / 100)
                if qtable_new[i,j,k] == 0:
                    qtable_new[i,j,k] = 1
                elif qtable_new[i,j,k] > 255:
                    qtable_new[i,j,k] = 255
    return qtable_new


def tozigzag(array:np.ndarray):
    """Read 2D array in diagnal zigzag and return 1D array"""
    n_rows,n_cols = array.shape
    zigzag = []
    for i in range(n_rows + n_cols - 1):
        if i % 2 == 0:
            for col in range(max(0, i - n_rows + 1), min(n_cols, i + 1)):
                row = i - col
                zigzag.append(array[row][col])
        else:
            for row in range(max(0, i - n_cols + 1), min(n_rows, i + 1)):
                col = i - row
                zigzag.append(array[row][col])
    return np.array(zigzag)


def encode_rlc(array:np.ndarray):
    """Perform run length coding on a 1D array and return the result"""
    encoded = []
    previous = None
    for element in array:
        if element != previous:
            if previous != None:
                encoded.append([previous, count])
            previous = element
            count = 1
        else:
            count += 1
    # logger.debug(f"encoded shape: {np.array(encoded).shape}, array shape: {array.shape}")
    if encoded[-1][0] != array[-1]:
        encoded.append([previous, count])
    return np.array(encoded)


def encode_dpcm(array:np.ndarray):
    """Perform differenctial pulse code modulation on 1D array and return"""
    array_dpcm = array.copy()
    for i in range(1,array.size):
        array_dpcm[i] = array[i] - array[i-1]
        # logger.debug(f"array[i]: {array[i]}, array[i-1]: {array[i-1]}, dpcm: {array_dpcm}")
    return array_dpcm


def getprobabilities(array:np.ndarray):
    """Returns a 2D array of symbols and their probabilities"""
    symbols,counts = np.unique(array, return_counts=True)
    probabilities = np.array(list(zip(symbols,counts/array.size)))
    probabilities = probabilities[np.lexsort((probabilities[:,0], 
                                              probabilities[:,1]))][::-1]
    # logger.debug(f"symbols: \n{symbols}")
    # logger.debug(f"counts: \n{counts}")
    # logger.debug(f"probabilities: \n{probabilities}")
    return probabilities


def encodesymbols(symbols:np.ndarray, codebook:dict):
    """Encode a 1D array of symbols into a sring of bits 
    accoring to a codebook
    """
    codewords = []
    for symbol in symbols:
        codewords.append(codebook.get(symbol))
    # logger.debug(f"codebook: {codebook}")
    # logger.debug(f"codewords: {codewords}")
    bitStream = "".join(codewords)
    return bitStream


def decodebitstream(bitstream:str, codebook:dict):
    """Decode a string of bits into a 1D array"""
    inverseCodebook = {v:k for k,v in codebook.items()}
    symbols = []
    receivedBits = ""
    for bit in bitstream:
        receivedBits += bit
        if receivedBits in inverseCodebook:
            symbols.append(inverseCodebook.get(receivedBits))
            receivedBits = ""
            # logger.debug(f"decodeBitstream: {receivedBits} in inverseCodebook")
        else:
            # logger.debug(f"decodeBitstream: {receivedBits} not in inverseCodebook")
            pass
    # logger.debug(f"bitstream \n{bitstream}")
    # logger.debug(f"inverse codebook: {inverseCodebook}")
    # logger.debug(f"decoded symbols: {symbols}")
    return np.int32(np.float32(np.array(symbols)))


def decode_dpcm(array:np.ndarray):
    """Decode DC values encoded with differential pulse code modulation"""
    array_dpcm = array.copy()
    dc_sum = 0
    for i in range(1,array.size):
        dc_sum += array[i-1]
        array_dpcm[i] = array[i] + dc_sum
    return array_dpcm


def decode_rlc(array:np.ndarray):
    """Decode run length coding and return a 1D array"""
    decoded = []
    for i in range(0,len(array),2):
        decoded.append([array[i]]*array[i+1])
    return np.hstack(decoded)


def fromzigzag(array:np.ndarray, blocksize:int=8):
    """Read 1D array and construct macroblock of given size in zigzag"""
    # Create a placeholder array
    result = np.zeros((blocksize,blocksize))
    index = 0
    i = 0
    j = 0
    direction = -1
    while index < len(array):
        result[i, j] = array[index]
        index += 1
        if direction == -1:
            if i == 0 or j == blocksize-1:
                direction = 1
                if j == blocksize-1:
                    i += 1
                else:
                    j += 1
            else:
                i -= 1
                j += 1
        else:
            if j == 0 or i == blocksize-1:
                direction = -1
                if i == blocksize-1:
                    j += 1
                else:
                    i += 1
            else:
                i += 1
                j -= 1
    return result


## Set up th logger
logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger("ee596-mp-imutils")
## Main sequence
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    ## Turn off numpy scientific notation
    np.set_printoptions(suppress=True)
    ## Test zigzag
    testarray = np.array([[11, 12, 13, 14, 15],
                          [21, 22, 23, 24, 25],
                          [31, 32, 33, 34, 35],
                          [41, 42, 43, 44, 45],
                          [51, 52, 53, 54, 55]])
    logger.debug(f"zigzag: \n{tozigzag(testarray)}")
    ## Test rlc
    testarray = np.array([1, 1, 2, 3, 3])
    logger.debug(f"rlc of {testarray}: \n{encode_rlc(testarray)}")
    ## Test dpcm
    logger.debug(f"dpcm of {testarray}: {encode_dpcm(testarray)}")
    ## Test probability
    testarray = np.array([1136, 1, -34, 1, -90, 1, -24, 1, -8, 1, -9, 1, -4, 3, 
                          -10, 1, 0, 1, -2, 1, 9, 1, -3, 1, -1, 1, 0, 3, 3, 1, 
                          6, 1, -5, 1, 0, 1, 3, 4, 0, 38])
    logger.debug(f"probability test array size: {testarray.size}")
    getprobabilities(testarray)
    ## Test huffman decoding
    codebooks = np.array([{"1.0": "0", "0.0": "101", "2.0": "1110", "-1.0": "1101", "-2.0": "10011", "6.0": "111100", "3.0": "110000", "5.0": "100011", "-8.0": "1111100", "11.0": "1111010", "9.0": "1100101", "4.0": "1100111", "-3.0": "1100011", "-4.0": "1100010", "-5.0": "1100100", "20.0": "1000101", "14.0": "11111011", "8.0": "11111010", "7.0": "1000001", "-6.0": "1000000", "-9.0": "1001000", "1078.0": "111111011", "169.0": "111111110", "66.0": "111101100", "64.0": "111101111", "52.0": "111101110", "49.0": "111101101", "46.0": "110011011", "38.0": "110011010", "22.0": "111111001", "18.0": "111111000", "17.0": "11001100", "16.0": "10001001", "13.0": "10001000", "10.0": "10000111", "-7.0": "10000110", "-14.0": "10010011", "-15.0": "10010010", "-19.0": "10000101", "-24.0": "10000100", "-26.0": "10010111", "-27.0": "10010110", "-28.0": "10010101", "-31.0": "111111100", "-33.0": "10010100", "-58.0": "111111111", "-79.0": "111111101", "-99.0": "111111010"}, {"1.0": "0", "0.0": "110", "2.0": "11111", "-1.0": "11101", "3.0": "111101", "12.0": "100101", "9.0": "101000", "8.0": "100111", "6.0": "111001", "4.0": "111000", "-2.0": "101111", "-5.0": "101101", "2062.0": "1000110", "51.0": "1010110", "50.0": "1000001", "45.0": "1001100", "44.0": "1001000", "43.0": "1000010", "42.0": "1111000", "36.0": "1010101", "35.0": "1010100", "13.0": "1000101", "7.0": "1000100", "5.0": "1011101", "-3.0": "1011100", "-4.0": "1000111", "-6.0": "1011001", "-8.0": "1010111", "-10.0": "1010011", "-20.0": "1010010", "-22.0": "1000000", "-29.0": "1001101", "-30.0": "1001001", "-36.0": "1011000", "-48.0": "1000011", "-234.0": "1111001"}, {"1.0": "0", "0.0": "100", "-1.0": "1101", "-2.0": "11110", "2.0": "11100", "5.0": "11001", "-3.0": "111011", "13.0": "1110101", "7.0": "101110", "3.0": "101001", "2113.0": "1011110", "343.0": "1010001", "90.0": "11111110", "59.0": "1011010", "53.0": "1100011", "51.0": "1100001", "43.0": "1010101", "40.0": "1010111", "29.0": "1010110", "28.0": "11111101", "22.0": "11111100", "21.0": "1011111", "14.0": "1011001", "10.0": "1011000", "6.0": "1010000", "4.0": "11111111", "-4.0": "11111011", "-5.0": "1011011", "-11.0": "11111001", "-15.0": "11111000", "-17.0": "1100010", "-31.0": "11111010", "-37.0": "1110100", "-60.0": "1100000", "-387.0": "1010100"}])
    bitstream = np.array([["111111011011111110101111011000110011011010000101010010101011100100101100111110110111101001111101101111100010000000111001010110001101010110101100100011000100100000101010110101010001001101101000111001010001010110101011110110101011110001011100111110101011000101", "10001100100000101001001010011010101000010111010100011101010000110010111101110100011101111111101110001110111111001101000010", "101111001100000010110100101100101110100010111110111100110010111110010111111110100000111100100000100111000010011100110101001110101001001010110"], ["110011010011001010111101110011111100001111100011110110101000100111101111101001100010010000000110011101100100010101100010010011011010111110001100000111100011000110001100001110100110101111000110111101010100110101000101110000001101001110101111100110101010110101011000011111010101110000", "1010111010000000101001001011000010001010101101011001011001011001011110110010111000110100111111110001101111000", "10100010110010111111100101100001010111011111000011101101101011111011010001111001101000110111100100110011111001101010011101011101010011111101"], ["11110001111110100100100001010111101001000101011010110001101100110001000010001000001011110001111010011111000110111101001111101010111011101101010111101001100010100111010111101110011010001010110101011111000010111000011010101111100110111101011100101", "111100101010100010101010100101010100110110111111111010101101011100101101111010011011110111101111111101001100", "101010001011100111011011010111100101110010100101110110111100001000110100111001001010000110101001010101"], ["11111111001111011110111111111010010010010010100010010111010010000111111001011111110001001001101000011001000100001000110110010101110011001000101010001101001101110010111101110111010011011001110001110011010101001110110101011110110100010101001100010110001111010101111011010001011111101011010101100011", "10011101000011010010000110010001000111000011011111100101011100101100111010001101010110", "1110001100010011000110111111000111110100110010100000101101101010010100000110101001100001"]])
    # print(decodebitstream(bitstream[0,0],codebooks[0]))
    logger.debug(f"huffman decoded: \n{decodebitstream(bitstream[0,0],codebooks[0])}")
    ## Test dpcm decoding
    reference = np.array([1078, 1116, 1122, 1291])
    testarray = np.array([1078, 38, 6, 169])
    decoded = decode_dpcm(testarray)
    logger.debug(f"dpcm \nencoded:  {testarray} \ndecoded:  {decoded} \nrefernce: {reference} \nerror:    {reference-decoded}")
    ## Test rlc decoding
    testarray = np.array([1078, 1, -79, 1, 66, 1, 46, 1, -19, 1, -28, 1, 2, 1,
                          -27, 1, 14, 1, 11, 1, 14, 1, -8, 1, -6, 1, 2, 1,
                          0, 1, -3, 1, 0, 1, -1, 1, -5, 1, -4, 1, 7, 1,
                          0, 1, -1, 1, 0, 1, 1, 1, -2, 1, -1, 1, 1, 1,
                          2, 1, 0, 1, 1, 1, 0, 1, -1, 1, 0, 2, -1, 1,
                          0, 2, 1, 1, 0, 4, -1, 1, 0, 20])
    decoded = decode_rlc(testarray)
    logger.debug(f"rlc decoded: \n{decoded}")
    ## Test unding zigzag
    reconstructed = fromzigzag(decoded)
    logger.debug(f"reconstructed: \n{reconstructed}")