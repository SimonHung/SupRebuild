# ===========================================================================================
# Presentation Graphic Stream (SUP files) rebuild:
# Rebuild SUP file for compatibility with ExoPlayer versions prior to AndroidX Media 1.2.1
#
# 2024/02/01
# ===========================================================================================

import os
import sys
import logging
import time
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

logging.getLogger().setLevel(logging.WARNING)
logging.basicConfig(format='%(message)s')


# milliseconds to HH:MM:SS.fff
def ms2hmsf(ms, cutms=0):
    ss, ms = divmod(ms, 1000)
    hh, ss = divmod(ss, 3600)
    mm, ss = divmod(ss, 60)
    if cutms:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    else:
        return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


#
# https://stackoverflow.com/questions/34913005/color-space-mapping-ycbcr-to-rgb (ANSER)
# https://web.archive.org/web/20180421030430/http://www.equasys.de/colorconversion.html
# https://en.wikipedia.org/wiki/YCbCr
# https://mymusing.co/bt-709-yuv-to-rgb-conversion-color/
#
# JPG conversion:
#   Y  =  0.299    * R + 0.587    * G + 0.114    * B + 0
#   Cb = -0.168736 * R - 0.331264 * G + 0.5      * B + 128
#   Cr =  0.5      * R - 0.418688 * G - 0.081312 * B + 128
#
# BT.709 conversion:
#   Y  =  0.183 * R + 0.614 * G + 0.062 * B + 16
#   Cb = -0.101 * R - 0.339 * G + 0.439 * B + 128
#   Cr =  0.439 * R - 0.399 * G - 0.040 * B + 128
#
def rgb2YCbCr(im, bt709=1):
    if bt709:
        xform = np.array([[.183, .614, .062], [-.101, -.339, .439], [.439, -.399, -.040]])
        ycbcr = im.dot(xform.T) + [16, 128, 128]
    else:
        xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
        ycbcr = im.dot(xform.T) + [0, 128, 128]

    return np.uint8(ycbcr)


#
# https://stackoverflow.com/questions/34913005/color-space-mapping-ycbcr-to-rgb (ANSER)
# https://web.archive.org/web/20180421030430/http://www.equasys.de/colorconversion.html
# https://en.wikipedia.org/wiki/YCbCr
# https://mymusing.co/bt-709-yuv-to-rgb-conversion-color/
#
# JPG conversion:
#   R = 1 * Y + 0        * (Cb - 128) + 1.402    * (Cr - 128)
#   G = 1 * Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
#   B = 1 * Y + 1.772    * (Cb - 128) + 0        * (Cr - 128)
#
# BT.709 Conversion:
#   R = 1.164 * (Y - 16) + 0     * (Cb - 128) + 1.793  * (Cr - 128)
#   G = 1.164 * (Y - 16) - 0.213 * (Cb - 128) - 0.533  * (Cr - 128)
#   B = 1.164 * (Y - 16) + 2.112 * (Cb - 128) + 0      * (Cr - 128)
#
def yCbCr2RGB(im, bt709=1):
    if bt709:
        xform = np.array([[1.164, 0, 1.793], [1.164, -0.213, -.533], [1.164, 2.112, 0]])
        rgb = im.astype(float) - [16, 128, 128]
    else:
        xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
        rgb = im.astype(float) - [0, 128, 128]

    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


#
# The Run-length encoding method is defined in the US 7912305 B1 patent.
# ref: https://blog.thescorpius.com/index.php/2017/07/15/presentation-graphic-stream-sup-files-bluray-subtitle-format/
#
# code ref: ffmpeg => pgssubdec.c => decode_rle()
#
#   Code                                 Meaning
# --------------------------------------------------------------------------------------
#   CCCCCCCC                             One pixel in color C
#   00000000 00LLLLLL                    L pixels in color 0 (L between 1 and 63)
#   00000000 01LLLLLL LLLLLLLL           L pixels in color 0 (L between 64 and 16383)
#   00000000 10LLLLLL CCCCCCCC           L pixels in color C (L between 3 and 63)
#   00000000 11LLLLLL LLLLLLLL CCCCCCCC	 L pixels in color C (L between 64 and 16383)
#   00000000 00000000                    End of line
#
def rleDecode(rleData, width, height):

    matrixPixels = []
    colPixels = []  # column pixels (width)
    curPos = 0

    while curPos < len(rleData):
        color = rleData[curPos]
        curPos += 1
        runSize = 1

        if color == 0:
            checkFlag = rleData[curPos]
            curPos += 1
            runSize = checkFlag & 0x3f  # xxLLLLLL

            if checkFlag & 0x40:  # x1 xxxxxx
                runSize = (runSize << 8) + rleData[curPos]  # LLLLLL LLLLLLLL
                curPos += 1
            if checkFlag & 0x80:  # 1x xxxxxx
                color = rleData[curPos]
                curPos += 1

        if runSize:
            colPixels.extend([color]*runSize)
        else:
            #  end of line
            if len(colPixels) != width:
                logging.error("[RLE-DECODE] row:%d with wrong column size = %d (width=%d)" %
                              (len(matrixPixels), len(colPixels), width))
                return None
            matrixPixels.append(colPixels)
            colPixels = []

    if colPixels:
        logging.error("[RLE-DECODE] unfinished colum data !")
        return None

    if len(matrixPixels) != height:
        logging.error("[RLE-DECODE] row of matrix = %d not same as height=%d" %
                      (len(matrixPixels), height))
        return None

    return matrixPixels


def rleEncode(matrixPixels, width, height):

    if len(matrixPixels) != height:
        logging.error("[RLE-ENCODE] rowSize(%d) != heigh(%d)" % (len(matrixPixels), height))
        return None

    rleData = []
    for rowIdx, colPixels in enumerate(matrixPixels):
        if len(colPixels) != width:
            logging.error("[RLE-ENCODE] row:%d , length(%d) != width(%d)" % (rowIdx, len(colPixels), width))
            return None
        curPos = 0
        while True:
            color = colPixels[curPos]
            prevPos = curPos
            while (curPos := curPos+1) < width and colPixels[curPos] == color:
                pass

            runSize = curPos - prevPos
            if color == 0:
                if runSize > 63:  # 00000000 01LLLLLL LLLLLLLL ( 0 , 64 <= size <= 16383 )
                    rleData += [0x00, 0x40 | (runSize >> 8), runSize & 0xFF]
                else:  # 00000000 00LLLLLL ( 0 , 1 <= size <= 63 )
                    rleData += [0x00, runSize]
            else:
                if runSize > 63:  # 000000 11LLLLLL LLLLLLLL CCCCCCCC ( C ,  64 <= size <= 16383 )
                    rleData += [0x00, 0xC0 | (runSize >> 8), runSize & 0xFF, color]
                elif runSize > 2:  # 00000000 10LLLLLL CCCCCCCC ( C , 3 <= size <= 63 )
                    rleData += [0x00, 0x80 | runSize, color]
                else:  # one or two pixel C
                    rleData += [color] * runSize

            if curPos >= width:
                rleData += [0x00, 0x00]
                break

    return bytes(rleData)


def mkDir(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError:
            # print("Create dir:'%s' failed !" % folder)
            return 1
    return 0


def createImage(rleImageData, paletteYCrCbA, width, height):
    matrixIndex = np.array(rleDecode(rleImageData, width, height), dtype=np.uint8)
    yCbCr = np.array([(x[0], x[2], x[1]) for x in paletteYCrCbA], dtype=np.uint8)  # swap 1,2 to [y, cb, cr]
    paletteRGB = yCbCr2RGB(yCbCr)
    matrixAlpha = np.array([[paletteYCrCbA[idx][3] for idx in line] for line in matrixIndex], dtype=np.uint8)
    alphaChannel = Image.fromarray(matrixAlpha, mode='L')  # L (8-bit pixels, grayscale)
    img = Image.fromarray(np.array(matrixIndex, dtype=np.uint8), mode='P')  # P (8-bit pixels, mapped to color palette)
    img.putpalette(paletteRGB)
    img.putalpha(alphaChannel)
    img = img.convert("RGBA")
    img = img.convert(mode="P", palette=1, colors=256)

    return img


def createPaletteImage(matrixIndex, matrixAlpha, paletteRGB):
    alphaChannel = Image.fromarray(matrixAlpha, mode='L')  # L (8-bit pixels, grayscale)
    img = Image.fromarray(matrixIndex, mode='P')  # P (8-bit pixels, mapped to color palette)
    img.putpalette(paletteRGB)
    img.putalpha(alphaChannel)
    img = img.convert("RGBA")
    img = img.convert(mode="P", palette=1, colors=256)

    return img


SegmentType = {
    0x14: 'PDS',  # Palette Definition Segment
    0x15: 'ODS',  # Object Definition Segment
    0x16: 'PCS',  # Presentation Composition Segment
    0x17: 'WDS',  # Window Definition Segment
    0x80: 'END'  # End Segment
}

ObjectDataFlag = {
    0x40: 'Last',
    0x80: 'First',
    0xC0: 'First and last'
}

CompositionState = {
    0x00: 'Normal',
    0x40: 'Acquisition Point',
    0x80: 'Epoch Start'
}


class SupRebuild:

    def __init__(self, supFile, createXml, fps):
        self.__createXml = createXml
        self.__fps = fps  # string
        self.__file = supFile

        with open(supFile, 'rb') as fp:
            self.__buf = bytearray(fp.read())
            fp.close()

        if self.__createXml:
            self.__xmlFolder = os.path.dirname(supFile) + "/xmlPng/"
            if mkDir(self.__xmlFolder):
                print("Create dir:'%s' for XML/PNG failed !" % self.__outFolder)
                self.__createXml = False

        self.fixed = 0       # add for pyCharm happy
        self.elapseTime = 0  # add for pyCharm happy
        self.__reset()

        self.__buildBuf = None
        self.__buildBufSize = 0
        self.__buildBufPosition = 0

        self.__headInfo = None
        self.__ptsByteData = b'\x00\x00\x00\x00'

        self.__clearSegmentData()

    def generateNewSUP(self):
        self.__buildBuf = bytearray(len(self.__buf)+65536*2)
        self.__buildBufSize = len(self.__buildBuf)
        self.__buildBufPosition = 0

        self.beginGetDisplaySet()
        lastInfo = {}
        imageInfo = {}
        while (rc := self.getNextDisplaySet()) > 0:

            if self.imageObjectsCount > 0:
                imageInfo = self.__getImageInfo()
                if imageInfo is None:
                    self.__buildBuf = None
                    break  # error

                if self.__createXml:
                    self.__setBdnXmlInfo(imageInfo)
                    self.__saveImage(imageInfo['img'])

                if lastInfo:
                    if lastInfo['pts']+1 < self.pts:
                        # build No image Segment
                        curPtsByteData = self.__ptsByteData   # save byte pts
                        self.__ptsByteData = lastInfo['ptsByteData']
                        self.__buildPCS(b'\x00', 0, lastInfo)  # Normal
                        self.__buildWDS(lastInfo)
                        self.__buildEND()
                        self.__ptsByteData = curPtsByteData  # restore byte pts
                        compositionState = self.compositionState.to_bytes(1, 'big')
                    else:  # continuous subtitles, skip no image segment and set compositionState = 'Acquisition Point'
                        self.fixExoplayerInfo['flashingInfo'].append((self.totalImageCount, ms2hmsf(self.pts)))
                        compositionState = b'\x40'  # Acquisition Point
                else:
                    compositionState = self.compositionState.to_bytes(1, 'big')

                self.__buildPCS(compositionState, 1, imageInfo)
                self.__buildWDS(imageInfo)

                paletteChangeed = self.__fixExoplayerImagePalette(imageInfo)
                self.__buildPDS()

                # build ODS
                if not paletteChangeed and self.imageObjectsCount == 1:   # use original rleData
                    imageObject = self.imageObjects[0]
                    NumberOfsplit = len(imageObject['rleFragmentSize'])
                    if NumberOfsplit == 1:  # only one rle data
                        self.__buildODS(0xC0, imageObject['rleDataLength'],
                                        imageObject['rleImageData'], imageInfo)  # 0xC0: First and Last
                    else:  # more than one data
                        rleDataPos = 0
                        for idx, rleDataSize in enumerate(imageObject['rleFragmentSize']):
                            seqFlag = 0
                            if idx == 0:
                                seqFlag = 0x80  # First
                            if idx == NumberOfsplit-1:
                                seqFlag = seqFlag | 0x40  # Last
                            rleData = imageObject['rleImageData'][rleDataPos:rleDataPos+rleDataSize]
                            self.__buildODS(seqFlag, imageObject['rleDataLength'], rleData, imageInfo)
                            rleDataPos += rleDataSize
                else:  # image object merged, need encode image data
                    if self.imageObjectsCount > 1:
                        self.fixExoplayerInfo['multiObjectInfo'].append((self.totalImageCount, ms2hmsf(self.pts)))
                    maxFragSize = 65515

                    rleData = rleEncode(imageInfo['matrixIndex'], imageInfo['width'], imageInfo['height'])
                    rleDataLength = len(rleData)
                    rleDataPos = 0
                    while rleDataLength - rleDataPos > maxFragSize:
                        seqFlag = 0
                        rleFragData = rleData[rleDataPos: rleDataPos+maxFragSize]
                        if rleDataPos == 0:
                            seqFlag = 0x80  # First
                        rleDataPos += maxFragSize
                        self.__buildODS(seqFlag, rleDataLength, rleFragData, imageInfo)

                    if rleDataPos == 0:  # First & Last
                        self.__buildODS(0xC0, rleDataLength, rleData, imageInfo)
                    else:  # Last
                        self.__buildODS(0x40, rleDataLength - rleDataPos, rleData[rleDataPos:], imageInfo)

                self.__buildEND()
                lastInfo = {}
            else:  # NO IMAGE
                # No image display set, delay save
                lastInfo = {
                    'compositionState': self.compositionState,
                    'pts': self.pts,
                    'ptsByteData': self.__ptsByteData,
                    'offsetX': imageInfo['offsetX'],
                    'offsetY': imageInfo['offsetY'],
                    'width': imageInfo['width'],
                    'height': imageInfo['height']
                }

                if self.__createXml:
                    self.__setBdnXmlInfo(None)

            if logging.root.level > logging.DEBUG:
                self.elapseTime = time.time() - self.__startTime
                print('\rProcessing Image:%d (%.2fs)' % (self.totalImageCount, self.elapseTime), end="")

            # for debug
            # if self.totalImageCount > 15:
            #    break

        else:  # while else
            if logging.root.level > logging.DEBUG:
                print("")
            if rc == 0:  # parser finish
                if self.imageObjectsCount != 0 or CompositionState[self.compositionState] != 'Normal':
                    logging.warning("[PARSER-END] This is not a complete sup file !")
                    if self.__createXml:
                        self.__setBdnXmlInfo(None, 1)  # abnormal
                else:
                    self.__ptsByteData = lastInfo['ptsByteData']
                    self.__buildPCS(b'\x00', 0, lastInfo)  # Normal
                    self.__buildWDS(lastInfo)
                    self.__buildEND()
                    logging.debug("[PARSER-END] SUP file parsing completed.")

                # newBufSize = len(self.__buildBuf)
                del self.__buildBuf[self.__buildBufPosition:]  # cut remaining buffer
                self.fixed = self.__showNewSupMsg()

                # logging.debug("oldBuf size = {:,}, newBuf Size = {:,}, newBufPos = {:,}, size after cut = {:,}".
                #              format(len(self.__buf), newBufSize, self.__buildBufPosition, len(self.__buildBuf)))

                if self.__createXml:
                    self.__createBdnXmlFile()

        self.elapseTime = time.time() - self.__startTime
        return self.__buildBuf

    def genDebugSwap(self):
        self.__buf = self.__buildBuf
        self.__buildBuf = None
        self.generateNewSUP()

    def beginGetDisplaySet(self):
        self.__reset()

    #  return > 0: success, < 0: error , = 0: end
    def getNextDisplaySet(self):
        self.__clearSegmentData()
        while self.__bufSize - self.__bufPosition >= 13:
            self.__headInfo = self.__getHeaderSegment()
            if self.__headInfo is None:
                return -1
            segmentType = self.__headInfo['type']
            size = self.__headInfo['size']
            match segmentType:
                case 'PCS':
                    if not self.__PresentationCompositionSegment(size):
                        return -1
                case 'WDS':
                    if not self.__WindowDefinitionSegment(size):
                        return -1
                case 'PDS':
                    if not self.__PaletteDefinitionSegment(size):
                        return -1
                case 'ODS':
                    if not self.__ObjectDefinitionSegment(size):
                        return -1
                case 'END':
                    if not self.__endSegment(size):
                        return -1

                    break  # display set complete

        if not self.__endCount:  # end of file
            if self.__bufSize != self.__bufPosition:
                logging.error("[FILE-END] Still some data remaining !")
                return -1
            else:
                return 0

        return 1

    def __getImageInfo(self):

        imageInfo = []
        for compObj in self.compositionObjects:
            oid = compObj['oid']
            for imageObject in self.imageObjects:
                if imageObject['oid'] == oid:
                    break
            else:
                logging.error("[IMG-INFO] oid=%d not found (image count=%d) !" % (oid, self.totalImageCount))
                return None

            matrixIndex = rleDecode(imageObject['rleImageData'], imageObject['width'], imageObject['height'])
            matrixAlpha = [[self.palette[idx][3] for idx in line] for line in matrixIndex]
            imageInfo.append({
                'objectVer': imageObject['objectVer'],
                'width': imageObject['width'],
                'height': imageObject['height'],
                'offsetX': compObj['offsetX'],
                'offsetY': compObj['offsetY'],
                'croppedFlag': compObj['croppedFlag'],
                'croppedByteData': compObj['croppedByteData'],
                'matrixIndex': matrixIndex,
                'matrixAlpha': matrixAlpha,
            })

        match len(imageInfo):
            case 0:
                return None
            case 1:
                retImageInfo = imageInfo[0]
            case _:  # combine multiple image objects into a single object
                retImageInfo = self.__getOverlayImageInfo(imageInfo)

        if self.__createXml and retImageInfo:
            matrixIndex = np.array(retImageInfo['matrixIndex'], dtype=np.uint8)
            matrixAlpha = np.array(retImageInfo['matrixAlpha'], dtype=np.uint8)
            yCbCr = np.array([(x[0], x[2], x[1]) for x in self.palette], dtype=np.uint8)  # swap 1,2 to [y, cb, cr]
            retImageInfo['img'] = createPaletteImage(matrixIndex, matrixAlpha, yCbCr2RGB(yCbCr))

        return retImageInfo

    def __getOverlayImageInfo(self, imageInfoList):
        top = left = sys.maxsize
        bottom = right = 0

        for idx, paletteItem in enumerate(self.palette):
            if paletteItem[3] == 0:
                # transparent index found
                transparentIdx = idx
                break
        else:
            logging.error("[OVERLAY-IMAGEINFO] transparent palette not found (imageCount=%d) !", self.totalImageCount)
            return None

        for imageInfo in imageInfoList:
            if top > imageInfo['offsetY']:
                top = imageInfo['offsetY']
            if left > imageInfo['offsetX']:
                left = imageInfo['offsetX']
            if bottom < imageInfo['offsetY'] + imageInfo['height']:
                bottom = imageInfo['offsetY'] + imageInfo['height']
            if right < imageInfo['offsetX'] + imageInfo['width']:
                right = imageInfo['offsetX'] + imageInfo['width']

        newWidth = right - left
        newHeight = bottom - top
        newMatrixIndex = np.full((newHeight, newWidth), transparentIdx, dtype=np.uint8)
        for imageInfo in imageInfoList:
            overlayX0 = imageInfo['offsetX'] - left
            overlayX1 = overlayX0 + imageInfo['width']
            overlayY0 = imageInfo['offsetY'] - top
            overlayY1 = overlayY0 + imageInfo['height']
            newMatrixIndex[overlayY0:overlayY1, overlayX0:overlayX1] = imageInfo['matrixIndex']

        newMatrixAlpha = [[self.palette[idx][3] for idx in line] for line in newMatrixIndex]
        return {
            'objectVer': b'\x00',  # always 0
            'width': newWidth,
            'height': newHeight,
            'offsetX': left,
            'offsetY': top,
            'croppedFlag': 0,  # don't support cropped image
            'croppedByteData': b'',
            'matrixIndex': newMatrixIndex.tolist(),  # list
            'matrixAlpha': newMatrixAlpha,  # list
        }

    def __showNewSupMsg(self):
        fixed = 0
        idx = 0
        for idx, flashingInfo in enumerate(self.fixExoplayerInfo['flashingInfo']):
            fixed += 1
            if not idx:
                print("\nFixing subtitle flashing at:", end='')
            if idx % 5 == 0:
                print("\n    ", end="")
            print("[%d] %s, " % (flashingInfo[0], flashingInfo[1]), end='')
        if idx > 0:
            print("")

        idx = 0
        for idx, multiObjectInfo in enumerate(self.fixExoplayerInfo['multiObjectInfo']):
            fixed += 1
            if not idx:
                print("\nMerge multi-images at:", end='')
            if idx % 5 == 0:
                print("\n    ", end="")
            print("[%d] %s, " % (multiObjectInfo[0], multiObjectInfo[1]), end='')
        if idx > 0:
            print("")

        if self.fixExoplayerInfo['palette']:
            fixed += 1
            print("\nFixing the subtitle color problem", end='')
            if self.fixExoplayerInfo['alpha0AtEnd']:
                print(" for 'ExoPlayer', but may cause 'PotPlayer' to display the color incorrectly !\n")
            else:
                print(".\n")

        return fixed

    def __reset(self):
        self.__bufSize = len(self.__buf)
        self.__bufPosition = 0
        self.__serialNo = 0
        self.__startTime = time.time()

        self.fixExoplayerInfo = {
            'flashingInfo': [],
            'palette': 0,
            'alpha0AtEnd': 0,
            'multiObjectInfo': []
        }

        self.fixed = 0
        self.elapseTime = 0
        self.totalImageCount = 0

        self.bdnXmlInfo = []

    def __clearSegmentData(self):
        self.__pcsCount = self.__wdsCount = self.__pdsCount = self.__odsCount = self.__endCount = 0
        # self.videoWidth = -1   # don't clear, keep for last end segment
        # self.videoHeight = -1  # don't clear, keep for last end segment
        self.compositionObjectCount = 0
        self.compositionObjects = []

        self.paletteId = -1
        self.palette = []

        self.imageObjectsCount = 0
        self.imageObjects = []

    def __getHeaderSegment(self):
        if self.__buf[self.__bufPosition] != 0x50 or self.__buf[self.__bufPosition+1] != 0x47:  # 'PG'
            logging.error("Invalid PG header !")
            return None
        pos = self.__bufSegmentStartPosition = self.__bufPosition
        buf = self.__buf
        self.__bufPosition += 13

        rc = {
            'pts': int(buf[2+pos:6+pos].hex(),  16)//90,
            'dts': int(buf[6+pos:10+pos].hex(), 16)//90,
            'type': SegmentType[buf[10+pos]],
            'size': int(buf[11+pos:13+pos].hex(), 16),
        }
        logging.debug("[PG]= %s-size:0x%04X-pos:0x%06X ------"
                      % (ms2hmsf(rc['pts']), rc['size'], pos))
        return rc

    def __bufTimeUpdate(self):  # NO USE
        curTimeByeData = self.__buf[2+self.__bufSegmentStartPosition:6+self.__bufSegmentStartPosition]
        if curTimeByeData != self.__ptsByteData:
            self.__buf[2 + self.__bufSegmentStartPosition:6+self.__bufSegmentStartPosition] = self.__ptsByteData
            self.bufTimeModified += 1

    def __PresentationCompositionSegment(self, size):
        if self.__pcsCount:
            logging.error("[PCS] Too many PCS segment !")
            return 0

        if size > (self.__bufSize - self.__bufPosition) or size < 11:
            logging.error("[PCS] Invalid segment size !")
            return 0

        pos = self.__bufPosition
        self.pts = self.__headInfo['pts']
        self.__ptsByteData = self.__buf[2+self.__bufSegmentStartPosition:6+self.__bufSegmentStartPosition]

        self.videoWidth = int(self.__buf[pos:pos+2].hex(), 16)
        self.videoHeight = int(self.__buf[pos+2:pos+4].hex(), 16)
        self.frameRate = int(self.__buf[pos+4:pos+5].hex(), 16)
        self.serialNo = int(self.__buf[pos+5:pos+7].hex(), 16)
        self.compositionState = int(self.__buf[pos+7:pos+8].hex(), 16)
        self.paletteUpdateFlag = int(self.__buf[pos+8:pos+9].hex(), 16)
        self.presentationPid = int(self.__buf[pos+9:pos+10].hex(), 16)
        self.compositionObjectCount = int(self.__buf[pos+10:pos+11].hex(), 16)
        self.__bufPosition += 11
        size -= 11

        logging.debug("[PCS]- %d - %s--- %dx%d - frameRate=%d ------"
                      % (self.serialNo, CompositionState[self.compositionState],
                         self.videoWidth, self.videoHeight, self.frameRate))

        self.compositionObjects = []
        for idx in range(self.compositionObjectCount):  # get composition objects
            if size < 8:
                logging.error("[PCS-OBJ] Invalid file size !")
                return 0

            pos = self.__bufPosition
            self.compositionObjects.append({
                'oid': int(self.__buf[pos:pos+2].hex(), 16),
                # skip windows id (1 byte)
                'croppedFlag': int(self.__buf[pos+3:pos+4].hex(), 16),
                'offsetX': int(self.__buf[pos+4:pos+6].hex(), 16),
                'offsetY': int(self.__buf[pos+6:pos+8].hex(), 16),
                'croppedByteData': b''
            })
            compObj = self.compositionObjects[idx]
            logging.debug("[PCS-OBJ] oid:%d, x:%d, y:%d " % (compObj['oid'], compObj['offsetX'], compObj['offsetY']))

            if compObj['croppedFlag'] & 0x80:  # if cropping
                compObj['croppedByteData'] = self.__buf[pos+8:pos+16]
                self.__bufPosition += 8
                size -= 8

            if compObj['offsetX'] > self.videoWidth or compObj['offsetY'] > self.videoHeight:
                logging.error("[PCS-OBJ] offsetX[%d] = %d, offsetY[%d] = %d videoWidth = %d, videoHeight = %d !" %
                              (idx, compObj['offsetX'], idx, compObj['offsetY'],
                               self.videoWidth, self.videoHeight))
                return 0

            self.__bufPosition += 8
            size -= 8

        if size:
            logging.error("[PCS] Invalid segment size, final size=%d !" % size)
            return 0

        self.__pcsCount += 1
        return 1

    def __WindowDefinitionSegment(self, size):
        logging.debug("[WDS]---------------------------------------------")
        if size > (self.__bufSize - self.__bufPosition):
            logging.error("[WDS] Invalid segment size !")
            return 0
        pos = self.__bufPosition
        self.__bufPosition += size

        self.windowsObjectCount = self.__buf[pos]
        self.windowsObjects = []
        pos += 1
        size -= 1
        for idx in range(self.windowsObjectCount):
            if size < 9:
                logging.error("[WDS] Invalid  size of windows info !")
                return 0
            self.windowsObjects.append({
                'wid': self.__buf[pos],
                'offsetX': int(self.__buf[pos+1:pos+3].hex(), 16),
                'offsetY': int(self.__buf[pos+3:pos+5].hex(), 16),
                'width': int(self.__buf[pos+5:pos+7].hex(), 16),
                'height': int(self.__buf[pos+7:pos+9].hex(), 16)
            })
            winObject = self.windowsObjects[idx]
            logging.debug("[WDS] wid:%d, x:%d, y:%d, %d*%d ----------------" %
                          (winObject['wid'], winObject['offsetX'], winObject['offsetY'],
                           winObject['width'], winObject['height']))

            pos += 9
            size -= 9

        self.__wdsCount += 1
        return 1

    def __PaletteDefinitionSegment(self, size):
        logging.debug("[PDS]-------")
        if self.__pdsCount:
            logging.error("[PDS] Too many PDS segment !")
            return 0

        if size > (self.__bufSize - self.__bufPosition):
            logging.error("[PDS] Invalid segment size !")
            return 0

        if (size % 5) != 2 or size < 7:
            logging.error("[PDS] segment size doesn't match (idx, Y, Cr, Cb, alpha0)*n+2, 1 <= n <= 256")
            return 0

        pos = self.__bufPosition
        self.paletteId = self.__buf[pos]
        self.paletteVer = self.__buf[pos+1:pos+2]
        pos += 2
        buf = self.__buf
        self.__paletteByteData = bytearray(buf[pos:pos+size-2])
        self.palette = [[0, 0, 0, 0]] * 256
        for idx in range(size // 5):
            # buf[pos]: index of palette, buf[pos+1]: Y, buf[pos+2]: Cr, buf[pos+3]: Cb, buf[pos+4]: alpha
            pIndex = buf[pos]
            if idx != pIndex:
                print("idx=%d != pIndex=%d" % (idx, pIndex))
            self.palette[buf[pos]] = [buf[pos+1], buf[pos+2], buf[pos+3], buf[pos+4]]
            pos += 5
        self.__bufPosition += size

        self.__pdsCount += 1
        return 1

    def __ObjectDefinitionSegment(self, size):
        if size > (self.__bufSize - self.__bufPosition) or size < 4:
            logging.error("[ODS%d][ID] Invalid segment size !" % self.imageObjectsCount)
            return 0
        pos = self.__bufPosition
        buf = self.__buf
        oid = int(buf[pos:pos+2].hex(), 16)
        objectVer = buf[pos+2:pos+3]
        flag = buf[pos+3]
        size -= 4
        logging.debug("[ODS]- ver:%d - %s ------" % (int(objectVer[0]), ObjectDataFlag[flag]))

        if flag & 0x80:  # First fragment
            # if flag == 0x80: # for debug
            #    print("ONLY FIRST FRAGMENT")
            if size < 7:
                logging.error("[ODS%d][OBJ] Invalid segment size !" % self.imageObjectsCount)
                return 0
            rleDataLength = int(buf[pos + 4:pos + 7].hex(), 16) - 4  # bitmap rle length, include width & height
            if rleDataLength < 4:
                logging.error("[ODS%d][SIZE] Invalid segment size (rle = %d) !" %
                              (self.imageObjectsCount, rleDataLength))
                return 0
            width = int(buf[pos + 7:pos + 9].hex(), 16)
            height = int(buf[pos + 9:pos + 11].hex(), 16)
            if width > self.videoWidth or height > self.videoHeight or width == 0 or height == 0:
                logging.error("[ODS%d] Invalid image dimensions %dX%d !" % (self.imageObjectsCount, width, height))
                return 0
            self.imageObjects.append({
                'oid': oid,
                'objectVer': objectVer,
                'width': width,
                'height': height,
                'rleDataLength': rleDataLength,
                'rleImageData': bytearray(),
                'rleFragmentSize': []   # If the data size is too large, it will be split into several fragments.
            })
            curImageObjectIdx = self.imageObjectsCount
            imgObject = self.imageObjects[curImageObjectIdx]
            logging.debug("[ODS-OBJ] oid:%d, %d*%d " % (imgObject['oid'], imgObject['width'], imgObject['height']))

            self.imageObjectsCount += 1
            self.__bufPosition += 11
            size -= 7
        else:
            for idx, imageObject in enumerate(self.imageObjects):
                if imageObject['oid'] == oid:
                    curImageObjectIdx = idx
                    break
            else:
                logging.error("[ODS%d] oid not found for LAST segment (oid = %d) !" % (self.imageObjectsCount, oid))
                return 0
            self.__bufPosition += 4

        imageObject = self.imageObjects[curImageObjectIdx]
        imageObject['rleFragmentSize'].append(size)
        curDataSize = len(imageObject['rleImageData'])
        rleDataLength = imageObject['rleDataLength']
        if curDataSize + size > rleDataLength:
            logging.error("[ODS%d] Invalid object size rle=%d used=%d append=%d !" %
                          (self.imageObjectsCount, rleDataLength, curDataSize, size))
            return 0
        imageObject['rleImageData'] += self.__buf[self.__bufPosition:self.__bufPosition+size]
        self.__bufPosition += size

        if flag == 0x40 and len(imageObject['rleImageData']) != rleDataLength:  # Last fragment
            logging.error("[ODS%d] Invalid object data size  flag=0x%0X rle=%d image data size = %d !" %
                          (self.imageObjectsCount, flag, rleDataLength, len(imageObject['rleImageData'])))
            return 0

        self.__odsCount += 1
        return 1

    def __endSegment(self, size):
        if size > 0:
            logging.error("[END] Segment size  > 0 (%d) !" % size)
            return 0
        if self.imageObjectsCount:
            self.totalImageCount += 1
            logging.debug("[END]= IMAGE COUNT = %04d, OBJS = %02d ============================="
                          % (self.totalImageCount, self.imageObjectsCount))
        else:
            logging.debug("[END]= NO IMAGE ==================================================")

        self.__endCount += 1
        return 1

    def __fixExoplayerPalette(self):
        paletteItems = len(self.__paletteByteData) // 5
        lowestAlphaIdx = -1
        lowestAlphaValue = 256
        entry0Idx = -1
        needFix = 0
        for itemIdx in range(paletteItems):
            alphaValue = self.__paletteByteData[itemIdx*5+4]
            if alphaValue < lowestAlphaValue:  # find the lowest alpha value
                lowestAlphaValue = alphaValue
                lowestAlphaIdx = itemIdx
            if entry0Idx < 0 and self.__paletteByteData[itemIdx*5] == 0:    # entry number = 0
                entry0Idx = itemIdx
                if self.__paletteByteData[itemIdx*5+4] != 0:  # if alpha of entry0 != 0 (transparent)
                    needFix = 1
        if needFix:
            self.fixExoplayerInfo['palette'] += 1
            if lowestAlphaValue > 0:  # no transparent entry
                if paletteItems >= 256:  # no more entry for add transparent entry
                    logging.warning("[FIX-PALETTE] No entry for fix transparent problem !")
                    return 0, -1, -1  # need fix, but can not fix
                else:
                    # if paletteItems = 255 and just only swap [entry0Idx] with [alpha0Idx (paletteItems)]
                    # the 'PotPlayer' will assume [255] always transparent
                    #
                    #  [entry0Idx]  => [lowestAlphaIdx] ==> [alpha0Idx] ==> [entry0Idx]
                    alpha0Idx = paletteItems
                    logging.debug("[Palette-Rotate] entry0[%d](%d) ==> lowAlpha[%d](%d) ==> alpha0[%d](%d)"
                                  % (entry0Idx, self.__paletteByteData[entry0Idx*5+4],
                                     lowestAlphaIdx, self.__paletteByteData[lowestAlphaIdx*5+4], alpha0Idx, 0))
                    # move the lowest value to new position ([paletteItems])
                    self.__paletteByteData += paletteItems.to_bytes(1, 'big') \
                        + self.__paletteByteData[lowestAlphaIdx*5+1:lowestAlphaIdx*5+5]

                    # move entry0 value to [lowestAlphaIdx]
                    self.__paletteByteData[lowestAlphaIdx*5+1:lowestAlphaIdx*5+5] =  \
                        self.__paletteByteData[entry0Idx*5+1:entry0Idx*5+5]

                    # entry0 set as transparent
                    self.__paletteByteData[entry0Idx*5+1:entry0Idx*5+5] = b'\x00\x00\x00\x00'

                    self.fixExoplayerInfo['alpha0AtEnd'] = 1
                    return 1, alpha0Idx, entry0Idx, lowestAlphaIdx
            else:
                # lowestAlphaIdx equal to alpha0Idx
                logging.debug("[Palette-Swap] entry0[%d](%d) <==> Alpha0[%d](%d)"
                              % (entry0Idx, self.__paletteByteData[entry0Idx * 5 + 4],
                                 lowestAlphaIdx, self.__paletteByteData[lowestAlphaIdx * 5 + 4]))
                entry0IdxNewValue = self.__paletteByteData[lowestAlphaIdx*5+1:lowestAlphaIdx*5+5]
                alpha0IdxNewValue = self.__paletteByteData[entry0Idx*5+1:entry0Idx*5+5]
                self.__paletteByteData[entry0Idx*5+1:entry0Idx*5+5] = entry0IdxNewValue
                self.__paletteByteData[lowestAlphaIdx*5+1:lowestAlphaIdx*5+5] = alpha0IdxNewValue

                return 1, lowestAlphaIdx, entry0Idx, None

        return 0, None, None, None

    def __fixExoplayerImagePalette(self, imageInfo):
        paletteChanged, alpha0Idx, entry0Idx, lowestAlphaIdx = self.__fixExoplayerPalette()
        if paletteChanged:
            matrixIndex = np.array(imageInfo['matrixIndex'])
            if lowestAlphaIdx is not None:
                # 3 index rotate alpha0Idx <== entry0Idx, entry0Idx <== lowestAlphaIdx, lowestAlphaIdx <== alpha0Idx
                matrixIndex = np.where(matrixIndex == alpha0Idx, entry0Idx,
                                       np.where(matrixIndex == entry0Idx, lowestAlphaIdx,
                                                np.where(matrixIndex == lowestAlphaIdx, alpha0Idx, matrixIndex)))
            else:
                # swap alpha0Idx <==> entry0Idx
                matrixIndex = np.where(matrixIndex == alpha0Idx, entry0Idx,
                                       np.where(matrixIndex == entry0Idx, alpha0Idx, matrixIndex))

            imageInfo['matrixIndex'] = matrixIndex.tolist()

        return paletteChanged

    def __buildHeader(self, segType, segSize):
        pos = self.__buildBufPosition
        self.__buildBuf[pos:pos+1] = b'\x50'    # 'P'
        self.__buildBuf[pos+1:pos+2] = b'\x47'  # 'G'
        self.__buildBuf[pos+2:pos+6] = self.__ptsByteData
        self.__buildBuf[pos+6:pos+10] = b'\x00\x00\x00\x00'
        self.__buildBuf[pos+10:pos+11] = segType
        self.__buildBuf[pos+11:pos+13] = segSize.to_bytes(2, 'big')  # segment maximun size = 65536
        self.__buildBufPosition += 13

    def __buildPCS(self, compositionState, compositionCount, imageInfo):
        self.__buildHeader(b'\x16', 19 if compositionCount else 11)

        pos = self.__buildBufPosition
        self.__buildBuf[pos:pos+2] = self.videoWidth.to_bytes(2, 'big')
        self.__buildBuf[pos+2:pos+4] = self.videoHeight.to_bytes(2, 'big')
        self.__buildBuf[pos+4:pos+5] = self.frameRate.to_bytes(1, 'big')  # keep original
        self.__buildBuf[pos+5:pos+7] = self.__serialNo.to_bytes(2, 'big')
        self.__buildBuf[pos+7:pos+8] = compositionState
        self.__buildBuf[pos+8:pos+9] = self.paletteUpdateFlag.to_bytes(1, 'big')
        self.__buildBuf[pos+9:pos+10] = b'\x00'   # only support one platte
        self.__buildBuf[pos+10:pos+11] = compositionCount.to_bytes(1, 'big')
        pos += 11
        self.__buildBufPosition += 11

        if compositionCount > 0:
            self.__buildBuf[pos:pos+2] = b'\x00\x00'  # only support one object (oid)
            self.__buildBuf[pos+2:pos+3] = b'\x00'    # only support one window (wid)

            self.__buildBuf[pos+3:pos+4] = imageInfo['croppedFlag'].to_bytes(1, 'big')   # cropping flag
            self.__buildBuf[pos+4:pos+6] = imageInfo['offsetX'].to_bytes(2, 'big')       # X offset on the screen
            self.__buildBuf[pos+6:pos+8] = imageInfo['offsetY'].to_bytes(2, 'big')       # Y offset on the screen
            self.__buildBufPosition += 8

            if imageInfo['croppedFlag'] & 0x80:  # if cropped
                self.__buildBuf[pos+8:pos+16] = imageInfo['croppedByteData']
                self.__buildBufPosition += 8

        self.__serialNo += 1

    def __buildWDS(self, imageInfo):
        self.__buildHeader(b'\x17', 10)

        pos = self.__buildBufPosition
        self.__buildBuf[pos:pos+1] = b'\x01'    # only support one window
        self.__buildBuf[pos+1:pos+2] = b'\x00'  # only support one windows so wid always 0
        self.__buildBuf[pos+2:pos+4] = imageInfo['offsetX'].to_bytes(2, 'big')
        self.__buildBuf[pos+4:pos+6] = imageInfo['offsetY'].to_bytes(2, 'big')
        self.__buildBuf[pos+6:pos+8] = imageInfo['width'].to_bytes(2, 'big')
        self.__buildBuf[pos+8:pos+10] = imageInfo['height'].to_bytes(2, 'big')
        self.__buildBufPosition += 10

    def __buildPDS(self):
        paletteSize = len(self.__paletteByteData)
        self.__buildHeader(b'\x14', paletteSize+2)

        pos = self.__buildBufPosition
        self.__buildBuf[pos:pos+1] = b'\x00'             # only support one palette, pid = 0
        self.__buildBuf[pos+1:pos+2] = self.paletteVer   # palette version
        self.__buildBuf[pos+2:pos+2+paletteSize] = self.__paletteByteData
        self.__buildBufPosition += paletteSize+2

    def __buildODS(self, seqFlag, rleDataSize, objectData, imageInfo):
        appendSize = 11 if (firstFrag := seqFlag & 0x80) else 4
        dataSize = len(objectData)

        self.__buildHeader(b'\x15', dataSize+appendSize)

        pos = self.__buildBufPosition
        self.__buildBuf[pos:pos+2] = b'\x00\x00'  # only support one image object, oid = 0
        self.__buildBuf[pos+2:pos+3] = imageInfo['objectVer']  # object version
        self.__buildBuf[pos+3:pos+4] = seqFlag.to_bytes()      # object data sequence flag

        if firstFrag:  # First fragment
            rleDataSize += 4  # include length of width & height
            self.__buildBuf[pos+4:pos+7] = rleDataSize.to_bytes(3, 'big')  # MAX SIZE = 2^24 = 16M
            self.__buildBuf[pos+7:pos+9] = imageInfo['width'].to_bytes(2, 'big')
            self.__buildBuf[pos+9:pos+11] = imageInfo['height'].to_bytes(2, 'big')
            self.__buildBuf[pos+11:pos+11+dataSize] = objectData
        else:
            self.__buildBuf[pos+4:pos+4+dataSize] = objectData

        self.__buildBufPosition += dataSize + appendSize

    def __buildEND(self):
        self.__buildHeader(b'\x80', 0)

    def __saveImage(self, img):
        imageFile = self.__xmlFolder + "%04d.png" % self.totalImageCount
        img.save(imageFile, format='PNG')

    def __setBdnXmlInfo(self, imageInfo, abnormalEnd=0):
        compState = CompositionState[self.compositionState]
        bdnIdx = self.totalImageCount - 1

        if abnormalEnd and self.bdnXmlInfo[bdnIdx]['endTime'] is None:
            self.bdnXmlInfo[bdnIdx]['endTime'] = self.bdnXmlInfo[bdnIdx]['startTime'] + 5000
            return

        if self.imageObjectsCount > 0:
            if compState != 'Epoch Start' and compState != 'Acquisition Point':
                logging.warning("[SET-BDNXML-INFO segmet state != 'Epoch Start' or 'Acquisition Point' (%s)"
                                % compState)

            if bdnIdx and self.bdnXmlInfo[bdnIdx-1]['endTime'] is None:  # update last endTime if empty
                self.bdnXmlInfo[bdnIdx-1]['endTime'] = self.pts

            self.bdnXmlInfo.append({'startTime': self.pts, 'endTime': None,
                                    'x': imageInfo['offsetX'], 'y': imageInfo['offsetY'],
                                    'width': imageInfo['width'], 'height': imageInfo['height']})
            return 1
        else:
            if compState != 'Normal':
                print("[SET-BDNXML-INFO] no image segment with state = '%s" % compState)
            self.bdnXmlInfo[bdnIdx]['endTime'] = self.pts
            return 0

    def __bdnTimeStr(self, msTime):
        if self.__fps == '1000':
            return ms2hmsf(msTime)
        else:
            return ms2hmsf(msTime, 1) + ":" + "%02d" % round((msTime % 1000 * float(self.__fps))/1000)

    def __createBdnXmlFile(self):
        numOfEvent = len(self.bdnXmlInfo)
        startTime = self.__bdnTimeStr(self.bdnXmlInfo[0]['startTime'])
        endTime = self.__bdnTimeStr(self.bdnXmlInfo[numOfEvent-1]['endTime'])
        xmlFile = self.__xmlFolder + Path(self.__file).stem + '.xml'
        f = open(xmlFile, "w")

        if self.videoWidth == 1920 and self.videoHeight == 1080:
            videoFormat = "1080p"
        else:
            videoFormat = "%dx%d" % (self.videoWidth, self.videoHeight)

        print('<?xml version="1.0" encoding="UTF-8"?>', file=f)
        print('<BDN Version="0.93" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
              'xsi:noNamespaceSchemaLocation="BD-03-006-0093b BDN File Format.xsd">', file=f)
        print('  <Description>', file=f)
        print('    <Name Title="subtitle_exp" Content="" />', file=f)
        print('    <Language Code="eng" />', file=f)  # always 'eng'
        print('    <Format VideoFormat="%s" FrameRate="%s" DropFrame="False" />' % (videoFormat, self.__fps), file=f)
        print('    <Events Type="Graphic" FirstEventInTC="%s" '
              'LastEventOutTC="%s" NumberofEvents="%d" />' % (startTime, endTime, numOfEvent), file=f)
        print('  </Description>', file=f)
        print('  <Events>', file=f)
        for idx, bdnEvent in enumerate(self.bdnXmlInfo):
            startTime = self.__bdnTimeStr(bdnEvent['startTime'])
            endTime = self.__bdnTimeStr(bdnEvent['endTime'])
            print('    <Event InTC="%s" OutTC="%s" Forced="False">' % (startTime, endTime), file=f)
            print('      <Graphic Width="%d" Height="%d" X="%d" Y="%d">%04d.png</Graphic>'
                  % (bdnEvent['width'], bdnEvent["height"], bdnEvent["x"], bdnEvent["y"], idx+1), file=f)
            print('    </Event>', file=f)
        print('  </Events>', file=f)
        print('</BDN>', file=f)

        f.close()


if __name__ == '__main__':
    def parseArgument():
        parser = argparse.ArgumentParser(
            description='Process a input ".sup" file and generate a output ".mod.sup" file.')
        parser.add_argument(dest="supFile", type=str, help='path to the sup file')
        parser.add_argument('-d', '--debug', action='store_true', help='display debug message')

        parser.add_argument('-xml', action='store_true', help='output XML/PNG files to folder "./xmlPng"')
        parser.add_argument('-fps', default='25', choices=['23.976', '24', '25', '29.97', '1000'],
                            help="frame rate for XML file, default = 25")

        return parser.parse_args()

    def main():
        args = parseArgument()
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        supFile = os.path.abspath(args.supFile)
        if not os.path.exists(supFile):
            print('Error: input file "%s" does not exist.' % supFile)
            sys.exit(1)

        sup = SupRebuild(supFile, args.xml, args.fps)
        buf = sup.generateNewSUP()

        if buf is not None:
            if not sup.fixed:
                print("\nNothing changed !")
            else:
                outFile = Path(supFile).with_suffix('.mod.sup')
                with open(outFile, "wb") as supFp:
                    supFp.write(buf)
                    supFp.close()
                print("\nCreate new sup file: {} ({:,} bytes)".format(outFile, len(buf)))

        print("Elapsed time: %.2fs" % sup.elapseTime)

        return 0 if buf is None else 1

    main()
