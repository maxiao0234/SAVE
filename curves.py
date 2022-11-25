# Copy from https://github.com/PvdPal. Thanks!

import math


def hilbertCurve(iteration, rotation, xOffset=0, yOffset=0):
    """
    The Hilbert Curve
    This function creates a Hilbert curve of the given size, the higher the given
    iteration is, the bigger the size of the Hilbert curve will be. The three
    latter variables are needed to make sure the curve has the correct rotation
    and offset. This is used to connect the multiple curves correctly.
    """
    if(iteration > 1):
        if(rotation == 'leftReverse'):
            firstExtension = hilbertCurve(iteration - 1, 'topReverse', math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
            secondExtension = hilbertCurve(iteration - 1, 'leftReverse', 0 + xOffset, 0 + yOffset)
            thirdExtension = hilbertCurve(iteration - 1, 'leftReverse', 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
            fourthExtension = hilbertCurve(iteration - 1, 'bottomReverse', math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
            return firstExtension + secondExtension + thirdExtension + fourthExtension
        elif(rotation == 'top'):
            firstExtension = hilbertCurve(iteration - 1, 'right', 0 + xOffset, 0 + yOffset)
            secondExtension = hilbertCurve(iteration - 1, 'top', 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
            thirdExtension = hilbertCurve(iteration - 1, 'top', math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
            fourthExtension = hilbertCurve(iteration - 1, 'left', math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
            return firstExtension + secondExtension + thirdExtension + fourthExtension
        elif(rotation == 'rightReverse'):
            firstExtension = hilbertCurve(iteration - 1, 'bottomReverse', 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
            secondExtension = hilbertCurve(iteration - 1, 'rightReverse', math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
            thirdExtension = hilbertCurve(iteration - 1, 'rightReverse', math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
            fourthExtension = hilbertCurve(iteration - 1, 'topReverse', 0 + xOffset, 0 + yOffset)
            return firstExtension + secondExtension + thirdExtension + fourthExtension
        elif(rotation == 'bottom'):
            firstExtension = hilbertCurve(iteration - 1, 'left', math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
            secondExtension = hilbertCurve(iteration - 1, 'bottom', math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
            thirdExtension = hilbertCurve(iteration - 1, 'bottom', 0 + xOffset, 0 + yOffset)
            fourthExtension = hilbertCurve(iteration - 1, 'right', 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
            return firstExtension + secondExtension + thirdExtension + fourthExtension
        elif(rotation == 'left'):
            firstExtension = hilbertCurve(iteration - 1, 'bottom', math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
            secondExtension = hilbertCurve(iteration - 1, 'left', 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
            thirdExtension = hilbertCurve(iteration - 1, 'left', 0 + xOffset, 0 + yOffset)
            fourthExtension = hilbertCurve(iteration - 1, 'top', math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
            return firstExtension + secondExtension + thirdExtension + fourthExtension
        elif(rotation == 'topReverse'):
            firstExtension = hilbertCurve(iteration - 1, 'leftReverse', math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
            secondExtension = hilbertCurve(iteration - 1, 'topReverse', math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
            thirdExtension = hilbertCurve(iteration - 1, 'topReverse', 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
            fourthExtension = hilbertCurve(iteration - 1, 'rightReverse', 0 + xOffset, 0 + yOffset)
            return firstExtension + secondExtension + thirdExtension + fourthExtension
        elif(rotation == 'right'):
            firstExtension = hilbertCurve(iteration - 1, 'top', 0 + xOffset, 0 + yOffset)
            secondExtension = hilbertCurve(iteration - 1, 'right', math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
            thirdExtension = hilbertCurve(iteration - 1, 'right', math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
            fourthExtension = hilbertCurve(iteration - 1, 'bottom', 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
            return firstExtension + secondExtension + thirdExtension + fourthExtension
        elif(rotation == 'bottomReverse'):
            firstExtension = hilbertCurve(iteration - 1, 'rightReverse', 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
            secondExtension = hilbertCurve(iteration - 1, 'bottomReverse', 0 + xOffset, 0 + yOffset)
            thirdExtension = hilbertCurve(iteration - 1, 'bottomReverse', math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
            fourthExtension = hilbertCurve(iteration - 1, 'leftReverse', math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
            return firstExtension + secondExtension + thirdExtension + fourthExtension
    else:
        if(rotation == 'left'):
            return [(1 + xOffset, 1 + yOffset),
                    (0 + xOffset, 1 + yOffset),
                    (0 + xOffset, 0 + yOffset),
                    (1 + xOffset, 0 + yOffset)]
        elif(rotation == 'top'):
            return [(0 + xOffset, 0 + yOffset),
                    (0 + xOffset, 1 + yOffset),
                    (1 + xOffset, 1 + yOffset),
                    (1 + xOffset, 0 + yOffset)]
        elif(rotation == 'right'):
            return [(0 + xOffset, 0 + yOffset),
                    (1 + xOffset, 0 + yOffset),
                    (1 + xOffset, 1 + yOffset),
                    (0 + xOffset, 1 + yOffset)]
        elif(rotation == 'bottom'):
            return [(1 + xOffset, 1 + yOffset),
                    (1 + xOffset, 0 + yOffset),
                    (0 + xOffset, 0 + yOffset),
                    (0 + xOffset, 1 + yOffset)]
        elif(rotation == 'leftReverse'):
            return [(1 + xOffset, 0 + yOffset),
                    (0 + xOffset, 0 + yOffset),
                    (0 + xOffset, 1 + yOffset),
                    (1 + xOffset, 1 + yOffset)]
        elif(rotation == 'topReverse'):
            return [(1 + xOffset, 0 + yOffset),
                    (1 + xOffset, 1 + yOffset),
                    (0 + xOffset, 1 + yOffset),
                    (0 + xOffset, 0 + yOffset)]
        elif(rotation == 'rightReverse'):
            return [(0 + xOffset, 1 + yOffset),
                    (1 + xOffset, 1 + yOffset),
                    (1 + xOffset, 0 + yOffset),
                    (0 + xOffset, 0 + yOffset)]
        elif(rotation == 'bottomReverse'):
            return [(0 + xOffset, 1 + yOffset),
                    (0 + xOffset, 0 + yOffset),
                    (1 + xOffset, 0 + yOffset),
                    (1 + xOffset, 1 + yOffset)]


def mooreCurve(iteration, rotation, xOffset=0, yOffset=0):
    """
    The Moore Curve
    This function creates a Moore curve of the given size, the higher the given
    iteration is, the bigger the size of the Moore curve will be. The three
    latter variables are needed to make sure the curve has the correct rotation
    and offset. This is used to connect the multiple curves correctly.
    """
    if(iteration > 1):
        if(rotation == 'right'):
            firstExtension = hilbertCurve(iteration - 1, 'top', 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
            secondExtension = hilbertCurve(iteration - 1, 'top', math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
            thirdExtension = hilbertCurve(iteration - 1, 'bottom', math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
            fourthExtension = hilbertCurve(iteration - 1, 'bottom', 0 + xOffset, 0 + yOffset)
            return firstExtension + secondExtension + thirdExtension + fourthExtension
        elif(rotation == 'top'):
            firstExtension = hilbertCurve(iteration - 1, 'leftReverse', 0 + xOffset, 0 + yOffset)
            secondExtension = hilbertCurve(iteration - 1, 'leftReverse', 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
            thirdExtension = hilbertCurve(iteration - 1, 'rightReverse', math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
            fourthExtension = hilbertCurve(iteration - 1, 'rightReverse', math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
            return firstExtension + secondExtension + thirdExtension + fourthExtension
        elif(rotation == 'left'):
            firstExtension = hilbertCurve(iteration - 1, 'bottom', math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
            secondExtension = hilbertCurve(iteration - 1, 'bottom', 0 + xOffset, 0 + yOffset)
            thirdExtension = hilbertCurve(iteration - 1, 'top', 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
            fourthExtension = hilbertCurve(iteration - 1, 'top', math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
            return firstExtension + secondExtension + thirdExtension + fourthExtension
        elif(rotation == 'bottom'):
            firstExtension = hilbertCurve(iteration - 1, 'rightReverse', math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
            secondExtension = hilbertCurve(iteration - 1, 'rightReverse', math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
            thirdExtension = hilbertCurve(iteration - 1, 'leftReverse', 0 + xOffset, 0 + yOffset)
            fourthExtension = hilbertCurve(iteration - 1, 'leftReverse', 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
            return firstExtension + secondExtension + thirdExtension + fourthExtension
        elif(rotation == 'rightReverse'):
            firstExtension = hilbertCurve(iteration - 1, 'bottomReverse', 0 + xOffset, 0 + yOffset)
            secondExtension = hilbertCurve(iteration - 1, 'bottomReverse', math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
            thirdExtension = hilbertCurve(iteration - 1, 'topReverse', math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
            fourthExtension = hilbertCurve(iteration - 1, 'topReverse', 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
            return firstExtension + secondExtension + thirdExtension + fourthExtension
        elif(rotation == 'topReverse'):
            firstExtension = hilbertCurve(iteration - 1, 'right', math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
            secondExtension = hilbertCurve(iteration - 1, 'right', math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
            thirdExtension = hilbertCurve(iteration - 1, 'left', 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
            fourthExtension = hilbertCurve(iteration - 1, 'left', 0 + xOffset, 0 + yOffset)
            return firstExtension + secondExtension + thirdExtension + fourthExtension
        elif(rotation == 'leftReverse'):
            firstExtension = hilbertCurve(iteration - 1, 'topReverse', math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
            secondExtension = hilbertCurve(iteration - 1, 'topReverse', 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
            thirdExtension = hilbertCurve(iteration - 1, 'bottomReverse', 0 + xOffset, 0 + yOffset)
            fourthExtension = hilbertCurve(iteration - 1, 'bottomReverse', math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
            return firstExtension + secondExtension + thirdExtension + fourthExtension
        elif(rotation == 'bottomReverse'):
            firstExtension = hilbertCurve(iteration - 1, 'left', 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
            secondExtension = hilbertCurve(iteration - 1, 'left', 0 + xOffset, 0 + yOffset)
            thirdExtension = hilbertCurve(iteration - 1, 'right', math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
            fourthExtension = hilbertCurve(iteration - 1, 'right', math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
            return firstExtension + secondExtension + thirdExtension + fourthExtension
    else:
        if(rotation == 'left'):
            return [(1 + xOffset, 1 + yOffset),
                    (0 + xOffset, 1 + yOffset),
                    (0 + xOffset, 0 + yOffset),
                    (1 + xOffset, 0 + yOffset)]
        elif(rotation == 'top'):
            return [(0 + xOffset, 0 + yOffset),
                    (0 + xOffset, 1 + yOffset),
                    (1 + xOffset, 1 + yOffset),
                    (1 + xOffset, 0 + yOffset)]
        elif(rotation == 'right'):
            return [(0 + xOffset, 0 + yOffset),
                    (1 + xOffset, 0 + yOffset),
                    (1 + xOffset, 1 + yOffset),
                    (0 + xOffset, 1 + yOffset)]
        elif(rotation == 'bottom'):
            return [(1 + xOffset, 1 + yOffset),
                    (1 + xOffset, 0 + yOffset),
                    (0 + xOffset, 0 + yOffset),
                    (0 + xOffset, 1 + yOffset)]
        elif(rotation == 'leftReverse'):
            return [(1 + xOffset, 0 + yOffset),
                    (0 + xOffset, 0 + yOffset),
                    (0 + xOffset, 1 + yOffset),
                    (1 + xOffset, 1 + yOffset)]
        elif(rotation == 'topReverse'):
            return [(1 + xOffset, 0 + yOffset),
                    (1 + xOffset, 1 + yOffset),
                    (0 + xOffset, 1 + yOffset),
                    (0 + xOffset, 0 + yOffset)]
        elif(rotation == 'rightReverse'):
            return [(0 + xOffset, 1 + yOffset),
                    (1 + xOffset, 1 + yOffset),
                    (1 + xOffset, 0 + yOffset),
                    (0 + xOffset, 0 + yOffset)]
        elif(rotation == 'bottomReverse'):
            return [(0 + xOffset, 1 + yOffset),
                    (0 + xOffset, 0 + yOffset),
                    (1 + xOffset, 0 + yOffset),
                    (1 + xOffset, 1 + yOffset)]


def z_orderCurve(iteration, xOffset=0, yOffset=0):
    """
    The Z-order Curve
    This function creates a z-order curve of the given size, the higher the given
    iteration is, the bigger the size of the z-order curve will be. The three
    latter variables are needed to make sure the curve has the correct rotation
    and offset. This is used to connect the multiple curves correctly.
    """
    if iteration > 1:
        firstExtension = z_orderCurve(iteration - 1, 0 + xOffset, math.pow(2, iteration - 1) + yOffset)
        secondExtension = z_orderCurve(iteration - 1, math.pow(2, iteration - 1) + xOffset, math.pow(2, iteration - 1) + yOffset)
        thirdExtension = z_orderCurve(iteration - 1, 0 + xOffset, 0 + yOffset)
        fourthExtension = z_orderCurve(iteration - 1, math.pow(2, iteration - 1) + xOffset, 0 + yOffset)
        return firstExtension + secondExtension + thirdExtension + fourthExtension
    else:
        return [(0 + xOffset, 1 + yOffset),
                (1 + xOffset, 1 + yOffset),
                (0 + xOffset, 0 + yOffset),
                (1 + xOffset, 0 + yOffset)]

