import numpy as np
import cv2
import face_alignment

# Initialize the chip resolution
chipSize = 300
chipCorners = np.float32([[0,0],
                          [chipSize,0],
                          [0,chipSize],
                          [chipSize,chipSize]])

# Initialize the face alignment tracker
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True, device="cuda")

# Start the webcam capture, exit with 'q'
cap = cv2.VideoCapture(0)
while(not (cv2.waitKey(1) & 0xFF == ord('q'))):
    ret, frame = cap.read()
    if(ret):
        # Run the face alignment tracker on the webcam image
        imagePoints = fa.get_landmarks_from_image(frame)
        if(imagePoints is not None):
            imagePoints = imagePoints[0]

            # Compute the Anchor Landmarks
            # This ensures the eyes and chin will not move within the chip
            rightEyeMean = np.mean(imagePoints[36:42], axis=0)
            leftEyeMean  = np.mean(imagePoints[42:47], axis=0)
            middleEye    = (rightEyeMean + leftEyeMean) * 0.5
            chin         = imagePoints[8]
            #cv2.circle(frame, tuple(rightEyeMean[:2].astype(int)), 30, (255,255,0))
            #cv2.circle(frame, tuple(leftEyeMean [:2].astype(int)), 30, (255,0,255))

            # Compute the chip center and up/side vectors
            mean = ((middleEye * 3) + chin) * 0.25
            centered = imagePoints - mean 
            rightVector = (leftEyeMean - rightEyeMean)
            upVector    = (chin        - middleEye)

            # Divide by the length ratio to ensure a square aspect ratio
            rightVector /= np.linalg.norm(rightVector) / np.linalg.norm(upVector)

            # Compute the corners of the facial chip
            imageCorners = np.float32([(mean + ((-rightVector - upVector)))[:2],
                                       (mean + (( rightVector - upVector)))[:2],
                                       (mean + ((-rightVector + upVector)))[:2],
                                       (mean + (( rightVector + upVector)))[:2]])

            # Compute the Perspective Homography and Extract the chip from the image
            chipMatrix = cv2.getPerspectiveTransform(imageCorners, chipCorners)
            chip = cv2.warpPerspective(frame, chipMatrix, (chipSize, chipSize))

        cv2.imshow('Webcam View', frame)
        cv2.imshow('Chip View', chip)

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
# https://gist.github.com/zalo/fa4396ae7a72b7683888fd9cd1c6d920
# Here's a quickie example for extracting Face Chips using this and OpenCV.
# https://github.com/1adrianb/face-alignment/issues/135
