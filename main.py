import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

font = cv2.FONT_HERSHEY_SIMPLEX

"""
# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
"""
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:   
        
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())



        #Declaring variables
        wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
        wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x

        thumb_cmc_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y
        thumb_cmc_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x

        index_tip_y  = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        index_mid_y  = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
        index_base_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        index_base_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
        index_mcp_x  = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
        index_mcp_y  = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y

        middle_tip_y  = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        middle_mid_y  = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
        middle_base_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        middle_base_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x
        middle_mcp_x  = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
        middle_mcp_y  = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y

        ring_tip_y  = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
        ring_tip_x  = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x
        ring_mid_y  = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y
        ring_mid_x  = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x
        ring_base_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
        ring_base_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x
        ring_mcp_x  = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x
        ring_mcp_y  = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y

        pinky_tip_y  = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
        pinky_tip_x  = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
        pinky_mid_y  = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y
        pinky_mid_x  = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x
        pinky_base_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
        pinky_base_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x
        pinky_mcp_x  = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x
        pinky_mcp_y  = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y

        thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
        thumb_ip_y  = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
        thumb_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y

        image = cv2.flip(image, 1)

        if(
            ((index_mcp_y  < wrist_y) and (index_mcp_y  < thumb_cmc_y)) and
            ((middle_mcp_y < wrist_y) and (middle_mcp_y < thumb_cmc_y)) and
            ((ring_mcp_y   < wrist_y) and (ring_mcp_y   < thumb_cmc_y)) and
            ((pinky_mcp_y  < wrist_y) and (pinky_mcp_y  < thumb_cmc_y))    
        ): # Hand up
            if(
              (middle_tip_y > middle_base_y) and
              (ring_tip_y   > ring_base_y  ) and 
              (pinky_tip_y  > pinky_base_y ) and
              (index_base_y < index_mid_y  ) and
              (index_mid_y  < index_tip_y  ) and 
              (thumb_tip_y < thumb_mcp_y)
            ):
              cv2.putText(image, "Crip Sign", (50, 50), font, 1, (255, 0, 0), 3)
              print("Crip Sign")
#            elif(
#                (ring_tip_y   > ring_base_y  ) and 
#                (index_tip_y  < index_base_y ) and
#                (middle_tip_y < middle_base_y) and
#                (pinky_tip_y  < pinky_base_y )
#            ): # At the request of Japi
#              cv2.putText(image, "Japi I gotchu", (50, 50), font, 1, (255, 0, 255), 3)
#              print("Japi I gotchu")
            elif(
                (index_tip_y  < index_base_y ) and
                (middle_tip_y < middle_base_y) and
                (ring_tip_y   > ring_base_y  ) and 
                (pinky_tip_y  > pinky_base_y )
            ): # Finger Gun
              cv2.putText(image, "PUT EM UP", (50, 50), font, 1, (255, 0, 255), 3)
              print("PUT EM UP")
            
            elif(
                (index_tip_y  > index_base_y ) and
                (middle_tip_y < middle_base_y) and
                (ring_tip_y   > ring_base_y  ) and 
                (pinky_tip_y  > pinky_base_y )
            ): # Middle Finger
              cv2.putText(image, "Flippin the bird", (50, 50), font, 1, (255, 0, 255), 3)
              print("Flippin the bird")

            else:
              cv2.putText(image, "Hand Up", (50, 50), font, 1, (255, 0, 255), 3)
              print("Hand Up")
        elif(
            ((index_mcp_y  > wrist_y) and (index_mcp_y  > thumb_cmc_y)) and
            ((middle_mcp_y > wrist_y) and (middle_mcp_y > thumb_cmc_y)) and
            ((ring_mcp_y   > wrist_y) and (ring_mcp_y   > thumb_cmc_y)) and
            ((pinky_mcp_y  > wrist_y) and (pinky_mcp_y  > thumb_cmc_y)) 
        ): # Hand Down
          cv2.putText(image, "Hand Down", (50, 50), font, 1, (255, 0, 255), 3)
          print("Hand Down")
        else: # Hand Mid
          cv2.putText(image, "Hand Mid", (50, 50), font, 1, (255, 0, 255), 3)
          print("Hand Mid")
    else: # If no hand is detected
      image = cv2.flip(image, 1)
      cv2.putText(image, "Nada", (50, 50), font, 1, (255, 0, 255), 3)
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()