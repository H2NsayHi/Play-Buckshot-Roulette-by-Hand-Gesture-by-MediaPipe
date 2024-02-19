import streamlit as st
import cv2 as cv
import numpy as np
import mediapipe as mp
import itertools
import copy
from collections import deque
from collections import Counter
from model import KeyPointClassifier
from model import PointHistoryClassifier
import csv
import time
import os
import random


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)


    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


def camera():
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]


    start_time = time.time()
    temp_gesture = None
    lock_gesture = None
    gesture_lock_status = False

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    use_brect = True



    # Camera capture
    cap = cv.VideoCapture(0)

    # Create a placeholder for the video
    video_placeholder = st.sidebar.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv.flip(frame, 1)  # Mirror display
        debug_frame = copy.deepcopy(frame)

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                brect = calc_bounding_rect(debug_frame, hand_landmarks)
                landmark_list = calc_landmark_list(debug_frame, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_frame, point_history)
                


                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == "Not to use":
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                debug_frame = draw_bounding_rect(use_brect, debug_frame, brect)
                debug_frame = draw_landmarks(debug_frame, landmark_list)
                debug_frame = draw_info_text(
                    debug_frame,
                    brect,
                    handedness,
                    str(keypoint_classifier_labels[hand_sign_id]),
                    str(most_common_fg_id[0][0]),
                )



                if gesture_lock_status:
                    text_size = cv.getTextSize(
                        f"Do you want to use {keypoint_classifier_labels[lock_gesture]}?",
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        2,
                    )[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = 40  # Adjust the vertical position as needed

                    cv.putText(
                        debug_frame,
                        f"Do you want to use {keypoint_classifier_labels[lock_gesture]}?",
                        (text_x, text_y),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv.LINE_AA,
                    ) 
                    if hand_sign_id == 5:
                        video_placeholder.empty()
                        return lock_gesture


                if hand_sign_id == temp_gesture:
                    if time.time() - start_time > 1.5:
                        gesture_lock_status = True
                        lock_gesture = temp_gesture
                else:
                    start_time = time.time()
                
                temp_gesture = hand_sign_id





        debug_frame = draw_point_history(debug_frame, point_history)
        video_placeholder.image(debug_frame, channels="BGR")


DEFAULT_HEALTH = 4
dir_path = os.path.dirname(os.path.realpath(__file__))

rounds = 0


camera_holder = st.empty()

user_items_status_place = st.sidebar.empty()
user_items_place = st.sidebar.empty()
deler_items_status_place = st.sidebar.empty()
dealer_items_place = st.sidebar.empty()


user_health_holder, dealer_health_holder = [_.empty() for _ in st.columns(2)]


talk_general_place = st.empty()

talk_holder = st.empty()
screen_holder = st.empty()
result_holder = st.empty()

bullet_status = st.empty()
bullet_place = st.empty()


def displayUserList(arr):
    # for col, i in zip(user_items_place.columns(len(arr)), arr):
    #     col.image(i)

    for i, col in enumerate(user_items_place.columns(4)):
        if i < len(arr):
            col.image(arr[i])
        else:
            col.empty()

def displayDealerList(arr):

    # for col, i in zip(dealer_items_place.columns(len(arr)), arr):
    #     col.image(i)

    for i, col in enumerate(dealer_items_place.columns(4)):
        if i < len(arr):
            col.image(arr[i])
        else:
            col.empty()

    
def displayHelp():
    dealer_items_place.write("""
    INSTRUCTIONS: \n
        - OBJECTIVE: SURVIVE.
        - A shotgun is loaded with a disclosed number of bullets, some of which will be blanks.
        - Participants are given a set amount of lives (default = 4 to survive.
        - You and 'The Dealer' will take turns shooting.
        - Aim at The Dealer or at yourself - shooting yourself with a blank skips the Dealers turn.
        - Participants are given items to help out. Use them wisely.
        - if you have chosen wrongly, type 'q'/'quit'/'back' to go back.
    
    ITEMS: \n
        • 🚬 = Gives the user an extra life.
        • 🍺 = Racks the shotgun and the bullet inside will be discarded.
        • 🔪 = Shotgun will deal double damage for one turn.
        • 🔍 = User will see what bullet is in the chamber.
        • ⛓ = Handcuffs the other person so they miss their next turn.
    Good Luck.
    """)


class Shotgun():
    def __init__(self):
        self.damage = 1
        self.rounds = []
    def doubleDamage(self):
        self.damage = 2
    def resetDamage(self):
        self.damage = 1
    
    def addRounds(self,live=0,blank=0):
        self.rounds.extend([True]*live)
        self.rounds.extend([False]*blank)
        random.shuffle(self.rounds)
            
    def pickRound(self):
        l = len(self.rounds)
        if not l: return
        return self.rounds.pop()
    
class Player():
    def __init__(self,health=DEFAULT_HEALTH,items=[]):
        self.health = health
        self.items = items
        self.turnsWaiting = 0
        
    def takeDamage(self,dmg=1):
        self.health = self.health - dmg
        return (not self.health)
    
    def addHealth(self,health=1):
        self.health = self.health +health
    
    def addRandomItems(self,n=None):
        if not n:
            n = random.randint(1,4)
        items = ["mat/beer.jpg",
                 "mat/cigarette.jpg",
                 "mat/glass.jpg",
                 "mat/handcuff.jpg",
                 "mat/saw.jpg"]
        self.items = [items[random.randint(0,4)] for _ in range(n)]
            
    def useItem(self,item, gun, effector):
        if not item in self.items:
            return False
        talk_general_place.write("Using gun")
        temp = self.items
        temp.remove(item)
        self.items = temp
        talk_holder.write(f"[{name}] USED: {item}")
        time.sleep(0)
        match item:
            case 'mat/saw.jpg':
                talk_holder.write("Let's cuff you off")
                gun.doubleDamage()
                playersaw_vid = cv.VideoCapture("mat/playersaw.mp4")
                playersaw_place = st.empty()

                while True:
                    ret, frame = playersaw_vid.read()
                    if not ret:
                        break
                    screen_holder.image(frame, channels="BGR")
                result_holder.write("Shotgun now does 2 damage.")

            case 'mat/glass.jpg':
    
                talk_holder.write("Shhhh~~~.. The next round is..")
                time.sleep(1)
                if gun.rounds[-1]:
                    playerglass_true_vid = cv.VideoCapture("mat/playerglass_true.mp4")
                    playerglass_true_place = st.empty()

                    while True:
                        ret, frame = playerglass_true_vid.read()
                        if not ret:
                            break
                        screen_holder.image(frame, channels="BGR")
                else:
                    playerglass_false_vid = cv.VideoCapture("mat/playerglass_false.mp4")
                    playerglass_false_place = st.empty()

                    while True:
                        ret, frame = playerglass_false_vid.read()
                        if not ret:
                            break
                        screen_holder.image(frame, channels="BGR")

                result_holder.write(["blank.", "LIVE."][gun.rounds[-1]])
                time.sleep(0)

            case 'mat/handcuff.jpg':
                if not effector: return False

                talk_holder.write("Look cuff you off")
                effector.missTurns(1)
                playerhandcuff_ai_vid = cv.VideoCapture("mat/playerhandcuff_ai.mp4")
                playerhandcuff_ai_place = st.empty()

                while True:
                    ret, frame = playerhandcuff_ai_vid.read()
                    if not ret:
                        break
                    screen_holder.image(frame, channels="BGR")
                result_holder.write("Dealer will now miss a turn.")
            
            case 'mat/beer.jpg':
                talk_holder.write("Shotgun has been racked. Round was.....")
                r = gun.pickRound()
                time.sleep(1)
                if r:
                    playerbeer_true_vid = cv.VideoCapture("mat/playerbeer_true.mp4")
                    playerbeer_true_place = st.empty()

                    while True:
                        ret, frame = playerbeer_true_vid.read()
                        if not ret:
                            break
                        screen_holder.image(frame, channels="BGR")
                else:
                    playerbeer_false_vid = cv.VideoCapture("mat/playerbeer_false.mp4")
                    playerbeer_false_place = st.empty()

                    while True:
                        ret, frame = playerbeer_false_vid.read()
                        if not ret:
                            break
                        screen_holder.image(frame, channels="BGR")
                result_holder.write(["blank.","LIVE."][r])

            case 'mat/cigarette.jpg':
                self.addHealth()
                user_health_holder.write(f"Your HP: {self.health}")
                talk_holder.write("So satisfying...")
                playercigarette_vid = cv.VideoCapture("mat/playercigarette.mp4")
                playercigarette_place = st.empty()

                while True:
                    ret, frame = playercigarette_vid.read()
                    if not ret:
                        break
                    screen_holder.image(frame, channels="BGR")
                result_holder.write(self.health)

            case _:
                talk_holder.write("uhm....")
                time.sleep(0)
                screen_holder.write("Game does not recognise the item.")
                result_holder.write("Try again")
                return False
        time.sleep(0)
        return True
    
    def missTurns(self,n=1):
        self.turnsWaiting = n
    
class AI(Player):
    def useItem(self,item,effector=None,gun=None):

        if item not in self.items or (
                item == 'mat/handcuff.jpg' and effector.turnsWaiting) or (
                item == 'mat/saw.jpg' and sg.damage != 1
                ): return False
        temp = self.items
        temp.remove(item)
        self.items = temp

        talk_holder.write(f"[DEALER] used {item}")
        time.sleep(0)

        match item:
            case 'mat/handcuff.jpg':
                talk_holder.write("Locked....")
                effector.missTurns()
                aihandcuff_player_vid = cv.VideoCapture("mat/aihandcuff_player.mp4")
                aihandcuff_player_place = st.empty()

                while True:
                    ret, frame = aihandcuff_player_vid.read()
                    if not ret:
                        break
                    screen_holder.image(frame, channels="BGR")
                result_holder.write("[DEALER] cuffed you.")
            case 'mat/saw.jpg':
                gun.doubleDamage()
                time.sleep(0)
                talk_holder.write("Double it up ~~")
                aisaw_vid = cv.VideoCapture("mat/aisaw.mp4")
                aisaw_place = st.empty()

                while True:
                    ret, frame = aisaw_vid.read()
                    if not ret:
                        break
                    screen_holder.image(frame, channels="BGR")
                result_holder.write("Shotgun now does 2 damage.")
            case 'mat/cigarette.jpg':
                self.addHealth()
                dealer_health_holder.write(f"Dealer HP: {self.health}")
                time.sleep(0)
                talk_holder.write("Ahhhh ~~~")
                aicigarette_vid = cv.VideoCapture("mat/aicigarette.mp4")
                aicigarette_place = st.empty()

                while True:
                    ret, frame = aicigarette_vid.read()
                    if not ret:
                        break
                    screen_holder.image(frame, channels="BGR")
                result_holder.write(f"[DEALER] now has {self.health} lives.")
            case 'mat/beer.jpg':
                r = gun.pickRound()
                time.sleep(0)
                talk_holder.write("Gun has been racked. THE ROUND IS..")
                time.sleep(1)
                if r:
                    aibeer_true_vid = cv.VideoCapture("mat/aibeer_true.mp4")
                    aibeer_true_place = st.empty()

                    while True:
                        ret, frame = aibeer_true_vid.read()
                        if not ret:
                            break
                        screen_holder.image(frame, channels="BGR")
                else:
                    aibeer_false_vid = cv.VideoCapture("mat/aibeer_false.mp4")
                    aibeer_false_place = st.empty()

                    while True:
                        ret, frame = aibeer_false_vid.read()
                        if not ret:
                            break
                        screen_holder.image(frame, channels="BGR")
                result_holder.write(["blank.","LIVE."][r])
            case 'mat/glass.jpg':
                r = gun.rounds[-1]
                talk_holder.write("[DEALER] has inspected the gun 🔍...")
                aiglass_vid = cv.VideoCapture("mat/aiglass.mp4")
                aiglass_place = st.empty()

                while True:
                    ret, frame = aiglass_vid.read()
                    if not ret:
                        break
                    screen_holder.image(frame, channels="BGR")
                time.sleep(1)
                result_holder.write(f"############################## {r}")
                if r:
                    self.useItem('mat/saw.jpg',gun=gun)
                    self.shoot(gun,effector)
                    return True
                self.shoot(gun)
                return True
                
        time.sleep(0)
        return True

    def shoot(self,gun,effector=None):
        r = gun.pickRound()
        talk_general_place.write("DEALER ARE SHOOTING")
        talk_holder.write("Shoot")
        if effector:
            time.sleep(0)
            if r:
                if effector.health - gun.damage < 1:
                    time.sleep(0)
                    
                effector.takeDamage(gun.damage)

                if effector.health > 0:
                    
                    time.sleep(0)
                    aishot_player_true_vid = cv.VideoCapture("mat/aishot_player_true.mp4")
                    aishot_player_true_place = st.empty()

                    while True:
                        ret, frame = aishot_player_true_vid.read()
                        if not ret:
                            break
                
                        screen_holder.image(frame, channels="BGR")
                    st.write()
                
                    time.sleep(0)
                    user_health_holder.write(f"Your HP: {effector.health}")
                    result_holder.write(f"BOOM! Lives left: {effector.health}")
                    time.sleep(0)
                    gun.resetDamage()
                return
        else:
            if r:
                self.takeDamage(1)
                
                time.sleep(0)
                aishot_ai_true_vid = cv.VideoCapture("mat/aishot_ai_true.mp4")
                aishot_ai_true_place = st.empty()

                while True:
                    ret, frame = aishot_ai_true_vid.read()
                    if not ret:
                        break
                    screen_holder.image(frame, channels="BGR")
                

                dealer_health_holder.write(f"Dealer HP: {self.health}")
                result_holder.write(f"BOOM! Dealer was shot. Dealer has {self.health} lives left.")
          
                time.sleep(0)
                gun.resetDamage()
                return

        aishot_ai_false_vid = cv.VideoCapture("mat/aishot_ai_false.mp4")
        aishot_ai_false_place = st.empty()

        while True:
            ret, frame = aishot_ai_false_vid.read()
            if not ret:
                break
            screen_holder.image(frame, channels="BGR")
        result_holder.write("*click* round was blank.")
        time.sleep(0)
        gun.resetDamage()


phd = st.empty()
user_items_place.image("mat/maxresdefault.jpg")
sg = Shotgun()
p1 = Player(DEFAULT_HEALTH)
dealer = AI(DEFAULT_HEALTH)

time.sleep(0)

displayHelp()

bullet_place.image("mat/c1Ca3c.png")
name = bullet_status.text_input("[DEALER]: PLEASE SIGN THE WAIVER.")
if name:
    user_health_holder.write("Your HP: 4")
    dealer_health_holder.write("Dealer HP: 4")

    while p1.health > 0 and dealer.health > 0:
        # load the shotgun
        live = random.randint(1,3)
        blank =  random.randint(1,3)
        sg.addRounds(live, blank)


        bullet_place.empty()
        for i, col in enumerate(bullet_place.columns(6)):
            if i == 0:
                col.image("mat/buckshot-roulette-feature-image.jpg")
            elif i < live+1:
                col.image("mat/live.png")
            elif i < live + blank +1:
                col.image("mat/blank.png")
            else:
                col.empty()
        
        time.sleep(0)
        bullet_status.write(f'<div style="text-align: center; font-size: 24px;"><strong>{live} LIVE, {blank} BLANK</strong></div>', unsafe_allow_html=True)
        time.sleep(0)


        #give the players items
        p1.addRandomItems()
        dealer.addRandomItems()
        user_items_status_place.write('<div style="text-align: center; font-size: 24px;"><strong>Your inventory</strong></div>', unsafe_allow_html=True)
        displayUserList(p1.items)
        time.sleep(0)
        deler_items_status_place.write('<div style="text-align: center; font-size: 24px;"><strong>Dealers inventory</strong></div>', unsafe_allow_html=True)
        displayDealerList(dealer.items)
        time.sleep(0)
        
        #start turns
        turn = random.choice([True,False])
        
        
        while sg.rounds and p1.health and dealer.health:
            
            if (turn and (not p1.turnsWaiting)) or dealer.turnsWaiting:
                # =========> PLAYERS TURN TO CHOOSE
                if dealer.turnsWaiting:
                    talk_general_place.write("*Dealer skips their turn*")
                    turn = not turn
                    time.sleep(1)
                    dealer.turnsWaiting = dealer.turnsWaiting - 1
                
                while sg.rounds:
                    opt = ""

                    if p1.items:

                        talk_general_place.write("IT IS YOUR TURN.")

                        talk_holder.write("USE ITEM or SHOOT")
                        inp = ""
                        while not inp:
                            inp = camera()
                            break

                        opt = "in-use"
 
                    if inp != 0 and p1.items: # ======== > PLAYER USE ITEM
            
                        if inp == 1:
                            item = "mat/cigarette.jpg"
                        elif inp == 2:
                            item = "mat/handcuff.jpg"
                        elif inp == 3:
                            item = "mat/beer.jpg"
                        elif inp == 4:
                            item = "mat/saw.jpg"
                        elif inp == 7:
                            item = "mat/glass.jpg"
                        
                        try:
                            p1.useItem(p1.items[p1.items.index(item)],sg,dealer)
                        except:
                            continue
                            
                
                        time.sleep(1)

                    else: # ============= > PLAYER SHOOT
                        talk_general_place.write("Shooting yourself will skip Dealer's turn if round is blank.\n")
                        screen_holder.write("Shoot:")
                        time.sleep(0)
                        result_holder.write("DEALER or YOU")
                        time.sleep(0)
    
                        inp = camera()

                        
                        time.sleep(0)
                        
                        if inp == 5: # shoot DEALER
                            r = sg.pickRound()
                            talk_holder.write("Take my shot")
                            time.sleep(0)
                            if r:
                                dealer.takeDamage(sg.damage)
        
                                time.sleep(0)
                                playershot_ai_true_vid = cv.VideoCapture("mat/playershot_ai_true.mp4")
                                playershot_ai_true_place = st.empty()

                                while True:
                                    ret, frame = playershot_ai_true_vid.read()
                                    if not ret:
                                        break
                                    screen_holder.image(frame, channels="BGR")

                                dealer_health_holder.write(f"Dealer HP: {dealer.health}")
                                result_holder.write(f"[DEALER] was shot. Dealers health: {dealer.health}")
                    
                                time.sleep(0)
                            else:
                                
                                time.sleep(0)
                                playershot_ai_false_vid = cv.VideoCapture("mat/playershot_ai_false.mp4")
                                playershot_ai_false_place = st.empty()

                                while True:
                                    ret, frame = playershot_ai_false_vid.read()
                                    if not ret:
                                        break
                                    screen_holder.image(frame, channels="BGR")
                                result_holder.write("*click* round was blank.")
                                time.sleep(0)
                      
                            break
                        elif inp == 6: # shoot YOURSELF
                            r = sg.pickRound()
                            talk_holder.write("Hope so hope so")
                            time.sleep(0)
                            if p1.health - sg.damage < 1:
                                time.sleep(0)
                            if r:
                                p1.takeDamage(sg.damage)
                                if p1.health > 0 :
                                    time.sleep(0)
                                    playershot_player_true_vid = cv.VideoCapture("mat/playershot_player_true.mp4")
                                    playershot_player_true_place = st.empty()

                                    while True:
                                        ret, frame = playershot_player_true_vid.read()
                                        if not ret:
                                            break
                                        screen_holder.image(frame, channels="BGR")


                                    user_health_holder.write(f"Your HP {p1.health}")
                                    result_holder.write(f"You got shot. YOUR HEALTH: {p1.health}")
                                    time.sleep(0)
                                break
                            else:
                                time.sleep(0)
                                playershot_player_false_vid = cv.VideoCapture("mat/playershot_player_false.mp4")
                                playershot_player_false_place = st.empty()

                                while True:
                                    ret, frame = playershot_player_false_vid.read()
                                    if not ret:
                                        break
                                    playershot_player_false_place.image(frame, channels="BGR")
                                st.write("*click* round was blank.")
                                time.sleep(0)
                                
                        
            else: #DEALERS turn
                if p1.turnsWaiting:
                    talk_general_place.write("You skip this turn.")
                    turn = not turn
                    time.sleep(1)
                    p1.turnsWaiting = p1.turnsWaiting - 1

                talk_general_place.write("DEALERS TURN.")
                time.sleep(0)

                while sg.rounds:
                    r = sg.rounds[-1]

                    # BEGIN AI DECIDING
                    talk_holder.write("[DEALER] is chosing...")
                    time.sleep(random.randint(15,45)/10)
                    
                    if False not in sg.rounds:
                        dealer.useItem('mat/saw.jpg',gun=sg)
                        dealer.shoot(sg,p1)
                        break

                    if True not in sg.rounds:
                        dealer.shoot(sg)
                        continue
                    
                    if dealer.health < DEFAULT_HEALTH:
                        if dealer.useItem('mat/cigarette.jpg'): continue

                    roundsLeft = len(sg.rounds)
                    if roundsLeft < 5:
                        if (sg.rounds.count(True)/roundsLeft) >= 0.5:
                            if dealer.useItem('mat/handcuff.jpg',p1): continue
                            if not p1.turnsWaiting and dealer.useItem('mat/glass.jpg',gun=sg): continue
                            if random.choice(sg.rounds) and dealer.useItem('mat/saw.jpg',gun=sg): continue
                            
                            dealer.shoot(sg,p1)
                            break
                        
                    
                    # ⛓, mag.g, knf, cg, 🍺
                    if random.choice([True,False]):
                        dealer.useItem(random.choice(dealer.items),p1,sg)
                        continue

                    temp = [None,p1][random.choice(sg.rounds)]
                    dealer.shoot(sg,temp)
                    sg.resetDamage()
                    if temp != None:
                        break
                    if r: break

            turn = not turn
            sg.resetDamage()
            

        sg.resetDamage()
        p1.turnsWaiting = 0
        rounds +=1
        dealer.turnsWaiting = 0
        if p1.health > 0 and dealer.health > 0:
            talk_general_place.write("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
            talk_holder.write("..NEXT ROUND..")
            time.sleep(3)
            screen_holder.write(f"DEALER'S HEALTH: {dealer.health}")
            time.sleep(3)
            result_holder.write(f"YOUR HEALTH: {p1.health}")
            time.sleep(3)



    talk_general_place.write("GAME OVER")

    if not dealer.health:
        talk_holder.write("You win..")
        screen_holder.image("mat/player_win.png")
    else:
        talk_holder.write("YOU DIED.")
        screen_holder.image("mat/ai_win.png")
