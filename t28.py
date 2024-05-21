import torch
import numpy as np
import cv2
from picamera2 import Picamera2, Preview
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
import RPi.GPIO as GPIO
import time
import curses
from Motor import Motor

# Set up GPIO pins for the ultrasonic sensor
TRIG_PIN = 27
ECHO_PIN = 22

# Car control class
class CarControl:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        GPIO.setmode(GPIO.BCM)
        self.motor = Motor()

    def forward(self):
        self.motor.setMotorModel(1500, 1500, -1500, 1500)  # Full power forward
        self.stdscr.addstr(0, 0, "Moving forward            ")

    def left(self):
        self.motor.setMotorModel(2000, 2000, 2000, -2000)
        self.stdscr.addstr(0, 0, "Turning left with increased power")

    def right(self):
        self.motor.setMotorModel(-2000, -2000, -2000, 2000)
        self.stdscr.addstr(0, 0, "Turning right with increased power")

    def stop(self):
        self.motor.setMotorModel(0, 0, 0, 0)
        self.stdscr.addstr(0, 0, "Stopping                    ")

    def reverse(self):
        self.motor.setMotorModel(1500, 1500, -1500, 1500)  # Full power backward
        self.stdscr.addstr(0, 0, "Moving backward             ")

# Setup GPIO for ultrasonic sensor
def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.output(TRIG_PIN, False)
    time.sleep(2)

# Get distance from ultrasonic sensor
def get_distance():
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    return distance

# Setup YOLO model
def setup_yolo(weights='yolov5n.pt', device='cpu'):
    model = DetectMultiBackend(weights, device=device)
    model.model.eval()  # Set model to evaluation mode
    return model

# Process frame for object detection
def process_frame(image, model, img_size=640, conf_thres=0.25, iou_thres=0.45):
    # Convert and normalize image, and ensure it is RGB (3 channels)
    img = torch.from_numpy(image[..., :3]).float().div(255.0)  # Drop alpha channel if present
    img = img.permute(2, 0, 1).unsqueeze(0)  # Rearrange dimensions to NCHW
    # Inference
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    return pred

# Define the central region (example: middle 50% of the frame width)
def is_in_central_region(xyxy, frame_width, frame_height, region_fraction=0.5):
    center_x = frame_width // 2
    region_width = frame_width * region_fraction
    left_bound = center_x - region_width // 2
    right_bound = center_x + region_width // 2
    x_center = (xyxy[0] + xyxy[2]) / 2
    return left_bound <= x_center <= right_bound

# Define if an object is in the left, center, or right region
def get_region(xyxy, frame_width, region_fraction=0.5):
    center_x = frame_width // 2
    region_width = frame_width * region_fraction
    left_bound = center_x - region_width // 2
    right_bound = center_x + region_width // 2
    x_center = (xyxy[0] + xyxy[2]) / 2

    if x_center < left_bound:
        return 'left'
    elif x_center > right_bound:
        return 'right'
    else:
        return 'center'

# Draw vertical dotted lines for the central region
def draw_central_region(frame, frame_width, frame_height, region_fraction=0.5):
    center_x = frame_width // 2
    region_width = int(frame_width * region_fraction)
    left_bound = center_x - region_width // 2
    right_bound = center_x + region_width // 2

    color = (0, 255, 0)  # Green
    thickness = 1
    line_type = cv2.LINE_AA

    # Draw dotted vertical lines
    for y in range(0, frame_height, 10):
        cv2.line(frame, (left_bound, y), (left_bound, y + 5), color, thickness, line_type)
        cv2.line(frame, (right_bound, y), (right_bound, y + 5), color, thickness, line_type)

# Main function
def main(stdscr):
    car = CarControl(stdscr)
    setup_gpio()
    camera = Picamera2()
    config = camera.create_preview_configuration()
    config['main']['size'] = (640, 480)
    config['main']['format'] = 'XRGB8888'  # Using XRGB format for compatibility with OpenCV
    camera.configure(config)
    camera.start()

    model = setup_yolo()

    obstacle_detected = False
    obstacle_width = None
    calculation = None

    try:
        while True:
            distance = get_distance()
            stdscr.addstr(1, 0, f"Distance: {distance} cm")
            stdscr.refresh()

            if distance < 100 and not obstacle_detected:  # If obstacle is within 1 meter
                car.stop()  # Stop the car

                # Wait up to 5 seconds for YOLO detection
                detection_made = False
                start_time = time.time()
                while time.time() - start_time < 5:
                    buffer = camera.capture_array()
                    preds = process_frame(buffer, model)
                    frame = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
                    frame_height, frame_width = frame.shape[:2]

                    draw_central_region(frame, frame_width, frame_height)

                    # Find the obstacle with the least distance from the bottom
                    min_distance_to_bottom = float('inf')
                    best_det = None
                    left_count = 0
                    right_count = 0
                    center_count = 0

                    for pred in preds:
                        for det in pred:
                            xyxy = det[:4].numpy()
                            conf = det[4]
                            cls = det[5]
                            region = get_region(xyxy, frame_width)
                            if region == 'left':
                                left_count += 1
                            elif region == 'right':
                                right_count += 1
                            else:
                                center_count += 1
                                distance_to_bottom = frame_height - int(xyxy[3])
                                if distance_to_bottom < min_distance_to_bottom:
                                    min_distance_to_bottom = distance_to_bottom
                                    best_det = det

                    if best_det is not None:
                        xyxy = best_det[:4].numpy()
                        conf = best_det[4]
                        cls = best_det[5]
                        label = f"{model.names[int(cls)]} {conf:.2f}"
                        width = int(xyxy[2] - xyxy[0])
                        height = int(xyxy[3] - xyxy[1])
                        obstacle_width = width
                        stdscr.addstr(2, 0, f"Detected {label}, Width: {width} pixels, Height: {height} pixels, Distance to bottom: {min_distance_to_bottom} pixels")
                        stdscr.addstr(3, 0, f"Obstacles - Left: {left_count}, Center: {center_count}, Right: {right_count}")
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(frame, f"Width: {width}px", (int(xyxy[0]), int(xyxy[1] - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(frame, f"Height: {height}px", (int(xyxy[0]), int(xyxy[1] - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(frame, f"Distance to bottom: {min_distance_to_bottom}px", (int(xyxy[0]), int(xyxy[1] - 55)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # Calculate and display width * distance / 10000
                        calculation = (width * distance) / 10000
                        stdscr.addstr(4, 0, f"Width * Distance / 10000: {calculation:.2f}")

                        obstacle_detected = True
                        detection_made = True
                        break

                    cv2.imshow('Camera Feed with Detections', frame)
                    cv2.waitKey(1)

                    if detection_made:
                        break

                if not detection_made:
                    stdscr.addstr(2, 0, "No YOLO detection made within 5 seconds, continuing forward")
                    car.forward()
                    continue

            elif distance <= 30:  # Stop the car if the obstacle is within 30 cm
                car.stop()
                stdscr.addstr(5, 0, f"Obstacle within 30 cm, stopping the car. Obstacle width: {obstacle_width} pixels")

                if calculation is None:  # No YOLO detection, always turn right
                    while distance <= 30:
                        stdscr.addstr(6, 0, "Turning right for 0.5 seconds")
                        car.right()
                        time.sleep(0.5)
                        distance = get_distance()
                        stdscr.addstr(1, 0, f"Distance: {distance} cm")
                        stdscr.refresh()

                    # Continue moving forward
                    stdscr.addstr(7, 0, "Continuing forward")
                    car.forward()

                else:  # YOLO detection made, use calculated turn and move
                    # Turn to the side with fewer obstacles for 0.5 seconds
                    if left_count <= right_count:
                        stdscr.addstr(6, 0, "Turning left for 0.5 seconds")
                        car.left()
                        time.sleep(0.5)
                    else:
                        stdscr.addstr(6, 0, "Turning right for 0.5 seconds")
                        car.right()
                        time.sleep(0.5)

                    # Move forward for 1.5 * calculated time
                    move_forward_time = 1.5 * calculation
                    stdscr.addstr(7, 0, f"Moving forward for {move_forward_time:.2f} seconds")
                    car.forward()
                    time.sleep(move_forward_time)

                    # Turn in the opposite direction for 0.5 seconds
                    if left_count <= right_count:
                        stdscr.addstr(8, 0, "Turning right for 0.5 seconds")
                        car.right()
                    else:
                        stdscr.addstr(8, 0, "Turning left for 0.5 seconds")
                        car.left()
                    time.sleep(0.5)

                    # Stop the car
                    stdscr.addstr(9, 0, "Stopping the car")
                    car.stop()

                    # Reset detection status
                    obstacle_detected = False
                    calculation = None

            else:
                car.forward()  # Continue moving forward

            time.sleep(0.1)

    except KeyboardInterrupt:
        stdscr.addstr(10, 0, "Stopped by user.")
    finally:
        camera.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    curses.wrapper(main)