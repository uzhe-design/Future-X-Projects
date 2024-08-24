import RPi.GPIO as GPIO
import time
import picamera
from tensorflow import keras
from PIL import Image
import numpy as np
import requests 

# Setup GPIO
GPIO.setmode(GPIO.BCM)

# Pins configuration
TRIG = 23
ECHO = 24
DC_MOTOR_PIN = 25
SERVO_PIN = 18
LOAD_SENSOR_PIN = 17
OLED_DISPLAY_PIN = 27

# Setup GPIO pins
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(DC_MOTOR_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(LOAD_SENSOR_PIN, GPIO.IN)
GPIO.setup(OLED_DISPLAY_PIN, GPIO.OUT)

# Setup PWM for servo
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

# Load pre-trained waste classification model
model = keras.models.load_model('/path/to/your/model.h5')

# Initialize camera
camera = picamera.PiCamera()

def distance():
    # Send pulse
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    dist = pulse_duration * 17150
    dist = round(dist, 2)
    return dist

def open_lid():
    # Activate DC motor to open the lid
    GPIO.output(DC_MOTOR_PIN, GPIO.HIGH)
    time.sleep(2)  # Adjust time as necessary
    GPIO.output(DC_MOTOR_PIN, GPIO.LOW)

def close_lid():
    # Activate DC motor to close the lid
    GPIO.output(DC_MOTOR_PIN, GPIO.LOW)

def capture_image():
    camera.capture('/home/pi/image.jpg')

def recognize_waste():
    image = Image.open('/home/pi/image.jpg')
    image = image.resize((224, 224))  # Resize image to fit the model input
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)

    waste_type = np.argmax(prediction)
    if waste_type == 0:
        return "Plastic"
    elif waste_type == 1:
        return "Paper"
    elif waste_type == 2:
        return "Glass"
    else:
        return "General"

def rotate_bin(waste_type):
    if waste_type == "Plastic":
        servo.ChangeDutyCycle(7)  # Adjust angle
    elif waste_type == "Paper":
        servo.ChangeDutyCycle(5)
    elif waste_type == "Glass":
        servo.ChangeDutyCycle(3)
    else:
        servo.ChangeDutyCycle(1)
    time.sleep(2)  # Give time to rotate

def open_inner_lid():
    GPIO.output(DC_MOTOR_PIN, GPIO.HIGH)
    time.sleep(1)  # Time to open inner lid
    GPIO.output(DC_MOTOR_PIN, GPIO.LOW)

def monitor_fill_level():
    load_value = GPIO.input(LOAD_SENSOR_PIN)
    return load_value > 0.95  

def send_fill_alert():
    url = 'https://futurex-central-management-system.example.com/alert'
    
    # Data to be sent to the CMS
    data = {
        "bin_id": "BIN_001",  # Unique ID for this bin
        "location": {
            "latitude": 3.11801,  #Coordinate of Faculty of Engineering Universiti Malaya
            "longitude": 101.65535
        },
        "status": "Full",
        "fill_level": "95%"
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("Alert sent successfully!")
        else:
            print(f"Failed to send alert. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending alert: {e}")

def display_waste_type(waste_type):
    # Use the OLED display to show the type of waste
    print(f"Displaying {waste_type} on OLED")

try:
    while True:
        # Hand Detection
        dist = distance()
        if dist < 10:
            print("Hand detected. Opening lid...")
            open_lid()

            # Wait for trash
            time.sleep(2)

            # Trash Detection
            trash_dist = distance()
            if trash_dist < 10:
                print("Trash detected. Capturing image...")
                capture_image()
                waste_type = recognize_waste()
                print(f"Recognized waste: {waste_type}")

                # Rotate bin and open inner lid
                rotate_bin(waste_type)
                open_inner_lid()

                # Display waste type on OLED
                display_waste_type(waste_type)

                # Close lids after operation
                close_lid()

        # Monitor bin fill level
        if monitor_fill_level():
            print("Bin is over 95% full! Sending alert...")
            send_fill_alert()  # Send alert to the central management system

        time.sleep(1)

except KeyboardInterrupt:
    servo.stop()
    GPIO.cleanup()
    print("Program terminated")
