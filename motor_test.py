import RPi.GPIO as GPIO
import time

# Define GPIO pins
IN1 = 17  # Motor input 1
IN2 = 27  # Motor input 2
ENA = 22  # PWM speed control

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

# PWM setup
pwm = GPIO.PWM(ENA, 1000)  # 1kHz frequency
pwm.start(50)  # Start with 50% speed

def motor_forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(70)  # Adjust speed

def motor_backward():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    pwm.ChangeDutyCycle(70)  # Adjust speed

def motor_stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(0)

# Run motor forward for 5 seconds
motor_forward()
time.sleep(5)

# Stop motor
motor_stop()
time.sleep(2)

# Run motor backward for 5 seconds
motor_backward()
time.sleep(5)

# Stop motor and cleanup
motor_stop()
GPIO.cleanup()
