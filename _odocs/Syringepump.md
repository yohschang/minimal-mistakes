---
title: "Syringe pump"
permalink: /odocs/pump/
excerpt: "Syringe pump"
sitemap: true

sidebar:
  nav: "odocs"

---

- I use arduino to control stepmoter, and push syringe to inject
- the command to control moter can be given out side ex. python  

```c
#include <Servo.h>
#include <Stepper.h>
#define STEPS 200  
int command=0;
int state = 0;
int now = 0;
Servo myservo; 
const int stepPin = 3;                          
const int dirPin = 4;
const int m0 = 10;
const int m1 = 9;
const int m2 = 8;
const int enable = 11;
int count = 10;
int i  = 1;  //delay per step  //6400 step total
int acc_i = 10 ; // for accelerate
void setup()
{
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  digitalWrite(m0, LOW);
  digitalWrite(m1, HIGH);
  digitalWrite(m2, HIGH);
  myservo.attach(5);
  Serial.begin(9600); 
}

void loop()
{ 
if (Serial.available()> 0)
  { command = Serial.read();
    //Serial.print(command);
  }
    Serial.print(now);
    if (command == '1'){      //forward
      digitalWrite(dirPin, LOW);
      now = 1;
    }
    else if (command == '2'){  //backward
      digitalWrite(dirPin, HIGH);
      now = 2;
    }
    else if (command == '3'){
      now =0;    
    }
    else if (command == '5'){
      int i = 1;
      
    }
    if (now!=0){
      count ++;
      if (count >= 100){       // delay r *20 ms before chnage condition
        for (int r = 0; r< 1 ; r++){
        digitalWrite(enable,LOW);
        digitalWrite(stepPin, HIGH);
        delay(acc_i);
        digitalWrite(stepPin, LOW);
        delay(acc_i);
        }
      count =0 ;
      }
      digitalWrite(enable,LOW);
      digitalWrite(stepPin, HIGH);
      delay(i);
      digitalWrite(stepPin, LOW);
      delay(i);
    }
    else{
      digitalWrite(enable,HIGH);
    }
}
```

- Connect and control using python 
```python
import serial
from time import sleep
import sys
PORT = "COM5"
BAUD = 9600
ser = serial.Serial(PORT,BAUD)
try:
    while True:
        while True:
            choice = input("1:backward,2:forward,3:stop").lower()
            if choice == "1":
                ser.write(b'1')
                sleep(0.5)
                break
            elif choice == "2":
                ser.write(b'2')
                sleep(0.5)
                break
            elif choice == "3":
                ser.write(b'3')
                sleep(0.5)
                break
            while ser.inWaiting:
                feed_back = ser.readline().decode()
                print(feed_back)
        if choice == "end":
            sys.exit(0)
except KeyboardInterrupt:
    ser.close()
    print('再見！')

```
