//DEFINE OUTPUT PINS
int CAM = 13; // Camera trigger
int LED405 = 8; // LED trigger for Thorlabs cube
int LED470 = 9;
int LED525 = 10;
int LED630 = 11;
int LED785 = 12;

int LEDopto = 2;   // signal for optogenetics system

//DEFINE INPUT PINS
int STATUS405 = 3; // if high, LED405 is used
int STATUS470 = 4; 
int STATUS525 = 5; 
int STATUS630 = 6; 
int STATUS785 = 7; 
int STATUS_ONOFF = 1;

// int AnalogPin = A5; // Debugging


//To control camera and light source
int FPS = 20; //Hz
bool FLAG_CAM = 0; // for camera trigger
int camDelay = 2000; //delay between light changes and camera trigger, 2 ms, to avoid bleedthrought between channels (bon terme?)
int camSig = 1000 + camDelay; // Duration of camera trigger signal (peak), 0.1 ms

unsigned long timeNow = 0;
unsigned long lastBlink = micros()-1000000/FPS;     // offset added to make sure LED will turn on on first cycle

// optogenetics variables
int OptoBlinkTime = 3000;    // duration of one OG blink, in us
int stimTime = 2 *1000000;     // optogenetics stimulation time in us
int acquTime = 30 *1000000;    // acquisition time for camera after stim in us

unsigned long OptoLastBlink = 0;   // last time of a blink in optogenetics stimulation
unsigned long lastStim = 0;    // time of the last optogenetics stimulation
unsigned long lastAcqu = 0;    // time of the last acquisition in optogenetics

bool FLAG_STIM = 0;   // To know if in a stimulatio or in a acquisition stage
bool FLAG_ACQU = 0;   // optogenetics flag
bool FLAG_OPTO_ON = 0;   // To know if within a stimulation, opto trigg is on or not (if in a opto blink cycle or not)


void setup() { 
  // Serial.begin(19200);   // debugging with analog pin

  //OUTPUT PINS
  pinMode(CAM, OUTPUT);
  digitalWrite(CAM, LOW);

  pinMode(LED405, OUTPUT);
  pinMode(LED470, OUTPUT);
  pinMode(LED525, OUTPUT);
  pinMode(LED630, OUTPUT);
  pinMode(LED785, OUTPUT);
  pinMode(LEDopto, OUTPUT);

  digitalWrite(LED405, LOW);
  digitalWrite(LED470, LOW);
  digitalWrite(LED525, LOW);
  digitalWrite(LED630, LOW);
  digitalWrite(LED785, LOW);
  digitalWrite(LEDopto, LOW);


  // INPUT PINS
  pinMode(STATUS405, INPUT);
  pinMode(STATUS470, INPUT);
  pinMode(STATUS525, INPUT);
  pinMode(STATUS630, INPUT);
  pinMode(STATUS785, INPUT);
  pinMode(STATUS_ONOFF, INPUT);

  if (digitalRead(STATUS405) == HIGH){
    digitalWrite(LED405, HIGH);
  } else if (digitalRead(STATUS470) == HIGH){
    digitalWrite(LED470, HIGH);
  } else if (digitalRead(STATUS525) == HIGH){
    digitalWrite(LED525, HIGH);
  } else if (digitalRead(STATUS630) == HIGH){
    digitalWrite(LED630, HIGH);
  } else if (digitalRead(STATUS785) == HIGH){
    digitalWrite(LED785, HIGH);
  }
}

void loop() {
  // put your main code here, to run repeatedly:
  timeNow = micros();

  if (digitalRead(STATUS_ONOFF) == LOW){
    digitalWrite(LED405, LOW);
    digitalWrite(LED470, LOW);
    digitalWrite(LED525, LOW);
    digitalWrite(LED630, LOW);
    digitalWrite(LED785, LOW);
    digitalWrite(LEDopto, LOW);
    digitalWrite(CAM, LOW);
    FLAG_CAM = 0;
    FLAG_ACQU = 1;
    FLAG_STIM = 0;
  }
 
  else if (digitalRead(STATUS_ONOFF) == HIGH) {
    if (FLAG_ACQU == 1){
      if (timeNow - lastStim < acquTime){
        // nothing, continues to blinking loop
      }
      else if (timeNow - lastStim > acquTime){
        FLAG_ACQU = 0;
        FLAG_STIM = 1;
        lastAcqu = timeNow;
      }
    }

    else if (FLAG_STIM == 1){
      if (timeNow - lastAcqu < stimTime){
        // nothing, to blinking loop
      }
      else if (timeNow - lastAcqu > stimTime){
      FLAG_STIM = 0;
      FLAG_ACQU = 1;
      lastStim = timeNow;
      }
    }

       
    // BLINKING LOOP
    if (timeNow - lastBlink > 1000000/FPS) {  // only once every FPS
      if (digitalRead(LED405) == HIGH){ // if LED405 is on
        digitalWrite(LED405, LOW); // Turning off LED405 
        if (digitalRead(STATUS470) == HIGH){ // Try LEDStatus, when one is high, turn the LED on
          digitalWrite(LED470, HIGH);
        } else if (digitalRead(STATUS525) == HIGH){
          digitalWrite(LED525, HIGH);
        } else if (digitalRead(STATUS630) == HIGH){
          digitalWrite(LED630, HIGH);
        } else if (digitalRead(STATUS785) == HIGH){
          digitalWrite(LED785, HIGH);
        } else if (FLAG_STIM == 1){
          digitalWrite(LEDopto, HIGH)
          FLAG_OPTO_ON = 1;
          OptoLastBlink = timeNow;
        } else if (digitalRead(STATUS405) == HIGH){ // if no other LEDStatus is high, beginning LED is turned back on
          digitalWrite(LED405, HIGH);
        } 
      }

      else if (digitalRead(LED470) == HIGH){ 
        digitalWrite(LED470, LOW); 
        if (digitalRead(STATUS525) == HIGH){
          digitalWrite(LED525, HIGH);
        } else if (digitalRead(STATUS630) == HIGH){
          digitalWrite(LED630, HIGH);
        } else if (digitalRead(STATUS785) == HIGH){
          digitalWrite(LED785, HIGH);
        } else if (FLAG_STIM == 1){
          digitalWrite(LEDopto, HIGH)
          FLAG_OPTO_ON = 1;
          OptoLastBlink = timeNow;
        } else if (digitalRead(STATUS405) == HIGH){ 
          digitalWrite(LED405, HIGH);
        } else if (digitalRead(STATUS470) == HIGH){ 
          digitalWrite(LED470, HIGH);
        }
      }

      else if (digitalRead(LED525) == HIGH){ 
        digitalWrite(LED525, LOW); 
        if (digitalRead(STATUS630) == HIGH){
          digitalWrite(LED630, HIGH);
        } else if (digitalRead(STATUS785) == HIGH){
          digitalWrite(LED785, HIGH);
        } else if (FLAG_STIM == 1){
          digitalWrite(LEDopto, HIGH)
          FLAG_OPTO_ON = 1;
          OptoLastBlink = timeNow;
        } else if (digitalRead(STATUS405) == HIGH){ 
          digitalWrite(LED405, HIGH);
        } else if (digitalRead(STATUS470) == HIGH){ 
          digitalWrite(LED470, HIGH);
        } else if (digitalRead(STATUS525) == HIGH){
          digitalWrite(LED525, HIGH);
        }
      }

      else if (digitalRead(LED630) == HIGH){ 
        digitalWrite(LED630, LOW); 
        if (digitalRead(STATUS785) == HIGH){
          digitalWrite(LED785, HIGH);
        } else if (FLAG_STIM == 1){
          digitalWrite(LEDopto, HIGH)
          FLAG_OPTO_ON = 1;
          OptoLastBlink = timeNow;
        } else if (digitalRead(STATUS405) == HIGH){ 
          digitalWrite(LED405, HIGH);
        } else if (digitalRead(STATUS470) == HIGH){ 
          digitalWrite(LED470, HIGH);
        } else if (digitalRead(STATUS525) == HIGH){
          digitalWrite(LED525, HIGH);
        } else if (digitalRead(STATUS630) == HIGH){
          digitalWrite(LED630, HIGH);
        }
      }

      else if (digitalRead(LED785) == HIGH){ 
        digitalWrite(LED785, LOW);
        if (FLAG_STIM == 1){
          digitalWrite(LEDopto, HIGH)
          FLAG_OPTO_ON = 1;
          OptoLastBlink = timeNow;
        } else if (digitalRead(STATUS405) == HIGH){ 
          digitalWrite(LED405, HIGH);
        } else if (digitalRead(STATUS470) == HIGH){ 
          digitalWrite(LED470, HIGH);
        } else if (digitalRead(STATUS525) == HIGH){
          digitalWrite(LED525, HIGH);
        } else if (digitalRead(STATUS630) == HIGH){
          digitalWrite(LED630, HIGH);
        } else if (digitalRead(STATUS785) == HIGH){
          digitalWrite(LED785, HIGH);
        }
      }

      else if (FLAG_OPTO_ON == 1){
        FLAG_OPTO_ON = 0;
        if (digitalRead(STATUS405) == HIGH){ 
          digitalWrite(LED405, HIGH);
        } else if (digitalRead(STATUS470) == HIGH){ 
          digitalWrite(LED470, HIGH);
        } else if (digitalRead(STATUS525) == HIGH){
          digitalWrite(LED525, HIGH);
        } else if (digitalRead(STATUS630) == HIGH){
          digitalWrite(LED630, HIGH);
        } else if (digitalRead(STATUS785) == HIGH){
          digitalWrite(LED785, HIGH);
        }
      }
      
      else{ // if no LED was found on, try status again and turn first one on
        if (digitalRead(STATUS405) == HIGH){
          digitalWrite(LED405, HIGH);
        } else if (digitalRead(STATUS470) == HIGH){
          digitalWrite(LED470, HIGH);
        } else if (digitalRead(STATUS525) == HIGH){
          digitalWrite(LED525, HIGH);
        } else if (digitalRead(STATUS630) == HIGH){
          digitalWrite(LED630, HIGH);
        } else if (digitalRead(STATUS785) == HIGH){
          digitalWrite(LED785, HIGH);
        }
      }

      lastBlink = timeNow;  // actualise time
      if (FLAG_OPTO_ON == 0){
        FLAG_CAM = 1;    // for camera trigger
      }
    }  // end blinking loop

    else if (digitalRead(LEDopto) == HIGH){
      // temporal condition for optogenetics ttl signal
      if (timeNow - OptoLastBlink > OptoBlinkTime){
        digitalWrite(LEDopto, LOW);
      }
    }

    else {
      if (digitalRead(LED405) == HIGH || digitalRead(LED470) == HIGH || digitalRead(LED525) == HIGH || digitalRead(LED630) == HIGH || digitalRead(LED785) == HIGH){
        if (FLAG_CAM == 1){
          if (timeNow - lastBlink > camDelay){
            digitalWrite(CAM, HIGH);
            FLAG_CAM = 0;
          }
        }
      }
      if (timeNow - lastBlink > camSig) {
        digitalWrite(CAM, LOW);
      }
    } // end camera trigger condition
  } // end ONOFF condition
}
