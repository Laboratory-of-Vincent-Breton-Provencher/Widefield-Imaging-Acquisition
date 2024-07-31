//DEFINE OUTPUT PINS
int CAM = 13; // Camera trigger
// int FRAME = 13;
int LED405 = 8;
int LED470 = 9;
int LED525 = 10;
int LED630 = 11;
int LED785 = 12;

//DEFINE INPUT PINS
int STATUS405 = 3; // if high, LED405 is used
int STATUS470 = 4; 
int STATUS525 = 5; 
int STATUS630 = 6; 
int STATUS785 = 7; 
int STATUS_ONOFF = 1;

// int AnalogPin = A5; // Debugging


//To control camera and light source. Also control mode
int FPS = 10; //Hz, 40 FP au 6e, 50 pour moi
bool FLAG = 0; // for camera trigegr
int camDelay = 8000; //delay between light changes and camera trigger, 0.5 ms, to avoid bleedthrought between channels (bon terme?)
int camSig = 1000 + camDelay; // Duration of camera trigger signal (peak), 0.5 ms

unsigned long timeNow = 0;
unsigned long lastBlink = micros()-1000000/FPS;     // offset added to make sure LED will turn on on first cycle


void setup() { 
  // put your setup code here, to run once:
  // Serial.begin(19200);   // debugging with analog pin

  //OUTPUT PINS
  pinMode(CAM, OUTPUT);
  digitalWrite(CAM, LOW);

  pinMode(LED405, OUTPUT);
  pinMode(LED470, OUTPUT);
  pinMode(LED525, OUTPUT);
  pinMode(LED630, OUTPUT);
  pinMode(LED785, OUTPUT);

  digitalWrite(LED405, LOW);
  digitalWrite(LED470, LOW);
  digitalWrite(LED525, LOW);
  digitalWrite(LED630, LOW);
  digitalWrite(LED785, LOW);

  // pinMode(FRAME, OUTPUT);
  // digitalWrite(FRAME,LOW);

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
    digitalWrite(CAM, LOW);
    FLAG = 0;
  }

  else if (digitalRead(STATUS_ONOFF) == HIGH) {
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
      FLAG = 1;    // for camera trigger
    }  // end temporal condition and loop

    else {
      if (digitalRead(LED405) == HIGH || digitalRead(LED470) == HIGH || digitalRead(LED525) == HIGH || digitalRead(LED630) == HIGH || digitalRead(LED785) == HIGH){
        if (FLAG == 1){
          if (timeNow - lastBlink > camDelay){
            digitalWrite(CAM, HIGH);
            FLAG = 0;
          }
        }
      }
      if (timeNow - lastBlink > camSig) {
        digitalWrite(CAM, LOW);
      }
    }  
  } // end ONOFF condition

// //  Change frame pin up or down
//   if (FLAG == 0){
//     digitalWrite(FRAME,HIGH);
//     FLAG = 1;
//   } else {
//     digitalWrite(FRAME,LOW);
//     FLAG = 0;
//   }

}
