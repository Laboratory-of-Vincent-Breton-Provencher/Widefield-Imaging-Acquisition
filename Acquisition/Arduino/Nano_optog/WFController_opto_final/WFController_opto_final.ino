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
int FPS = 50; //Hz
bool FLAG_CAM = 0; // for camera trigger
int camDelay = 8000; //delay between light changes and camera trigger, 8 ms, to avoid contamination between channels
int camSig = 1000 + camDelay; // Duration of camera trigger signal (peak), 0.1 ms

unsigned long timeNow = 0;
unsigned long lastBlink = micros()-1000000/FPS;     // offset added to make sure LED will turn on on first cycle

// optogenetics variables
unsigned long OptoBlinkTime = 3000;    // duration of one OG blink, in us
unsigned long stimTime = 2 *1000000;     // optogenetics stimulation time in us
unsigned long acquTime = 15 *1000000;    // acquisition time for camera after stim in us

unsigned long OptoLastBlink = 0;   // last time of a blink in optogenetics stimulation
unsigned long lastStim = 0;    // time of the last optogenetics stimulation
unsigned long lastAcqu = 0;    // time of the last acquisition in optogenetics

bool FLAG_STIM = 0;   // To know if in a stimulation or in an acquisition stage
bool FLAG_OPTO_ON = 0;   // To know if within a stimulation, opto trigg is on or not (if in a opto blink cycle or not)

// --- Change for variable opto frequency ---
const int NB_LEDS = 5;
int LEDS[] = {LED405, LED470, LED525, LED630, LED785};
int STATUS[] = {STATUS405, STATUS470, STATUS525, STATUS630, STATUS785};
int ledCounter = 0; // Led counter
int lastLEDIndex = 0; // Last LED index

// For multiple frequencies during one session
const int ledBeforeOptoList[] = {3, 5, 10}; // Frequency = FPS / (ledBeforeOpto + 1)
const int NB_FREQ = sizeof(ledBeforeOptoList) / sizeof(ledBeforeOptoList[0]);
int freqIndex = 0;
int ledBeforeOpto = ledBeforeOptoList[0]; // Initial value

// --------- Ajout pour nombre de cycles ---------
int NB_CYCLES = 5;      // Nombre de cycles souhaité
int cycleCounter = 0;   // Compteur de cycles
bool protocolDone = false;
// ----------------------------------------------

// --------- Ajout pour acquisition finale ---------
bool finalAcquisition = false;
unsigned long finalAcqStart = 0;
// -----------------------------------------------

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

  // Check which LED is on and turn it on
  for (int i = 0; i < NB_LEDS; i++) {
    if (digitalRead(STATUS[i]) == HIGH) {
      digitalWrite(LEDS[i], HIGH);
      lastLEDIndex = i;
      break;
    }
  }
}

void loop() {
  timeNow = micros();

  // --------- Gestion acquisition finale ---------
  if (protocolDone && !finalAcquisition) {
    // Démarre l'acquisition finale
    finalAcquisition = true;
    finalAcqStart = timeNow;
    FLAG_STIM = 0; // Acquisition mode
    ledCounter = 0;
    // Les LEDs vont continuer à clignoter normalement
  }
  if (protocolDone && finalAcquisition) {
    // Acquisition finale en cours
    if (timeNow - finalAcqStart >= acquTime) {
      // Acquisition finale terminée, on arrête tout
      for (int i = 0; i < NB_LEDS; i++) digitalWrite(LEDS[i], LOW);
      digitalWrite(LEDopto, LOW);
      digitalWrite(CAM, LOW);
      return;
    }
  }
  // ---------------------------------------------

  // Arrêt du protocole après NB_CYCLES (avant acquisition finale)
  if (protocolDone && finalAcquisition == false) {
    // On laisse la gestion à la section acquisition finale ci-dessus
    // (ne rien faire ici)
  }
  else if (digitalRead(STATUS_ONOFF) == LOW) {
    for (int i = 0; i < NB_LEDS; i++) digitalWrite(LEDS[i], LOW);
    digitalWrite(LEDopto, LOW);
    digitalWrite(CAM, LOW);
    FLAG_CAM = 0;
    FLAG_STIM = 0;
    FLAG_OPTO_ON = 0;
    lastAcqu = timeNow;
    lastStim = timeNow;
    ledCounter = 0;
    lastLEDIndex = 0;
    freqIndex = 0;
    ledBeforeOpto = ledBeforeOptoList[0];
    cycleCounter = 0;
    protocolDone = false;
    finalAcquisition = false;
    finalAcqStart = 0;
  } 
  else if (digitalRead(STATUS_ONOFF) == HIGH && !(protocolDone && finalAcquisition)) {
    // Alternate acquisition/stimulation
    if (FLAG_STIM == 0) {
      if (timeNow - lastStim >= acquTime) {
        FLAG_STIM = 1;
        lastAcqu = timeNow;
      }
    } else if (FLAG_STIM == 1) {
      if (timeNow - lastAcqu >= stimTime) {
        FLAG_STIM = 0;
        lastStim = timeNow;
        ledCounter = 0;

        // Next frequency
        freqIndex = (freqIndex + 1) % NB_FREQ;
        ledBeforeOpto = ledBeforeOptoList[freqIndex];

        // Incrémenter le compteur de cycles à chaque retour à la première fréquence
        if (freqIndex == 0) {
          cycleCounter++;
          if (cycleCounter >= NB_CYCLES) {
            protocolDone = true;
          }
        }
      }
    }

    // BLINKING LOOP 
    if (timeNow - lastBlink > 1000000/FPS) {
      int currentLED = -1;
      // Trouver la LED actuellement allumée
      for (int i = 0; i < NB_LEDS; i++) {
        if (digitalRead(LEDS[i]) == HIGH) {
          currentLED = i;
          digitalWrite(LEDS[i], LOW);
          break;
        }
      }

      // Gestion du compteur et de l'opto
      if (ledCounter >= ledBeforeOpto && FLAG_STIM == 1) {
        digitalWrite(LEDopto, HIGH);
        FLAG_OPTO_ON = 1;
        OptoLastBlink = timeNow;
        ledCounter = 0;
        
      } else {
        // Alterner automatiquement les LEDs en reprenant là où on s'est arrêté
        int nextLED;
        if (currentLED == -1) {
          // Si aucune LED n'est allumée (après opto), reprendre à la suivante
          nextLED = (lastLEDIndex + 1) % NB_LEDS;
        } else {
          nextLED = (currentLED + 1) % NB_LEDS;
        }
        digitalWrite(LEDS[nextLED], HIGH);
        ledCounter++;
        lastLEDIndex = nextLED; // Mémorise la dernière LED allumée
      }

      lastBlink = timeNow;
      if (FLAG_OPTO_ON == 0) FLAG_CAM = 1;
    }

    // Gestion extinction opto
    if (digitalRead(LEDopto) == HIGH && (timeNow - OptoLastBlink > OptoBlinkTime)) {
      digitalWrite(LEDopto, LOW);
      FLAG_OPTO_ON = 0;
    }

    // Déclenchement caméra (si une LED est allumée)
    bool anyLEDOn = false;
    for (int i = 0; i < NB_LEDS; i++) {
      if (digitalRead(LEDS[i]) == HIGH) {
        anyLEDOn = true;
        break;
      }
    }
    if (anyLEDOn && FLAG_CAM == 1 && (timeNow - lastBlink > camDelay)) {
      digitalWrite(CAM, HIGH);
      FLAG_CAM = 0;
    }
    if (timeNow - lastBlink > camSig) {
      digitalWrite(CAM, LOW);
    }
  }
}