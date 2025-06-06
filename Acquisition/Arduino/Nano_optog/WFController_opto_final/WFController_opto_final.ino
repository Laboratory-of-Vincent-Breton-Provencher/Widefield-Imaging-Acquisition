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

// --------- Gestion des cycles et acquisition finale ---------
int NB_CYCLES = 2; // Nombre de répétitions par fréquence
int freqCounters[NB_FREQ] = {0}; // Compteur pour chaque fréquence
bool protocolDone = false;
bool finalAcquisition = false;
unsigned long finalAcqStart = 0;
unsigned long finalAcqDuration = 5UL * 60UL * 1000000UL; // 5 minutes en microsecondes
// ----------------------------------------------------------

// --------- Acquisition initiale ---------
unsigned long firstAcqDuration = 30UL * 1000000UL; // 30 secondes (modifiable)
bool firstAcqDone = false;
unsigned long firstAcqStart = 0;
// ----------------------------------------

void setup() {
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

  // Si STATUS_ONOFF est LOW, on reset tout
  if (digitalRead(STATUS_ONOFF) == LOW) {
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
    for (int i = 0; i < NB_FREQ; i++) freqCounters[i] = 0;
    protocolDone = false;
    finalAcquisition = false;
    finalAcqStart = 0;
    firstAcqDone = false;
    firstAcqStart = 0;
    return;
  }

  // --------- ACQUISITION INITIALE (avant tout protocole) ---------
  if (!firstAcqDone) {
    if (firstAcqStart == 0) {
      firstAcqStart = timeNow;
      // Allume la première LED active ou la première de la liste
      int firstLED = -1;
      for (int i = 0; i < NB_LEDS; i++) {
        if (digitalRead(STATUS[i]) == HIGH) {
          digitalWrite(LEDS[i], HIGH);
          lastLEDIndex = i;
          firstLED = i;
          break;
        }
      }
      if (firstLED == -1) {
        digitalWrite(LEDS[0], HIGH);
        lastLEDIndex = 0;
      }
      lastBlink = timeNow;
      FLAG_CAM = 1;
    }

    // Alternance LEDs/caméra pendant la durée voulue
    if (timeNow - lastBlink > 1000000/FPS) {
      int currentLED = -1;
      for (int i = 0; i < NB_LEDS; i++) {
        if (digitalRead(LEDS[i]) == HIGH) {
          currentLED = i;
          digitalWrite(LEDS[i], LOW);
          break;
        }
      }
      int nextLED = (currentLED == -1) ? (lastLEDIndex + 1) % NB_LEDS : (currentLED + 1) % NB_LEDS;
      digitalWrite(LEDS[nextLED], HIGH);
      lastLEDIndex = nextLED;
      lastBlink = timeNow;
      FLAG_CAM = 1;
    }

    // Déclenchement caméra
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

    // Fin de l'acquisition initiale
    if (timeNow - firstAcqStart >= firstAcqDuration) {
      for (int i = 0; i < NB_LEDS; i++) digitalWrite(LEDS[i], LOW);
      digitalWrite(CAM, LOW);
      firstAcqDone = true;
      // Prépare le protocole normal
      lastBlink = timeNow;
    }
    return;
  }

  // --------- MODE ACQUISITION FINALE (5min sans opto) ---------
  if (protocolDone) {
    if (!finalAcquisition) {
      finalAcquisition = true;
      finalAcqStart = timeNow;
      // On coupe l'opto
      digitalWrite(LEDopto, LOW);
      FLAG_OPTO_ON = 0;
      FLAG_STIM = 0;
      ledCounter = 0;
    }

    // Pendant 5 minutes, on alterne juste les LEDs et la caméra
    if (timeNow - lastBlink > 1000000/FPS) {
      int currentLED = -1;
      for (int i = 0; i < NB_LEDS; i++) {
        if (digitalRead(LEDS[i]) == HIGH) {
          currentLED = i;
          digitalWrite(LEDS[i], LOW);
          break;
        }
      }
      int nextLED = (currentLED == -1) ? (lastLEDIndex + 1) % NB_LEDS : (currentLED + 1) % NB_LEDS;
      digitalWrite(LEDS[nextLED], HIGH);
      lastLEDIndex = nextLED;
      lastBlink = timeNow;
      FLAG_CAM = 1;
    }

    // Déclenchement caméra
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

    // Fin des 5 minutes
    if (timeNow - finalAcqStart >= finalAcqDuration) {
      for (int i = 0; i < NB_LEDS; i++) digitalWrite(LEDS[i], LOW);
      digitalWrite(CAM, LOW);
      digitalWrite(LEDopto, LOW);
      // On ne refait plus rien, on reste dans ce mode
    }
    return;
  }

  // --------- MODE PROTOCOLE NORMAL (cycles acquisition/stim/opto) ---------
  if (digitalRead(STATUS_ONOFF) == HIGH) {
    // Acquisition/stimulation
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

        // Incrémenter le compteur pour la fréquence courante
        freqCounters[freqIndex]++;
        // Passer à la fréquence suivante
        freqIndex = (freqIndex + 1) % NB_FREQ;
        ledBeforeOpto = ledBeforeOptoList[freqIndex];

        // Vérifier si toutes les fréquences ont atteint NB_CYCLES
        bool allDone = true;
        for (int i = 0; i < NB_FREQ; i++) {
          if (freqCounters[i] < NB_CYCLES) {
            allDone = false;
            break;
          }
        }
        if (allDone) {
          protocolDone = true;
        }
      }
    }

    // BLINKING LOOP 
    if (timeNow - lastBlink > 1000000/FPS) {
      int currentLED = -1;
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
        int nextLED;
        if (currentLED == -1) {
          nextLED = (lastLEDIndex + 1) % NB_LEDS;
        } else {
          nextLED = (currentLED + 1) % NB_LEDS;
        }
        digitalWrite(LEDS[nextLED], HIGH);
        ledCounter++;
        lastLEDIndex = nextLED;
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