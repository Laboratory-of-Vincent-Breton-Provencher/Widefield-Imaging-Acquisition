//DEFINE OUTPUT PINS
int PINMODE_405 = 8; // Activate 405 nm on FPController
int PINMODE_470 = 9; 
int PINMODE_525 = 10; 
int PINMODE_630 = 11; 
int PINMODE_785 = 12; 

//DEFINE INPUT PINS
int in_405 = 3;
int in_470 = 4;
int in_525 = 5;
int in_630 = 6;
int in_785 = 7;
int TTL = 2;

// Flags for light mode (on or off)
bool FLAG405 = 0;
bool FLAG470 = 0;
bool FLAG525 = 0;
bool FLAG630 = 0;
bool FLAG785 = 0;

int val = 0; // serial read value, for light selection

int val405 = 0;
int val470 = 0;
int val525 = 0;
int val630 = 0;
int val785 = 0;

int prev405 = 0;
int prev470 = 0;
int prev525 = 0;
int prev630 = 0;
int prev785 = 0;


int valTTL = 0;
int FLAG = 0; // Increase from 0 to 9 and reset


void setup() {
   // initialize serial communication at 9600 bits per second: (default for communicating with Bonsai)
  Serial.begin(9600);

  // OUTPUT PINS
  pinMode(PINMODE_405, OUTPUT);
  pinMode(PINMODE_470, OUTPUT);
  pinMode(PINMODE_525, OUTPUT);
  pinMode(PINMODE_630, OUTPUT);
  pinMode(PINMODE_785, OUTPUT);

  // Light selection
  digitalWrite(PINMODE_405, LOW);
  digitalWrite(PINMODE_470, LOW);
  digitalWrite(PINMODE_525, LOW);
  digitalWrite(PINMODE_630, LOW);
  digitalWrite(PINMODE_785, LOW);

  // INPUT PINS
  pinMode(in_405, INPUT);
  pinMode(in_470, INPUT);
  pinMode(in_525, INPUT);
  pinMode(in_630, INPUT);
  pinMode(in_785, INPUT);

  pinMode(TTL, INPUT);
  }

void loop() {
 //wait for a command or loop the code in mode > 0
  //while(!(Serial.available()) && mode == 0);

  // Update mode if a command is sent through serial port
  if (Serial.available()){
    val = Serial.read();
    Serial.println(val, DEC);
    
    if (val == '0'){
      FLAG405 = 0;
      FLAG470 = 0;
      FLAG525 = 0;
      FLAG630 = 0;
      FLAG785 = 0;
    } else if (val == '1'){
        if (FLAG405 == 0){FLAG405 = 1;}
        else if (FLAG405 == 1) {FLAG405 = 0;}
    } else if (val == '2'){
        if (FLAG470 == 0){FLAG470 = 1;}
        else if (FLAG470 == 1){FLAG470 = 0;}
    } else if (val == '3'){
        if (FLAG525 == 0){FLAG525 = 1;}
        else if (FLAG525 == 1){FLAG525 = 0;}
    } else if (val == '4'){
        if (FLAG630 == 0){FLAG630 = 1;}
        else if (FLAG630 == 1){FLAG630 = 0;}
    } else if (val == '5'){
        if (FLAG785 == 0){FLAG785 = 1;}
        else if (FLAG785 == 1){FLAG785 = 0;}
    } else {
      FLAG405 = 1;
      FLAG470 = 1;
      FLAG525 = 1;
      FLAG630 = 1;
      FLAG785 = 1;
    }

    //  switch (val){    // didn't work, used if instead

    //  case '0':
    //   FLAG405 = 0;
    //   FLAG470 = 0;
    //   FLAG525 = 0;
    //   FLAG630 = 0;
    //   FLAG785 = 0;
    //   break;
    //  case '1':    // 405 nm
    //   if (FLAG405 == 0){FLAG405 = 1;}
    //   else {FLAG405 = 0;}
    //   break;
    //  case '2':    // 470 nm
    //   if (FLAG470 == 0){FLAG470 = 1;}
    //   else {FLAG470 = 0;}
    //   break;
    //  case '3':    // 525 nm
    //   if (FLAG525 == 0){FLAG525 = 1;}
    //   else {FLAG525 = 0;}
    //   break;
    //  case '4':    // 630 nm
    //   if (FLAG630 == 0){FLAG630 = 1;}
    //   else {FLAG630 = 0;}
    //   break;
    //  case '5':    // 785 nm
    //   if (FLAG785 == 0){FLAG785 = 1;}
    //   else {FLAG785 = 0;}
    //   break;
    //  default:  // reflectance only
    //   FLAG405 = 1;
    //   FLAG470 = 1;
    //   FLAG525 = 1;
    //   FLAG630 = 1;
    //   FLAG785 = 0;
    //   break;
    //  }
       
     //Turn off camera for 2 sec. This helps to know bonsai when camera starts and when arduino starts   
     digitalWrite(PINMODE_405, LOW);
     digitalWrite(PINMODE_470, LOW);
     digitalWrite(PINMODE_525, LOW);
     digitalWrite(PINMODE_630, LOW);
     digitalWrite(PINMODE_785, LOW);
     FLAG = 0; // reset counter to zero
    //  delay(2000); // pause for 2000 ms
     
    if (FLAG405 == 1){
      digitalWrite(PINMODE_405, HIGH);
    } else {digitalWrite(PINMODE_405, LOW);}
    if (FLAG470 == 1){
      digitalWrite(PINMODE_470, HIGH);
    } else {digitalWrite(PINMODE_470, LOW);}
    if (FLAG525 == 1){
      digitalWrite(PINMODE_525, HIGH);
    } else {digitalWrite(PINMODE_525, LOW);}
    if (FLAG630 == 1){
      digitalWrite(PINMODE_630, HIGH);
    } else {digitalWrite(PINMODE_630, LOW);}
    if (FLAG785 == 1){
      digitalWrite(PINMODE_785, HIGH);
    } else {digitalWrite(PINMODE_785, LOW);}
  }

  val405 = digitalRead(in_405);
  val470 = digitalRead(in_470);
  val525 = digitalRead(in_525);
  val630 = digitalRead(in_630);
  val785 = digitalRead(in_785);
  
  valTTL = digitalRead(TTL);

  if ((val405 - prev405) > 0 || (val470 - prev470) > 0 || (val525 - prev525) > 0 || (val630 - prev630) > 0 || (val785 - prev785) > 0) { // only print at the onset of every frame
    Serial.print(val405);
    Serial.print(val470);
    Serial.print(val525);
    Serial.print(val630);
    Serial.print(val785);
    Serial.print(valTTL);
    Serial.println(padding( FLAG, 4));  
    
    FLAG += 1;
    if (FLAG > 9999){
      FLAG = 0;
    }
  }

  // Store pinB and pinV for comparing with next read
  prev405 = val405;
  prev470 = val470;
  prev525 = val525;
  prev630 = val630;
  prev785 = val785;
}

int padding( int number, byte width ) {
  int currentMax = 10;
  for (byte i=1; i<width; i++){
    if (number < currentMax) {
      Serial.print("0");
    }
    currentMax *= 10;
  } 
  return number;
}
