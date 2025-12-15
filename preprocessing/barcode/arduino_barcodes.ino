/*
  Arduino barcode generator for 2o-313
  Written by Sumiya Kuroda, Mrsic-Flogel lab

  Heavily adapted from exisitng pipelines:
  Optogenetics and Neural Engineering Core ONE Core at bit.ly/onecore
  Open Ephys at https://open-ephys.org/
*/
#include <LiquidCrystal.h> 

/////////////////LCD Panel Setting///////////////////////////////////
LiquidCrystal lcd(8, 9, 4, 5, 6, 7); 
int lcd_key     = 0; 
int adc_key_in  = 0; 
#define btnLEFT   3 
#define btnUP     1 
#define btnDOWN   2 
#define btnRIGHT  0 
#define btnSELECT 4 
#define btnNONE   5 

/////////////////Barcode Setting///////////////////////////////////
const int TOTAL_TIME = 1000;     // Total time between barcode initiation (includes initialization pulses) in milliseconds. The length of time between one barcode and the next
const int LED_INTER_PIN = 13; // LED indicator for interbarcode delay
const int OUTPUT_PIN = 2;   // Digital pin to output the Barcode TTL
const int BARCODE_BITS = 16;   // Beetle (and uno) use 32 bits
byte PREVIOUS_BUTTON_STATE = HIGH;         // What state was the button last time. Initiate as HIGH
const int BARCODE_TIME = 30;  // time for each bit of the barcode to be on/off in milliseconds
const int INITIALIZATION_TIME = 10;  // We warp the beginning and ending of the barcode with 'some signal', well distinct from a barcode pulse, in milliseconds

const int INITIALIZATION_PULSE_TIME = 3 * INITIALIZATION_TIME;  // We wrap the barcode with a train of on/off/on initialization, then we have the barcode,
// and again on/off/on initialization.
const int TOTAL_BARCODE_TIME = 2 * INITIALIZATION_PULSE_TIME + BARCODE_TIME * BARCODE_BITS; // the total time for the initialization train and barcode signal
const int WAIT_TIME = TOTAL_TIME - TOTAL_BARCODE_TIME; // the total time we wait until starting the next wrapped barcode

static bool isRunning = false;
int barcode;   // initialize a variable to hold our barcode

/////////////////Functions///////////////////////////////////
void setup() {
  pinMode(OUTPUT_PIN, OUTPUT); // initialize digital pin
  pinMode(LED_INTER_PIN, OUTPUT); // initialize digital pin

  barcode = 1000; // random(0, pow(2, BARCODE_BITS)); // generates a random number between 0 and 2^16 (4294967296)
  // (example: if barcode = 4, in binary that would be 0000000000000100)

  lcd.begin(16, 2);            // start the LCD library 
  lcd.setCursor(0, 0); 
  lcd.print("2p-313 Sync TTL"); // print first line
  lcd.setCursor(0, 1); 
  lcd.print("L:Start R:Stop"); // print second line 
}

void loop() {
  lcd_key = read_LCD_buttons();  // read the buttons 
  lcd.setCursor(0,1); 
  switch (lcd_key)               // depending on which button was pushed, we perform an action 
  {
    case btnLEFT: 
    {
       lcd.print("Running NOW!  ");
       // lcd.print(adc_key_in); // shows the actual threshold voltage at analog input 0 
       isRunning = true; 
       break;
    }
    case btnRIGHT: 
    {
       lcd.print("Stopped.      ");
       // lcd.print(adc_key_in); // shows the actual threshold voltage at analog input 0 
       isRunning = false;
       break;
    }
    case btnNONE: 
    {
       break;
    }
  }
  if (isRunning) OUTPUTBARCODE();  
}

int OUTPUTBARCODE (){
  // start barcode with a distinct pulse to signal the start. high, low, high
  digitalWrite(OUTPUT_PIN, HIGH); delay(INITIALIZATION_TIME);
  digitalWrite(OUTPUT_PIN, LOW); delay(INITIALIZATION_TIME);
  digitalWrite(OUTPUT_PIN, HIGH); delay(INITIALIZATION_TIME);

  barcode += 1;
  // increment barcode on each cycle. Our initial value of barcode = 4
  // (in binary 0000000000000100) becomes barcode = 5 (or 0000000000000101)

  // BARCODE SECTION
  for (int i = 0; i < BARCODE_BITS; i++) // for between 0-15 (we will read all 16 bits)
  {
    int barcodedigit = bitRead(barcode >> i, 0);
    // bitRead(x, n) Reads the bit of number x at bit n. The '>>' is a
    // rightshift bitwise operator. For i=0 (0000000000000101) outputs 1. For
    // i=1 (0000000000000010) outputs 0.

    if (barcodedigit == 1)    // if the digit is 1
    {
      digitalWrite(OUTPUT_PIN, HIGH);  // set the output pin to high
    }
    else
    {
      digitalWrite(OUTPUT_PIN, LOW);   // else set it to low
    }
    delay(BARCODE_TIME);   // delay 30 ms
  }

  // end barcode with a distinct pulse to signal the beginning. high, low, high
  digitalWrite(OUTPUT_PIN, HIGH); delay(INITIALIZATION_TIME);
  digitalWrite(OUTPUT_PIN, LOW); delay(INITIALIZATION_TIME);
  digitalWrite(OUTPUT_PIN, HIGH); delay(INITIALIZATION_TIME);
  digitalWrite(OUTPUT_PIN, LOW); // set it to LOW and reset

  // Then be sure to wait long enough before starting the next barcode
  digitalWrite(LED_INTER_PIN, HIGH);
  delay(WAIT_TIME);
  digitalWrite(LED_INTER_PIN, LOW);
}

// read the buttons 
int read_LCD_buttons () {
  adc_key_in = analogRead(0);      // read the value from the sensor 
  // we add approx 50 to those values and check to see if we are close 
  if (adc_key_in > 1000) return btnNONE; // We make this the 1st option for speed reasons since it will be the most likely result 
  if (adc_key_in < 50)   return btnRIGHT;   
  if (adc_key_in < 195)  return btnUP;  
  if (adc_key_in < 380)  return btnDOWN;
  if (adc_key_in < 555)  return btnLEFT;  
  if (adc_key_in < 790)  return btnSELECT;  
  return btnNONE;  // when all others fail, return None
}
