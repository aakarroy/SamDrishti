#include <Arduino.h>
#include <Wire.h>
#include <VL53L0X.h>
#include <SparkFun_VL53L5CX_Library.h>

// --- UART & mmWave Configuration ---
#define RADAR_RX_PIN 16 
#define RADAR_TX_PIN 17 
#define RADAR_BAUD 256000 
HardwareSerial RadarSerial(1);
const int MAX_FRAME_LEN = 64;
uint8_t rx_buf[MAX_FRAME_LEN];
int rx_count = 0;

int lastRadarDist = 8000;
unsigned long lastRadarTime = 0;
bool radarOverride = false;
String radarDirection = "FRONT";

// --- I2C Multiplexer & ToF Configuration ---
#define TCAADDR 0x70
VL53L0X tofLeft;
VL53L0X tofRight;
SparkFun_VL53L5CX tofFront;

const int MAX_DISTANCE = 200;    // 8 meters max limit
int lastFrontDist = MAX_DISTANCE; // Memory for non-blocking 5CX

// --- Motor & Haptic Configuration ---
const int MOTOR_LEFT_PIN = 25; 
const int MOTOR_RIGHT_PIN = 26;

const int PWM_FREQ = 1000;
const int PWM_RESOLUTION = 8;     // 0-255
const int PWM_40_PERCENT = 102;   // 40% of 255

// Helper to switch I2C channels on the TCA9548A
void tcaSelect(uint8_t i) {
  if (i > 7) return;
  Wire.beginTransmission(TCAADDR);
  Wire.write(1 << i);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Wire.begin(); 
  
  // Initialize Radar
  RadarSerial.begin(RADAR_BAUD, SERIAL_8N1, RADAR_RX_PIN, RADAR_TX_PIN);

  // Initialize Motors & Fix Default-On issue
  pinMode(MOTOR_LEFT_PIN, OUTPUT);
  digitalWrite(MOTOR_LEFT_PIN, LOW);
  ledcSetup(0, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(MOTOR_LEFT_PIN, 0);
  ledcWrite(0, 0); 

  pinMode(MOTOR_RIGHT_PIN, OUTPUT);
  digitalWrite(MOTOR_RIGHT_PIN, LOW);
  ledcSetup(1, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(MOTOR_RIGHT_PIN, 1);
  ledcWrite(1, 0);

  // Initialize ToF Sensors via Multiplexer
  tcaSelect(0); // Port 0 for Left ToF
  tofLeft.setTimeout(500);
  if (!tofLeft.init()) Serial.println("Failed to boot ToF Left (Port 0)");

  tcaSelect(1); // Port 1 for Right ToF
  tofRight.setTimeout(500);
  if (!tofRight.init()) Serial.println("Failed to boot ToF Right (Port 1)");

  tcaSelect(2); // Port 2 for Front 5CX
  if (!tofFront.begin()) {
    Serial.println("Failed to boot ToF Front (Port 2)");
  } else {
    tofFront.setResolution(8 * 8); // Enable all 64 zones
    tofFront.setRangingFrequency(15); 
    tofFront.startRanging();
  }
}

void processRadarData() {
  if (rx_buf[0] == 0xAA && rx_buf[1] == 0xFF) {
    if (rx_buf[4] == 0x00 && rx_buf[5] == 0x00 && rx_buf[6] == 0x00 && rx_buf[7] == 0x00) {
      radarOverride = false; 
    } else {
      int16_t raw_x = (rx_buf[5] << 8) | rx_buf[4];
      int16_t raw_y = (rx_buf[7] << 8) | rx_buf[6];
      
      int16_t x_pos = raw_x & 0x7FFF;
      if ((raw_x & 0x8000) == 0) x_pos = -x_pos;

      int16_t y_pos = raw_y & 0x7FFF;
      if ((raw_y & 0x8000) == 0) y_pos = -y_pos;

      int currentDist = sqrt((x_pos * x_pos) + (y_pos * y_pos));
      unsigned long currentTime = millis();
      
      if (lastRadarTime > 0 && currentTime > lastRadarTime) {
        float dt = (currentTime - lastRadarTime) / 1000.0;
        float speed = (currentDist - lastRadarDist) / dt; // mm/s
        
        // Trigger: Closer than 2.5m AND approaching faster than 1m/s
        if (currentDist < 2500 && speed < -1000) {
          radarOverride = true;
          if (x_pos < -300) radarDirection = "LEFT";
          else if (x_pos > 300) radarDirection = "RIGHT";
          else radarDirection = "FRONT";
        } else {
          radarOverride = false;
        }
      }
      
      lastRadarDist = currentDist;
      lastRadarTime = currentTime;
    }
  }
}

int getFront5CXMinDistance() {
  tcaSelect(2);
  if (tofFront.isDataReady()) {
    VL53L5CX_ResultsData data;
    tofFront.getRangingData(&data);
    
    int currentMin = MAX_DISTANCE;
    for (int i = 0; i < 64; i++) {
      uint8_t status = data.target_status[i];
      if (status == 5 || status == 6 || status == 9 || status == 10) {
        if (data.distance_mm[i] > 0 && data.distance_mm[i] < currentMin) {
          currentMin = data.distance_mm[i];
        }
      }
    }
    lastFrontDist = currentMin; 
  }
  return lastFrontDist;
}

void loop() {
  // 1. Process mmWave
  while (RadarSerial.available()) {
    uint8_t incomingByte = RadarSerial.read();
    if (rx_count >= MAX_FRAME_LEN) rx_count = 0;
    rx_buf[rx_count++] = incomingByte;
    
    if (rx_count > 1 && rx_buf[rx_count - 2] == 0x55 && rx_buf[rx_count - 1] == 0xCC) {
      processRadarData();
      rx_count = 0; 

      // 2. Logic Execution
      if (radarOverride) {
        Serial.print("OVERRIDE,APPROACHING,Dir:");
        Serial.print(radarDirection);
        Serial.print(",Dist:");
        Serial.println(lastRadarDist);

        // Turn on both motors to 40%
        ledcWrite(0, PWM_40_PERCENT);
        ledcWrite(1, PWM_40_PERCENT);
        
      } else {
        tcaSelect(0);
        int distLeft = tofLeft.readRangeSingleMillimeters();
        if (tofLeft.timeoutOccurred() || distLeft > MAX_DISTANCE) distLeft = MAX_DISTANCE;

        tcaSelect(1);
        int distRight = tofRight.readRangeSingleMillimeters();
        if (tofRight.timeoutOccurred() || distRight > MAX_DISTANCE) distRight = MAX_DISTANCE;

        int distFront = getFront5CXMinDistance();

        int globalMin = distLeft;
        String closestDir = "LEFT";

        if (distRight < globalMin) {
          globalMin = distRight;
          closestDir = "RIGHT";
        }
        if (distFront < globalMin) {
          globalMin = distFront;
          closestDir = "FRONT";
        }

        Serial.print("NORMAL,MinDist:");
        Serial.print(globalMin);
        Serial.print(",Dir:");
        Serial.println(closestDir);

        if (globalMin >= MAX_DISTANCE) {
          ledcWrite(0, 0); 
          ledcWrite(1, 0); 
        } else if (closestDir == "LEFT") {
          ledcWrite(0, PWM_40_PERCENT); 
          ledcWrite(1, 0);
        } else if (closestDir == "RIGHT") {
          ledcWrite(0, 0);
          ledcWrite(1, PWM_40_PERCENT); 
        } else if (closestDir == "FRONT") {
          ledcWrite(0, PWM_40_PERCENT); 
          ledcWrite(1, PWM_40_PERCENT); 
        }
      }
    }
  }
}