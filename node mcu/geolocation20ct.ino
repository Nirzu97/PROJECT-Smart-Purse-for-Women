#include <FirebaseESP8266.h>
#include <FirebaseESP8266HTTPClient.h>
#include <FirebaseJson.h>
#include <jsmn.h>

#include <SoftwareSerial.h>
#include<ArduinoJson.h>

#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>

#define FIREBASE_HOST "https://smart-purse-9d542.firebaseio.com/"
#define FIREBASE_AUTH "Mc9XCOzzb5aunF2LzQkmkz9f2mMdoOd6hbHhMwAf"
FirebaseData firebaseData;


SoftwareSerial bt(D6, D7);

char myssid[] = "Gf 1 2.4";         
char mypass[] = "19121993";         
const char* Host = "www.googleapis.com";
String thisPage = "/geolocation/v1/geolocate?key=";
int status = WL_IDLE_STATUS;

String jsonString = "{\n";
double latitude    = 0.0;
double longitude   = 0.0;
double accuracy    = 0.0;
int more_text = 1; 
 
float lat,lon,alt,spe;
    String path = "Smart_Purse/location";


void setup()  
{
  Serial.begin(9600);
 bt.begin(9600);
 WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);
Serial.println("Setup done");
  WiFi.begin(myssid, mypass);
  
  while (WiFi.status() !=WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
   Firebase.begin(FIREBASE_HOST, FIREBASE_AUTH);
    Firebase.setReadTimeout(firebaseData, 1000 * 60);
 
  Firebase.setwriteSizeLimit(firebaseData, "tiny");

 
//  timer.setInterval(5000L,loc);
  pinMode(D2,INPUT_PULLUP);
//  gpio.mode(15,gpio.INPUT,gpio.FLOAT);
//digitalWrite(D8,LOW);
  pinMode(D1,OUTPUT);
  digitalWrite(D1,LOW);
  //attachInterrupt(digitalPinToInterrupt(4),mail,RISING);
  pinMode(D5,INPUT_PULLUP);
 //digitalWrite(D5,LOW);
  //attachInterrupt(digitalPinToInterrupt(14),cell,RISING);
  flocation();
}


void loop()
{
//Serial.println("loop");
if(digitalRead(D2)==LOW)
{
  digitalWrite(D2,HIGH);
  mail();
  digitalWrite(D2,HIGH);
}
  if(digitalRead(D5)==LOW)
{
  digitalWrite(D5,HIGH);
  cell();
  digitalWrite(D5,HIGH);
}
  if(bt.available())
  {
  
    char x=bt.read();
    bt.print("9");
    if(x==1 or x=='1')
    {
      digitalWrite(D1,HIGH);
  delay(500);
  digitalWrite(D1,LOW);
  delay(100);
  digitalWrite(D1,HIGH);
  delay(500);
  digitalWrite(D1,LOW);
  delay(100);
  digitalWrite(D1,HIGH);
  delay(500);
  digitalWrite(D1,LOW);
  delay(100);
  digitalWrite(D1,HIGH);
  delay(500);
  digitalWrite(D1,LOW);
  Serial.println("buzz purse");
    }
      
      
    
    Serial.println(x);
  }
  else
  {int t1=0,t=millis();
    while(t1-t<5000)
    {
      delay(500);
      t1=millis();
      
    }
    if(bt.available())
    {
    }
    else
    {
      int t1=0,t=millis();
    while(t1-t<10000)
    {
      delay(500);
      t1=millis();
      
    }
       if(bt.available())
    {
    }
    else{
      Serial.println("distance");
  digitalWrite(D1,HIGH);
  delay(500);
  digitalWrite(D1,LOW);
  delay(100);
  digitalWrite(D1,HIGH);
  delay(500);
  digitalWrite(D1,LOW);
  delay(100);
  digitalWrite(D1,HIGH);
  delay(500);
  digitalWrite(D1,LOW);
  delay(100);
  digitalWrite(D1,HIGH);
  delay(500);
  digitalWrite(D1,LOW);
    }
    }
  }
  
  

  
}
void mail()
{
  int i=1;
  
  Serial.println("SOS");
  digitalWrite(D1,HIGH);
  delay(1000);
  digitalWrite(D1,LOW);
  delay(1000);
  digitalWrite(D1,HIGH);
  delay(1000);
  digitalWrite(D1,LOW);

  bt.print(i);
  //flocation();
}

void cell()
{
  int i=2;
  Serial.println("cell buzz");
  bt.print(i);
}















void flocation()
{
  Serial.println("Start");
 
  
  char bssid[6];
  DynamicJsonBuffer jsonBuffer;
  
  Serial.println("scan start");
  int n = WiFi.scanNetworks();
  Serial.println("scandone");

  if (n == 0)
    Serial.println("no networks found");
  else
  {
    Serial.print(n);
    Serial.println(" networksfound...");
    if (more_text) 
    {    
      Serial.println("{");
      Serial.println("\"homeMobileCountryCode\":234,");  
      Serial.println("\"homeMobileNetworkCode\":27,");  
      Serial.println("\"radioType\":\"gsm\",");         
      Serial.println("\"carrier\":\"Vodafone\",");       
      Serial.println("\"cellTowers\": [");                    
      Serial.println("],");     
      Serial.println("\"wifiAccessPoints\": [");

      for (int i = 0; i < n;++i)
      {
        Serial.println("{");
        Serial.print("\"macAddress\" : \"");
        Serial.print(WiFi.BSSIDstr(i));
        Serial.println("\",");
        Serial.print("\"signalStrength\": ");
        Serial.println(WiFi.RSSI(i));

        if (i < n - 1)
        {
          Serial.println("},");
        }
        else
        {         
          Serial.println("}");
        }
      }     
    Serial.println("]");
    Serial.println("}");
    }
    Serial.println(" ");

  }
  jsonString = "{\n";
  jsonString +="\"homeMobileCountryCode\": 234,\n"; 
  jsonString +="\"homeMobileNetworkCode\": 27,\n";  
  jsonString +="\"radioType\": \"gsm\",\n";        
  jsonString +="\"carrier\": \"Vodafone\",\n";      
  jsonString +="\"wifiAccessPoints\": [\n";

  for (int j = 0; j < n; ++j)
  {
    jsonString += "{\n";
    jsonString +="\"macAddress\" : \"";
    jsonString +=(WiFi.BSSIDstr(j));
    jsonString +="\",\n";
    jsonString +="\"signalStrength\": ";
    jsonString += WiFi.RSSI(j);
    jsonString += "\n";
    if (j < n - 1)
    {
      jsonString +="},\n";
    }
    else
    {
      jsonString +="}\n";
    }
  }

  jsonString += ("]\n");
  jsonString += ("}\n");

  Serial.println("");

  WiFiClientSecure client;
  client.setInsecure();
  
  if (client.connect(Host,443))
  {
    Serial.println("Connected");
    client.println("POST " + thisPage + "AIzaSyCwMP_mkzW66lye5P7fz7tQlyVmM47poJE" + " HTTP/1.1");
    client.println("Host: " + (String)Host);
    client.println("User-Agent: NodeMCU (ESP32)");
    client.println("Content-Type: application/json");
    client.println("Connection: Keep-Alive");
    client.print("Content-Length: ");
    client.println(jsonString.length());
    client.println();
    client.print(jsonString);
    delay(500);
  }
  
  while (client.available()) 
  {
    String line =client.readStringUntil('\r');
    if (more_text)
    {
      Serial.print(line);
    }
    JsonObject& root =jsonBuffer.parseObject(line);
    if (root.success())
    {
      latitude    =root["location"]["lat"];
      longitude   =root["location"]["lng"];
      accuracy   = root["accuracy"];
    }
  }

  Serial.println("closing connection");
  Serial.println();
  client.stop();
  Serial.print("Latitude =");
  Serial.println(latitude);
  Serial.print("Longitude =");
  Serial.println(longitude);
  Serial.print("Accuracy =");
  Serial.println(accuracy);
  
  
  //delay(10000);
 
Firebase.setDouble(firebaseData, path + "/lat", latitude);
Firebase.setDouble(firebaseData, path + "/lon", longitude);

  Serial.println("blynk stop");
}
