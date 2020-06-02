/*
* Elipse Plant Manager - EPM Processor - Thermal Comfort Fuzzy KPI - video example
* Copyright (C) 2020 Elipse Software.
* Distributed under the MIT License.
* (See accompanying file LICENSE.txt or copy at http://opensource.org/licenses/MIT)
*/

#include <ESP8266WiFi.h>
#include <PubSubClient.h>

#define LOOPDELAY 2000  // 2 segundos para cada leitura do sensor e publicação no Mqtt Broker
//#define DEBUG         //Se descomentar esta linha vai habilitar a 'impressão' na porta serial

// Dados da rede Wifi
const char* ssid = "Nome_da_rede";
const char* password = "senha_da_rede";
IPAddress ip(10, 0, 0, 100);        // IP estático - poderia usar IP dinâmico
IPAddress gateway(10, 0, 0, 1);     // IP do roteador da sua rede wifi
IPAddress subnet(255, 255, 255, 0); // Máscara de rede da sua rede wifi

// Configuração do MQTT Broker
#define MQTTSERVER         "Endereço_do_Broker" //URL do servidor MQTT
#define MQTTSERVERPORT     1883                 //Porta do servidor MQTT
#define MQTTSERVERUSER     "usuário"            //Usuário
#define MQTTSERVERPASSWORD "senha"              //Senha
#define MQTTPUBTOPICLDR    "epmtr/ldr"          // Tópico

// Configuração dos pinos do ESP8266 NodeMCU
const int PIN_LDR = A0; //pino LDR

// Variáveis globais
WiFiClient espClient;            //Instância do WiFiClient
PubSubClient client(espClient);  //Passando a instância do WiFiClient para a instância do PubSubClient

// Funções auxiliares
void printSerial(String msg, bool nline = true);
String readSensor2Mqtt();
void reconnectMqtt();
void disconnectMqtt();
void pubMqttBroker();

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Funções SETUP & LOOP
/////////////////////////////////////////////////////////////////////////////////////////////////////////

void setup()
{
  #ifdef DEBUG
    Serial.begin(9600);
  #endif
  
  //Conectando a rede Wifi
  WiFi.config(ip, gateway, subnet);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    hold(500);
  }
 
  printSerial("WiFi connected");
  printSerial("IP address: ");
  printSerial(WiFi.localIP().toString());
  
  pinMode(PIN_LDR, INPUT);

  client.setServer(MQTTSERVER, MQTTSERVERPORT);
}

void loop()
{
   printSerial("-------------------------------------------------------------------  ", false);
   printSerial("Iniciando Loop ...");

    // MQTT Client
    if (!client.connected()){
      reconnectMqtt();
      printSerial("------------  LOOP MQTT iniciado  ------------");
      }
      pubMqttBroker();
      hold(LOOPDELAY);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Funções auxiliares
/////////////////////////////////////////////////////////////////////////////////////////////////////////

//Função para imprimir na porta serial
void printSerial(String msg, bool nline)
{
  #ifdef DEBUG
    if(nline)
      Serial.println(msg);
    else
      Serial.print(msg);
  #endif
}

// Função para leitura do sensor e composição da mensagem para publicação no MQTT Broker
String readSensor2Mqtt()
{
  float lPercent = 0.0;
  int sensorValue = analogRead(A0);           // Ler o pino Analógico A0 onde está o LDR
  lPercent = sensorValue * (100.0 / 1024.0);  // Converter a leitura analógica (que vai de 0 - 1023) para percentual
  Serial.println(lPercent);
  return String(lPercent);
}

// *** MQTT (pub/sub) ***
void reconnectMqtt()
{
  while(!client.connected()) {
    bool conectado = client.connect("EPMESP8266Client", MQTTSERVERUSER, MQTTSERVERPASSWORD, "willTopic", 0, 1, "", true);
    if(conectado) {
    printSerial("Mqtt Broker conectado!");
    }
    else {
      printSerial("Falhou ao tentar conectar. Codigo: ", false);
      printSerial( String(client.state()).c_str()), false;
      printSerial(" tentando novamente em 5 segundos");
      hold(5000);
      reconnectMqtt();
    }
  }
}

void disconnectMqtt()
{
  printSerial("Fechando a conexao com o servidor MQTT...");
  client.disconnect();
}

//Função que envia os dados de intensidade luminosa.
void pubMqttBroker()
{
  if (!client.connected()) {
    printSerial("MQTT desconectado! Tentando reconectar...");
    reconnectMqtt();
  }
  //Publicando no MQTT
  String strMqttMsg = readSensor2Mqtt();
  char* msgMqtt = new char[strMqttMsg.length() + 1];
  strcpy(msgMqtt, strMqttMsg.c_str());
  // Publicação no tópico  MQTTPUBTOPICHC
  client.publish(MQTTPUBTOPICLDR, msgMqtt, true);
  delete [] msgMqtt;
  printSerial("% Luz publicada!!!");
}

// Internet e Coisas - André Michelon
void hold(const unsigned int &ms)
{
  // Non blocking delay
  // Delay para evitar que ocorra reset por travamento do hardware
  // ATENÇÃO! O comando delay pode gerar TRAVAMENTO E RESET do ESP8266!!
  unsigned long m = millis();
  while (millis() - m < ms) {
    yield();
  }
}
