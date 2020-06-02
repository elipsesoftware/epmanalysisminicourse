# Exemplo prático sobre coleta de dados e algoritmo de compressão

Este exemplo apresenta de maneira simplificada:

* informações relacionadas às variáveis de um PIMS
* mecanismo de coleta de dados em tempo real - usando uma interface de comunicação MQTT
* aplicação do algoritmo de compressão com descarte de dados

## Passos para reprodução

1. Criar 2 variáveis no EPM Server (Basic Variables): **ESP8266_pinLDR** e **ESP8266_pinLDR_C**;
2. Configurar as variáveis para que estejam *operacionais*, *dsiponíveis para armazenar dados*, conversão para *FLOAT*, unidade de medida *%*, limites de *0* a *100* e ambas recebendo dados da mesma interface de comunicação configurada para ler o tópico **epmtr/ldr** do Broker MQTT configurado para a mesma.
3. Aplicação usando o código do arquivo *NodeMCU_LDR.ino* em um NodeMCU - ESP8266 - conforme as images a seguir:

**Esquema**

![Esquemático NodMCU-LDR](NodeMCU_LDR_bb.png?raw=true "esquema NodeMCU-LDR")

**Foto**

![Foto NodMCU-LDR](NodeMCU_LDR.jpg?raw=true "foto NodeMCU-LDR")

## Links úteis

* em breve... : link para o vídeo 5 da série: Minicurso EPM Analysis
* [Serie Sensores - LDR - Internet e Coisas #22](https://youtu.be/9Gl3eXnXCd8)
* [Playlist Curso MQTT - canal Internet e Coisas](https://www.youtube.com/playlist?list=PLMmiQibT0iTblNNF_y6_xZfvid5LcWK6_)
