#pragma once

#include "sphere.h"
#include "light.h"

/// <summary>
/// Funkcja obliczaj�ca kolory pikseli na teksturze wy�wietlanej w oknie.
/// </summary>
/// <param name="dev">Tablica kolor�w pikseli</param>
/// <param name="lights">Struktura zawieraj�ca informacje o �wiat�ach</param>
/// <param name="w">Szeroko�� okna</param>
/// <param name="h">Wysoko�� okna</param>
/// <param name="considerLightColor">Czy �wiat�a maj� o�wietla� na bia�o, czy na swoje kolory</param>
/// <param name="ks">Wsp�czynnik b�ysku do wzoru Phonga</param>
/// <param name="alpha">Wsp�czynnik pot�gowy do wzoru Phonga</param>
extern "C" void calculateBackgroundColor(uchar4* dev, lights lights, int w, int h, bool considerLightColor, float ks, float alpha);

/// <summary>
/// Funkcja generuj�ca losowe kule i zapisuj�ca dane w pami�ci constant karty graficznej.
/// </summary>
extern "C" void sendDataToConstantMemory();