#pragma once

#include "sphere.h"
#include "light.h"

/// <summary>
/// Funkcja obliczaj¹ca kolory pikseli na teksturze wyœwietlanej w oknie.
/// </summary>
/// <param name="dev">Tablica kolorów pikseli</param>
/// <param name="lights">Struktura zawieraj¹ca informacje o œwiat³ach</param>
/// <param name="w">Szerokoœæ okna</param>
/// <param name="h">Wysokoœæ okna</param>
/// <param name="considerLightColor">Czy œwiat³a maj¹ oœwietlaæ na bia³o, czy na swoje kolory</param>
/// <param name="ks">Wspó³czynnik b³ysku do wzoru Phonga</param>
/// <param name="alpha">Wspó³czynnik potêgowy do wzoru Phonga</param>
extern "C" void calculateBackgroundColor(uchar4* dev, lights lights, int w, int h, bool considerLightColor, float ks, float alpha);

/// <summary>
/// Funkcja generuj¹ca losowe kule i zapisuj¹ca dane w pamiêci constant karty graficznej.
/// </summary>
extern "C" void sendDataToConstantMemory();