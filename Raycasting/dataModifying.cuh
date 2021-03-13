#include "sphere.h"
#include "light.h"

/// <summary>
/// Funkcja modyfikuj�ca wylosowane jednorodne warto�ci
/// </summary>
/// <param name="data">Wylosowane dane</param>
extern "C" void modifyData(spheres* data);

/// <summary>
/// Funkcja przesuwaj�ca �wiat�a po okr�gu
/// </summary>
/// <param name="lights">Wska�nik na struktur� �wiate�</param>
extern "C" void moveLights(lights* lights);