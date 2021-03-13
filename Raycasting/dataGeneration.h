#pragma once

#include "sphere.h"
#include "light.h"

#define curandCheck(x) if (x != CURAND_STATUS_SUCCESS)\
	printf("Error occured with cuRAND\n");

/// <summary>
/// Funkcja generuj�ca losowe kule.
/// </summary>
/// <returns>Struktura wylosowanych sphere_count kul (sphere_count zdefiniowane w constants.h)</returns>
spheres generateSpheres();

/// <summary>
/// Funkcja generuj�ca �wiat�a.
/// </summary>
/// <returns>Struktura wygenerowanych �wiate�</returns>
lights generateLights(int count);