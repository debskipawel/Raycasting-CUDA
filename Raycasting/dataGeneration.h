#pragma once

#include "sphere.h"
#include "light.h"

#define curandCheck(x) if (x != CURAND_STATUS_SUCCESS)\
	printf("Error occured with cuRAND\n");

/// <summary>
/// Funkcja generuj¹ca losowe kule.
/// </summary>
/// <returns>Struktura wylosowanych sphere_count kul (sphere_count zdefiniowane w constants.h)</returns>
spheres generateSpheres();

/// <summary>
/// Funkcja generuj¹ca œwiat³a.
/// </summary>
/// <returns>Struktura wygenerowanych œwiate³</returns>
lights generateLights(int count);