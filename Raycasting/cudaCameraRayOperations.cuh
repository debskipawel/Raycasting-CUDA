#include <cuda_runtime.h>

#include "ray.h"

/// <summary>
/// Funkcja obliczaj¹ca promieñ w danym punkcie tekstury.
/// </summary>
/// <param name="u">Liczba [0.0, 1.0], szerokoœæ na jakiej siê znajduje piksel (pixel.x / width)</param>
/// <param name="v">Liczba [0.0, 1.0], wysokoœæ na jakiej siê znajduje piksel (pixel.y / height)</param>
/// <returns>Promieñ wychodz¹cy z kamery, przechodz¹cy przez piksel</returns>
__device__ ray getRay(float u, float v);

/// <summary>
/// Funkcja zwracaj¹ca wspó³rzêdne kamery.
/// </summary>
/// <returns>Wspó³rzêdne, w których znajduje siê kamera</returns>
__device__ float3 getOrigin();

/// <summary>
/// Funkcja skaluj¹ca aspect ratio ogl¹danego ekranu (wykorzystywane przy zmianie rozmiarów okna).
/// </summary>
/// <param name="w">Szerokoœæ okna</param>
/// <param name="h">Wysokoœæ okna</param>
__global__ void modifyRatio(int w, int h);

/// <summary>
/// Funkcja skaluj¹ca aspect ratio ogl¹danego ekranu (wykorzystywane przy zmianie rozmiarów okna).
/// </summary>
/// <param name="w">Szerokoœæ okna</param>
/// <param name="h">Wysokoœæ okna</param>
extern "C" void modifyAspectRatio(int width, int height);

/// <summary>
/// Funkcja przesuwaj¹ca kamerê.
/// </summary>
/// <param name="direction">Kierunek przesuniêcia (sta³e z constants.h)</param>
extern "C" void moveCamera(int direction);

/// <summary>
/// Funkcja obracaj¹ca kamerê.
/// </summary>
/// <param name="tetha">K¹t obrotu</param>
extern "C" void rotateCamera(float tetha);