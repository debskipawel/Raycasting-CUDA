#include <cuda_runtime.h>

#include "ray.h"

/// <summary>
/// Funkcja obliczaj�ca promie� w danym punkcie tekstury.
/// </summary>
/// <param name="u">Liczba [0.0, 1.0], szeroko�� na jakiej si� znajduje piksel (pixel.x / width)</param>
/// <param name="v">Liczba [0.0, 1.0], wysoko�� na jakiej si� znajduje piksel (pixel.y / height)</param>
/// <returns>Promie� wychodz�cy z kamery, przechodz�cy przez piksel</returns>
__device__ ray getRay(float u, float v);

/// <summary>
/// Funkcja zwracaj�ca wsp�rz�dne kamery.
/// </summary>
/// <returns>Wsp�rz�dne, w kt�rych znajduje si� kamera</returns>
__device__ float3 getOrigin();

/// <summary>
/// Funkcja skaluj�ca aspect ratio ogl�danego ekranu (wykorzystywane przy zmianie rozmiar�w okna).
/// </summary>
/// <param name="w">Szeroko�� okna</param>
/// <param name="h">Wysoko�� okna</param>
__global__ void modifyRatio(int w, int h);

/// <summary>
/// Funkcja skaluj�ca aspect ratio ogl�danego ekranu (wykorzystywane przy zmianie rozmiar�w okna).
/// </summary>
/// <param name="w">Szeroko�� okna</param>
/// <param name="h">Wysoko�� okna</param>
extern "C" void modifyAspectRatio(int width, int height);

/// <summary>
/// Funkcja przesuwaj�ca kamer�.
/// </summary>
/// <param name="direction">Kierunek przesuni�cia (sta�e z constants.h)</param>
extern "C" void moveCamera(int direction);

/// <summary>
/// Funkcja obracaj�ca kamer�.
/// </summary>
/// <param name="tetha">K�t obrotu</param>
extern "C" void rotateCamera(float tetha);