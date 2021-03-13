#include "sphere.h"
#include "light.h"

/// <summary>
/// Funkcja modyfikuj¹ca wylosowane jednorodne wartoœci
/// </summary>
/// <param name="data">Wylosowane dane</param>
extern "C" void modifyData(spheres* data);

/// <summary>
/// Funkcja przesuwaj¹ca œwiat³a po okrêgu
/// </summary>
/// <param name="lights">WskaŸnik na strukturê œwiate³</param>
extern "C" void moveLights(lights* lights);