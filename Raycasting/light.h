#pragma once

/// <summary>
/// Struktura zawieraj¹ca informacje o œwiat³ach.
/// </summary>
struct lights
{
	float* x;
	float* y;
	float* z;

	float* r;
	float* g;
	float* b;

	unsigned int count;
};