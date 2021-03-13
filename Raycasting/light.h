#pragma once

/// <summary>
/// Struktura zawierająca informacje o światłach.
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