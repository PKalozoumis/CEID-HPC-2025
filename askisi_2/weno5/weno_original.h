#pragma once

void weno_minus_original(
	const float* restrict const a, const float* restrict const b, const float* restrict const c,
	const float* restrict const d, const float* restrict const e, float* restrict const out, const int NENTRIES);