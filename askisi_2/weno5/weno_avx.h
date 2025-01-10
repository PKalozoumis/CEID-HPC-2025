#pragma once
#include <x86intrin.h>

float weno_minus_core_reference(const float a, const float b, const float c, const float d, const float e)
{
		const float is0 = a*(a*(float)(4./3.)  - b*(float)(19./3.)  + c*(float)(11./3.)) + b*(b*(float)(25./3.)  - c*(float)(31./3.)) + c*c*(float)(10./3.);
		const float is1 = b*(b*(float)(4./3.)  - c*(float)(13./3.)  + d*(float)(5./3.))  + c*(c*(float)(13./3.)  - d*(float)(13./3.)) + d*d*(float)(4./3.);
		const float is2 = c*(c*(float)(10./3.) - d*(float)(31./3.)  + e*(float)(11./3.)) + d*(d*(float)(25./3.)  - e*(float)(19./3.)) + e*e*(float)(4./3.);

		const float is0plus = is0 + (float)WENOEPS;
		const float is1plus = is1 + (float)WENOEPS;
		const float is2plus = is2 + (float)WENOEPS;

		const float alpha0 = (float)(0.1)*((float)1/(is0plus*is0plus));
		const float alpha1 = (float)(0.6)*((float)1/(is1plus*is1plus));
		const float alpha2 = (float)(0.3)*((float)1/(is2plus*is2plus));
		const float alphasum = alpha0+alpha1+alpha2;
		const float inv_alpha = ((float)1)/alphasum;

		const float omega0 = alpha0 * inv_alpha;
		const float omega1 = alpha1 * inv_alpha;
		const float omega2 = 1-omega0-omega1;

		return omega0*((float)(1.0/3.)*a - (float)(7./6.)*b + (float)(11./6.)*c) +
					 omega1*(-(float)(1./6.)*b + (float)(5./6.)*c + (float)(1./3.)*d) +
					 omega2*((float)(1./3.)*c  + (float)(5./6.)*d - (float)(1./6.)*e);
}

static inline float weno_minus_core(const float a, const float b, const float c, const float d, const float e)
{

  __m512 vec1 = _mm512_set_ps(a, b, c, 0.0f, a, b, c, 0.0f, a, b, c, 0.0f, b, c, d, 0.0f);

  __m256 vec2 = _mm256_set_ps(b, c, d, 0.0f, c, d, e, 0.0f);

  //-----------------------------------------------------------------------------------------------------------

  __m512 vec1_div1 = _mm512_set_ps(4., 4., 10., 0.0f, -19., -13., -31., 0.0f, 11., 5., 11., 0.0f, 25., 13., 25., 0.0f);
  __m512 vec1_div2 = _mm512_set1_ps(3.0f);
  __m512 val1 = _mm512_set_ps(a, b, c, 0.0f, b, c, d, 0.0f, c, d, e, 0.0f, b, c, d, 0.0f);

  __m256 vec2_div1 = _mm256_set_ps(-31.0f, -13.0f, -19.0f, 0.0f, 10.0f, 4.0f, 4.0f, 0.0f);
  __m256 vec2_div2 = _mm256_set1_ps(3.0f);
  __m256 val2 = _mm256_set_ps(c, d, e, 0.0f, c, d, e, 0.0f);

  //-----------------------------------------------------------------------------------------------------------

  __m512 div1 = _mm512_div_ps(vec1_div1, vec1_div2);
  __m512 mul1 = _mm512_mul_ps(val1, div1);

  __m256 div2 = _mm256_div_ps(vec2_div1, vec2_div2);
  __m256 mul2 = _mm256_mul_ps(val2, div2);

  //Calculate first part
  __m512 temp0 = _mm512_mul_ps(vec1, mul1);

  __m128 temp00 = _mm512_extractf32x4_ps(temp0, 0);
  __m128 temp01 = _mm512_extractf32x4_ps(temp0, 1);
  __m128 temp02 = _mm512_extractf32x4_ps(temp0, 2);
  __m128 temp03 = _mm512_extractf32x4_ps(temp0, 3);

  __m128 sum0 = _mm_add_ps(temp00, _mm_add_ps(_mm_add_ps(temp01, temp02), temp03));

  //Calculate second part
  __m256 temp1 = _mm256_mul_ps(vec2, mul2);

  __m128 temp10 = _mm256_extractf32x4_ps(temp1, 0);
  __m128 temp11 = _mm256_extractf32x4_ps(temp1, 1);

  __m128 sum1 = _mm_add_ps(temp10, temp11);

  //--------------------------------------------------------------------

  __m128 is = _mm_add_ps(sum0, sum1);

  __m128 wenoeps = _mm_set1_ps((float)WENOEPS);
  __m128 is_plus = _mm_add_ps(is, wenoeps);
  is_plus = _mm_mul_ps(is_plus, is_plus);

  __m128 alpha = _mm_div_ps(_mm_set_ps(0.1f, 0.6f, 0.3f, 0.0f), is_plus);

  __m128 alphasum = _mm_hadd_ps(alpha, alpha);
  alphasum = _mm_hadd_ps(alphasum, alphasum);

  __m128 omega = _mm_div_ps(alpha, alphasum);

  __m128 omega0_vars = _mm_set_ps(a, b, c, 0);
  __m128 omega0_num = _mm_set_ps(1.0f, -7.0f, 11.0f, 0.0f);
  __m128 omega0_denom = _mm_set_ps(3.0f, 6.0f, 6.0f, 1.0f);
  __m128 omega_temp0 = _mm_mul_ps(_mm_div_ps(omega0_num, omega0_denom), omega0_vars);
  omega_temp0 = _mm_hadd_ps(omega_temp0, omega_temp0);
  omega_temp0 = _mm_hadd_ps(omega_temp0, omega_temp0);

  __m128 omega1_vars = _mm_set_ps(b, c, d, 0);
  __m128 omega1_num = _mm_set_ps(-1.0f, 5.0f, 1.0f, 0.0f);
  __m128 omega1_denom = _mm_set_ps(6.0f, 6.0f, 3.0f, 1.0f);
  __m128 omega_temp1 = _mm_mul_ps(_mm_div_ps(omega1_num, omega1_denom), omega1_vars);
  omega_temp1 = _mm_hadd_ps(omega_temp1, omega_temp1);
  omega_temp1 = _mm_hadd_ps(omega_temp1, omega_temp1);

  __m128 omega2_vars = _mm_set_ps(c, d, e, 0);
  __m128 omega2_num = _mm_set_ps(1.0f, 5.0f, -1.0f, 0.0f);
  __m128 omega2_denom = _mm_set_ps(3.0f, 6.0f, 6.0f, 1.0f);
  __m128 omega_temp2 = _mm_mul_ps(_mm_div_ps(omega2_num, omega2_denom), omega2_vars);
  omega_temp2 = _mm_hadd_ps(omega_temp2, omega_temp2);
  omega_temp2 = _mm_hadd_ps(omega_temp2, omega_temp2);

  float __attribute__((aligned(16))) omega0_arr[4], omega1_arr[4], omega2_arr[4];
  _mm_store_ps(omega0_arr, omega_temp0);
  _mm_store_ps(omega1_arr, omega_temp1);
  _mm_store_ps(omega2_arr, omega_temp2);

  __m128 res = _mm_mul_ps(_mm_set_ps(omega0_arr[0], omega1_arr[0], omega2_arr[0], 0.0f), omega);
  res = _mm_hadd_ps(res, res);
  res = _mm_hadd_ps(res, res);

  float __attribute__((aligned(16))) res_arr[4];
  _mm_store_ps(res_arr, res);
  return res_arr[0];
}

void weno_minus(const float *restrict const a, const float *restrict const b, const float *restrict const c,
                          const float *restrict const d, const float *restrict const e, float *restrict const out,
                          const int NENTRIES)
{
  for (int i = 0; i < NENTRIES; ++i)

    out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
}

void weno_minus_reference(const float *restrict const a, const float *restrict const b, const float *restrict const c,
                          const float *restrict const d, const float *restrict const e, float *restrict const out,
                          const int NENTRIES)
{



  for (int i = 0; i < NENTRIES; ++i)
    out[i] = weno_minus_core_reference(a[i], b[i], c[i], d[i], e[i]);
}

