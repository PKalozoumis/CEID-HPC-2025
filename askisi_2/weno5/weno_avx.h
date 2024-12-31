#pragma once
#include <x86intrin.h>

static inline float weno_minus_core(const float a, const float b, const float c, const float d, const float e)
{

  //[is2, is1, is0]

  /*
  __m128 abc = _mm_set_ps(a, b, c, 0.0f);
  __m128 bcd = _mm_set_ps(b, c, d, 0.0f);
  __m128 cde = _mm_set_ps(c, d, e, 0.0f);*/

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

  __m512 temp0 = _mm512_mul_ps(vec1, mul1);

  __m128 temp00 = _mm512_extractf32x4_ps(temp0, 0);
  __m128 temp01 = _mm512_extractf32x4_ps(temp0, 1);
  __m128 temp02 = _mm512_extractf32x4_ps(temp0, 2);
  __m128 temp03 = _mm512_extractf32x4_ps(temp0, 3);

  __m128 sum0 = _mm_add_ps(temp00, _mm_add_ps(_mm_add_ps(temp01, temp02), temp03));

  //--------------------------------------------------------------------

  __m256 temp1 = _mm256_mul_ps(vec2, mul2);

  __m128 temp10 = _mm256_extractf32x4_ps(temp1, 0);
  __m128 temp11 = _mm256_extractf32x4_ps(temp1, 1);
  __m128 temp12 = _mm256_extractf32x4_ps(temp1, 2);

  __m128 sum1 = _mm_add_ps(temp10, _mm_add_ps(temp11, temp12));

  //--------------------------------------------------------------------

  __m128 is = _mm_add_ps(sum0, sum1);
  /*
  const float is0 = a * a * (float)(4. / 3.) + a*-b * (float)(19. / 3.) + a*  c * (float)(11. / 3.) 	+ b * b * (float)(25. / 3.) 					- b* c * (float)(31. / 3.) + c * c * (float)(10. / 3.);
  const float is1 = b * b * (float)(4. / 3.) + b*-c * (float)(13. / 3.) + b* d * (float)(5. / 3.) 		+ c * c * (float)(13. / 3.) 					-c* d * (float)(13. / 3.) + d * d * (float)(4. / 3.);
  const float is2 = c * c * (float)(10. / 3.) + c*-d * (float)(31. / 3.) + c* e * (float)(11. / 3.) 	+ d *d * (float)(25. / 3.) 				- d* e * (float)(19. / 3.) + e * e * (float)(4. / 3.);
*/

  __m128 wenoeps = _mm_set1_ps((float)WENOEPS);
  __m128 is_plus = _mm_add_ps(is, wenoeps);
  is_plus = _mm_mul_ps(is_plus, is_plus);

  __m128 alpha = _mm_div_ps(_mm_set_ps(0.1f, 0.6f, 0.3f, 0.0f), is_plus);

  /*
	[a3,a2,a1,a0]
	[b3,b2,b1,b0]

	[b3+b2, b1+b0, a3+a2, a1+a0]

  	[1, 6, 3, 0]
		|
		V
	[7, 3, 7, 3]
		|
		V
	[10, 10, 10, 10]
*/

  __m128 alphasum = _mm_hadd_ps(alpha, alpha);
  __m128 alphasum = _mm_hadd_ps(alphasum, alphasum);

  __m128 omega = _mm_div_ps(alpha, alphasum);
	

  //[(float)WENOEPS, (float)WENOEPS, (float)WENOEPS]

  /*
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
const float omega2 = 1-omega0-omega1; // 1 - alpha0 * inv_alpha - alpha1 * inv_alpha = 1 - inv_alpha(alpha0 + alpha1) = 1 - (alpha0+alpha1)/(alpha0+alpha1+alpha2) = alpha2/(alpha0 + alpha1 + alpha2)

  return omega0*((float)(1.0/3.)*a - (float)(7./6.)*b + (float)(11./6.)*c) +
          omega1*(-(float)(1./6.)*b + (float)(5./6.)*c + (float)(1./3.)*d) +
          omega2*((float)(1./3.)*c  + (float)(5./6.)*d - (float)(1./6.)*e);*/
}

void weno_minus_reference(const float *restrict const a, const float *restrict const b, const float *restrict const c,
                          const float *restrict const d, const float *restrict const e, float *restrict const out,
                          const int NENTRIES)
{
  for (int i = 0; i < NENTRIES; ++i)

    out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
}
