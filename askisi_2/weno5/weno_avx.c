#include <x86intrin.h>
#include "weno_avx.h"

#ifndef WENOEPS
#define WENOEPS 1.e-6
#endif

static inline float weno_minus_core_old(const float a, const float b, const float c, const float d, const float e)
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

static inline float weno_minus_core(const float* restrict const a, const float* restrict const b, const float* restrict const c,
const float* restrict const d, const float* restrict const e, float* restrict const out, int i)
{
	__m512 va = _mm512_load_ps((void*)(a+i));
	__m512 vb = _mm512_load_ps((void*)(b+i));
	__m512 vc = _mm512_load_ps((void*)(c+i));
	__m512 vd = _mm512_load_ps((void*)(d+i));
	__m512 ve = _mm512_load_ps((void*)(e+i));

    __m512 _4_div_3 = _mm512_set1_ps(4.f/3.f);
    __m512 _19_div_3 = _mm512_set1_ps(19.f/3.f);
    __m512 _11_div_3 = _mm512_set1_ps(11.f/3.f);
    __m512 _25_div_3 = _mm512_set1_ps(25.f/3.f);
    __m512 _31_div_3 = _mm512_set1_ps(31.f/3.f);
    __m512 _10_div_3 = _mm512_set1_ps(10.f/3.f);
    __m512 _13_div_3 = _mm512_set1_ps(13.f/3.f);
    __m512 _5_div_3 = _mm512_set1_ps(5.f/3.f);
    __m512 _1_f =_mm512_set1_ps(1.0f);
    __m512 _1_div_3 = _mm512_set1_ps(1.f/3.f);
    __m512 _5_div_6 = _mm512_set1_ps(5.f/6.f);
    __m512 _1_div_6 = _mm512_set1_ps(1.f/6.f);
	
	//--------------------------------------------------------------------------------------------------

	__m512 is0_temp0 = _mm512_mul_ps(va, 
							_mm512_add_ps(
								_mm512_sub_ps(
									_mm512_mul_ps(va, _4_div_3),
									_mm512_mul_ps(vb, _19_div_3)
								),
								_mm512_mul_ps(vc, _11_div_3)
							)
						);

	__m512 is0_temp1 = _mm512_mul_ps(vb, 
		_mm512_sub_ps(
			_mm512_mul_ps(vb, _25_div_3),
			_mm512_mul_ps(vc, _31_div_3)
		)
	);

	__m512 is0_temp2 = _mm512_mul_ps(_mm512_mul_ps(vc, vc), _10_div_3);

	__m512 is0 = _mm512_add_ps(_mm512_add_ps(is0_temp0, is0_temp1), is0_temp2);

	//--------------------------------------------------------------------------------------------------

    __m512 is1_temp0 = _mm512_mul_ps(vb,
                            _mm512_add_ps( 
                                _mm512_sub_ps(
                                    _mm512_mul_ps(va, _4_div_3),
                                    _mm512_mul_ps(vc, _13_div_3)
                                ),
                            _mm512_mul_ps(vd, _5_div_3)
                            )
                        );
    
    __m512 is1_temp1 = _mm512_mul_ps(vc,
                            _mm512_sub_ps(
                                _mm512_mul_ps(vc, _13_div_3),
                                _mm512_mul_ps(vd, _13_div_3)
                            )
                        );
    
    __m512 is1_temp2 = _mm512_mul_ps(_mm512_mul_ps(vd,vd), _4_div_3);
    
    __m512 is1 = _mm512_add_ps(_mm512_add_ps(is1_temp0,is1_temp1), is1_temp2);

	//--------------------------------------------------------------------------------------------------

	__m512 is2_temp0 = _mm512_mul_ps(vc, 
		_mm512_add_ps(
			_mm512_sub_ps(
				_mm512_mul_ps(vc, _10_div_3),
				_mm512_mul_ps(vd, _31_div_3)
			),
			_mm512_mul_ps(ve, _11_div_3)
		)
	);

	__m512 is2_temp1 = _mm512_mul_ps(vd, 
		_mm512_sub_ps(
			_mm512_mul_ps(vd, _25_div_3),
			_mm512_mul_ps(ve, _19_div_3)
		)
	);

	__m512 is2_temp2 = _mm512_mul_ps(_mm512_mul_ps(ve, ve), _4_div_3);

	__m512 is2 = _mm512_add_ps(_mm512_add_ps(is2_temp0, is2_temp1), is2_temp2);

    /*        
    const float is0 = a*(a*(float)(4./3.)  - b*(float)(19./3.)  + c*(float)(11./3.)) + b*(b*(float)(25./3.)  - c*(float)(31./3.)) + c*c*(float)(10./3.);
	const float is1 = b*(b*(float)(4./3.)  - c*(float)(13./3.)  + d*(float)(5./3.))  + c*(c*(float)(13./3.)  - d*(float)(13./3.)) + d*d*(float)(4./3.);
	const float is2 = c*(c*(float)(10./3.) - d*(float)(31./3.)  + e*(float)(11./3.)) + d*(d*(float)(25./3.)  - e*(float)(19./3.)) + e*e*(float)(4./3.);
	*/

    //====================================================================================
    
    __m512 is0plus = _mm512_add_ps(is0, _mm512_set1_ps((float)WENOEPS));
    __m512 is1plus = _mm512_add_ps(is1, _mm512_set1_ps((float)WENOEPS));
    __m512 is2plus = _mm512_add_ps(is2, _mm512_set1_ps((float)WENOEPS));


    /*
	const float is0plus = is0 + (float)WENOEPS;
	const float is1plus = is1 + (float)WENOEPS;
	const float is2plus = is2 + (float)WENOEPS;
    */
    
    //====================================================================================

	__m512 alpha0 = _mm512_mul_ps(_mm512_set1_ps(0.1f), _mm512_div_ps(_1_f, _mm512_mul_ps(is0plus, is0plus)));

    
    __m512 alpha1 = _mm512_mul_ps(_mm512_set1_ps(0.6f),_mm512_div_ps(_1_f, _mm512_mul_ps(is1plus,is1plus)));
                    
	__m512 alpha2 = _mm512_mul_ps(_mm512_set1_ps(0.3f), _mm512_div_ps(_1_f, _mm512_mul_ps(is2plus, is2plus)));

	/*
	const float alpha0 = (float)(0.1)*((float)1/(is0plus*is0plus));
	const float alpha1 = (float)(0.6)*((float)1/(is1plus*is1plus));
	const float alpha2 = (float)(0.3)*((float)1/(is2plus*is2plus));
	*/
	
	__m512 alphasum = _mm512_add_ps(_mm512_add_ps(alpha0, alpha1), alpha2);

	__m512 inv_alpha = _mm512_div_ps(_1_f, alphasum);

	//const float alphasum = alpha0+alpha1+alpha2;
	//const float inv_alpha = ((float)1)/alphasum;

    //====================================================================================

    __m512 omega0 = _mm512_mul_ps(alpha0,inv_alpha);
    __m512 omega1 = _mm512_mul_ps(alpha1,inv_alpha);
    __m512 omega2 = _mm512_sub_ps(_mm512_sub_ps(_mm512_set1_ps(1),omega0),omega1);
    
    /*
	const float omega0 = alpha0 * inv_alpha;
	const float omega1 = alpha1 * inv_alpha;
	const float omega2 = 1-omega0-omega1;
    */

    //====================================================================================

	__m512 omega0_temp = _mm512_mul_ps(omega0,
		_mm512_add_ps(
			_mm512_sub_ps(_mm512_mul_ps(va, _1_div_3), _mm512_mul_ps(vb, _mm512_set1_ps(7.0f/6.0f))),
			_mm512_mul_ps(vc, _mm512_set1_ps(11.0f/6.0f))
		)
	);

    __m512 omega1_temp = _mm512_mul_ps(omega1,
                            _mm512_add_ps(
                                _mm512_sub_ps(
                                    _mm512_mul_ps(vc, _5_div_6),
                                    _mm512_mul_ps(vb, _1_div_6)
                                ),
                            _mm512_mul_ps(vd, _1_div_3)
                            )
                        );


	__m512 omega2_temp = _mm512_mul_ps(omega2,
		_mm512_sub_ps(
			_mm512_add_ps(_mm512_mul_ps(vc, _1_div_3), _mm512_mul_ps(vd, _5_div_6)),
			_mm512_mul_ps(ve, _1_div_6)
		)
	);

	_mm512_store_ps((void*)(out+i), _mm512_add_ps(_mm512_add_ps(omega0_temp, omega1_temp), omega2_temp));
}

void weno_minus_avx(
	const float* restrict const a, const float* restrict const b, const float* restrict const c,
	const float* restrict const d, const float* restrict const e, float* restrict const out, const int NENTRIES)
{
	for (int i = 0; i < NENTRIES; i += 16)
		weno_minus_core(a, b, c, d, e, out, i);
}