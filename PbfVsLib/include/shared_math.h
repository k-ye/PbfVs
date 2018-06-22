#ifndef shared_math_h
#define shared_math_h

#include <type_traits>

#include "glm_headers.h"
#include "typedefs.h"

// value borrowed from glm::pi
#define PI_FLT (float)3.14159265358979323846264338327950288
#define kFloatEpsilon (float)1e-6

namespace pbf {
template <typename T> T Interpolate(T val, T vmin, T vmax, T rmin, T rmax) {
  return (val - vmin) * (rmax - rmin) / (vmax - vmin) + rmin;
}

namespace math_impl_ {

struct FloatingPointTag {};
struct IntegralTag {};
struct UnknownTag {};

template <typename T> struct Tagger {
  typedef
      typename IfElse<std::is_floating_point<T>::value, FloatingPointTag,
                      // Cannot use std::enable_if here, because type system is
                      // a compile time thing. That is, all types must be known
                      // at compile type, there is no short circuit. This
                      // results in the definition of UnknownTag and is not so
                      // fancy.
                      typename IfElse<std::is_integral<T>::value, IntegralTag,
                                      UnknownTag>::type>::type tag;
};

float GenRandom(FloatingPointTag, float lo, float hi);
int GenRandom(IntegralTag, int lo, int hi);
} // namespace math_impl_

template <typename T> T GenRandom(T lo, T hi) {
  return math_impl_::GenRandom(typename math_impl_::Tagger<T>::tag{}, lo, hi);
}
} // namespace pbf

#endif // shared_math_h
