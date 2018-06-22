#include "../include/shared_math.h"

#include <stdlib.h> // srand, rand

namespace pbf {
namespace math_impl_ {
float GenRandom(FloatingPointTag, float lo, float hi) {
  float result =
      static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (hi - lo)) +
      lo;
  return result;
}

int GenRandom(IntegralTag, int lo, int hi) {
  return ((rand() % (hi - lo)) + lo);
}

} // namespace math_impl_
} // namespace pbf
