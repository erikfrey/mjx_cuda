#ifndef MUJOCO_MJX_GOOGLE_CUDA_ENGINE_UTIL_BLAS_CU_H_
#define MUJOCO_MJX_GOOGLE_CUDA_ENGINE_UTIL_BLAS_CU_H_

namespace mujoco::mjx::cuda {

inline __device__ void zero(float* x, int n) {
  memset(x, 0, n * sizeof(float));
}

inline __device__ void copy(float* dst, const float* src, int n) {
  memcpy(dst, src, n * sizeof(float));
}

inline __device__ void normalize(float* x, int n) {
  float norm = 0.0f;
  for (int i = 0; i < n; ++i) {
    norm += x[i] * x[i];
  }
  norm = sqrt(norm);
  for (int i = 0; i < n; ++i) {
    x[i] /= norm;
  }
}

// multiply 3-by-3 matrix by vector
inline __device__ void mulMatVec3(float res[3], const float mat[9],
                                  const float vec[3]) {
  res[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
  res[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
  res[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

inline __device__ void addTo(float* res, const float* vec, int n) {
  for (int i = 0; i < n; ++i) {
    res[i] += vec[i];
  }
}

inline __device__ void addToScl(float* res, const float* vec, float scl,
                                int n) {
  for (int i = 0; i < n; ++i) {
    res[i] += vec[i] * scl;
  }
}

// res = vec1 - vec2
inline __device__ void sub(float* res, const float* vec1, const float* vec2,
                           int n) {
  for (int i = 0; i < n; ++i) {
    res[i] = vec1[i] - vec2[i];
  }
}

}  // namespace mujoco::mjx::cuda

#endif  // MUJOCO_MJX_GOOGLE_CUDA_ENGINE_UTIL_BLAS_CU_H_
