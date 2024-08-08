#ifndef MUJOCO_MJX_GOOGLE_CUDA_ENGINE_UTIL_SPATIAL_CU_H_
#define MUJOCO_MJX_GOOGLE_CUDA_ENGINE_UTIL_SPATIAL_CU_H_

namespace mujoco::mjx::cuda {

// multiply quaternions
inline __device__ void mulQuat(float res[4], const float qa[4],
                               const float qb[4]) {
  const float tmp[4] = {
    qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3],
    qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2],
    qa[0]*qb[2] - qa[1]*qb[3] + qa[2]*qb[0] + qa[3]*qb[1],
    qa[0]*qb[3] + qa[1]*qb[2] - qa[2]*qb[1] + qa[3]*qb[0]
  };
  res[0] = tmp[0];
  res[1] = tmp[1];
  res[2] = tmp[2];
  res[3] = tmp[3];
}

// rotate vector by quaternion
inline __device__ void rotVecQuat(float res[3], const float vec[3],
                                  const float quat[4]) {
  // tmp = q_w * v + cross(q_xyz, v)
  const float tmp[3] = {
    quat[0] * vec[0] + quat[2] * vec[2] - quat[3] * vec[1],
    quat[0] * vec[1] + quat[3] * vec[0] - quat[1] * vec[2],
    quat[0] * vec[2] + quat[1] * vec[1] - quat[2] * vec[0]
  };

  // res = v + 2 * cross(q_xyz, t)
  res[0] = vec[0] + 2 * (quat[2] * tmp[2] - quat[3] * tmp[1]);
  res[1] = vec[1] + 2 * (quat[3] * tmp[0] - quat[1] * tmp[2]);
  res[2] = vec[2] + 2 * (quat[1] * tmp[1] - quat[2] * tmp[0]);
}

// convert axisAngle to quaternion
inline __device__ void axisAngle2Quat(float res[4], const float axis[3],
                                      float angle) {
  const float s = sin(angle * 0.5);
  res[0] = cos(angle * 0.5);
  res[1] = axis[0] * s;
  res[2] = axis[1] * s;
  res[3] = axis[2] * s;
}

// convert quaternion to 3D rotation matrix
inline __device__ void quat2Mat(float res[9], const float quat[4]) {
  const float q00 = quat[0] * quat[0];
  const float q01 = quat[0] * quat[1];
  const float q02 = quat[0] * quat[2];
  const float q03 = quat[0] * quat[3];
  const float q11 = quat[1] * quat[1];
  const float q12 = quat[1] * quat[2];
  const float q13 = quat[1] * quat[3];
  const float q22 = quat[2] * quat[2];
  const float q23 = quat[2] * quat[3];
  const float q33 = quat[3] * quat[3];

  res[0] = q00 + q11 - q22 - q33;
  res[4] = q00 - q11 + q22 - q33;
  res[8] = q00 - q11 - q22 + q33;

  res[1] = 2 * (q12 - q03);
  res[2] = 2 * (q13 + q02);
  res[3] = 2 * (q12 + q03);
  res[5] = 2 * (q23 - q01);
  res[6] = 2 * (q13 - q02);
  res[7] = 2 * (q23 + q01);
}

}  // namespace mujoco::mjx::cuda

#endif  // MUJOCO_MJX_GOOGLE_CUDA_ENGINE_UTIL_SPATIAL_CU_H_
