#ifndef MUJOCO_MJX_GOOGLE_CUDA_ENGINE_CORE_SMOOTH_H_
#define MUJOCO_MJX_GOOGLE_CUDA_ENGINE_CORE_SMOOTH_H_

#include "cuda/include/driver_types.h"
#include "xla/ffi/api/ffi.h"

namespace mujoco::mjx::cuda {

xla::ffi::Error LaunchKinematicsKernel(cudaStream_t stream,
        xla::ffi::Buffer<xla::ffi::DataType::F32> qpos0,
        xla::ffi::Buffer<xla::ffi::DataType::S32> body_jntadr,
        xla::ffi::Buffer<xla::ffi::DataType::S32> body_jntnum,
        xla::ffi::Buffer<xla::ffi::DataType::S32> body_parentid,
        xla::ffi::Buffer<xla::ffi::DataType::S32> body_mocapid,
        xla::ffi::Buffer<xla::ffi::DataType::F32> body_pos,
        xla::ffi::Buffer<xla::ffi::DataType::F32> body_quat,
        xla::ffi::Buffer<xla::ffi::DataType::F32> body_ipos,
        xla::ffi::Buffer<xla::ffi::DataType::F32> body_iquat,
        xla::ffi::Buffer<xla::ffi::DataType::S32> jnt_type,
        xla::ffi::Buffer<xla::ffi::DataType::S32> jnt_qposadr,
        xla::ffi::Buffer<xla::ffi::DataType::F32> jnt_axis,
        xla::ffi::Buffer<xla::ffi::DataType::F32> jnt_pos,
        xla::ffi::Buffer<xla::ffi::DataType::F32> geom_pos,
        xla::ffi::Buffer<xla::ffi::DataType::F32> geom_quat,
        xla::ffi::Buffer<xla::ffi::DataType::F32> site_pos,
        xla::ffi::Buffer<xla::ffi::DataType::F32> site_quat,
        xla::ffi::Buffer<xla::ffi::DataType::F32> qpos,
        xla::ffi::Buffer<xla::ffi::DataType::F32> mocap_pos,
        xla::ffi::Buffer<xla::ffi::DataType::F32> mocap_quat,
        xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> xanchor,
        xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> xaxis,
        xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> xmat,
        xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> xpos,
        xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> xquat,
        xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> xipos,
        xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> ximat,
        xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> geom_xpos,
        xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> geom_xmat,
        xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> site_xpos,
        xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> site_xmat);

}  // namespace mujoco::mjx::cuda

#endif  // MUJOCO_MJX_GOOGLE_CUDA_ENGINE_CORE_SMOOTH_H_
