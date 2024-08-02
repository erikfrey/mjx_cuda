#include "engine_core_smooth.h"

#include "mujoco/include/mujoco.h"
#include "engine_util_blas.cu.h"
#include "engine_util_spatial.cu.h"

namespace mujoco::mjx::cuda {

namespace ffi = xla::ffi;

namespace {

__global__ void kinematicsKernel(
    unsigned int n, unsigned int nq, unsigned int njnt, unsigned int nbody,
    unsigned int ngeom, unsigned int nsite, unsigned int nmocap,
    const float* qpos0, const int* body_jntadr, const int* body_jntnum,
    const int* body_parentid, const int* body_mocapid, const float* body_pos,
    const float* body_quat, const float* body_ipos, const float* body_iquat,
    const int* jnt_type, const int* jnt_qposadr, const float* jnt_axis,
    const float* jnt_pos, const float* geom_pos, const float* geom_quat,
    const float* site_pos, const float* site_quat, const float* qpos,
    const float* mocap_pos, const float* mocap_quat, float* xanchor,
    float* xaxis, float* xmat, float* xpos, float* xquat, float* xipos,
    float* ximat, float* geom_xpos, float* geom_xmat, float* site_xpos,
    float* site_xmat) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= n) {
    return;
  }

  // batch index into mjx.Data
  qpos = qpos + tid * nq;
  mocap_pos = mocap_pos + tid * nmocap * 3;
  mocap_quat = mocap_quat + tid * nmocap * 4;

  xanchor = xanchor + tid * njnt * 3;
  xaxis = xaxis + tid * njnt * 3;
  xmat = xmat + tid * nbody * 9;
  xpos = xpos + tid * nbody * 3;
  xquat = xquat + tid * nbody * 4;
  xipos = xipos + tid * nbody * 3;
  ximat = ximat + tid * nbody * 9;
  geom_xpos = geom_xpos + tid * ngeom * 3;
  geom_xmat = geom_xmat + tid * ngeom * 9;
  site_xpos = site_xpos + tid * nsite * 3;
  site_xmat = site_xmat + tid * nsite * 9;

  // set world position and orientation
  zero(xpos, 3);
  zero(xquat, 4);
  zero(xipos, nbody * 3);
  zero(xmat, nbody * 9);
  zero(ximat, nbody * 9);

  xquat[0] = 1.0f;
  xmat[0] = xmat[4] = xmat[8] = 1.0f;
  ximat[0] = ximat[4] = ximat[8] = 1.0f;

  // TODO(erikfrey): bodies in the same tree depth can be processed in parallel
  // tried threading over bodies and it was actually significantly slower -
  // need to investigate further

  // compute global cartesian positions and orientations of all bodies
  for (int i = 1; i < nbody; i++) {
    float lxpos[3], lxquat[4];
    int jntadr = body_jntadr[i];
    int jntnum = body_jntnum[i];

    // free joint
    if (jntnum == 1 && jnt_type[jntadr] == mjJNT_FREE) {
      // get qpos address
      int qadr = jnt_qposadr[jntadr];

      // copy pos and quat from qpos
      copy(lxpos, qpos + qadr, 3);
      copy(lxquat, qpos + qadr + 3, 4);
      normalize(lxquat, 4);

      // assign xanchor and xaxis
      copy(xanchor + 3 * jntadr, lxpos, 3);
      copy(xaxis + 3 * jntadr, jnt_axis + 3 * jntadr, 3);
    } else {  // regular or no joint
      int pid = body_parentid[i];

      // get body pos and quat: from model or mocap
      const float *bodypos, *bodyquat;
      float quat[4];
      if (body_mocapid[i] >= 0) {
        bodypos = mocap_pos + 3 * body_mocapid[i];
        copy(quat, mocap_quat + 4 * body_mocapid[i], 4);
        normalize(quat, 4);
        bodyquat = quat;
      } else {
        bodypos = body_pos + 3 * i;
        bodyquat = body_quat + 4 * i;
      }

      // apply fixed translation and rotation relative to parent
      if (pid) {
        mulMatVec3(lxpos, xmat + 9 * pid, bodypos);
        addTo(lxpos, xpos + 3 * pid, 3);
        mulQuat(lxquat, xquat + 4 * pid, bodyquat);
      } else {
        // parent is the world
        copy(lxpos, bodypos, 3);
        copy(lxquat, bodyquat, 4);
      }

      // accumulate joints, compute xpos and xquat for this body
      float lxanchor[3], lxaxis[3];
      for (int j = 0; j < jntnum; j++) {
        // get joint id, qpos address, joint type
        int jid = jntadr + j;
        int qadr = jnt_qposadr[jid];
        int jtype = jnt_type[jid];

        // compute axis in global frame; ball jnt_axis is (0,0,1), set by
        // compiler
        rotVecQuat(lxaxis, jnt_axis + 3 * jid, lxquat);

        // compute anchor in global frame
        rotVecQuat(lxanchor, jnt_pos + 3 * jid, lxquat);
        addTo(lxanchor, lxpos, 3);

        // apply joint transformation
        switch (jtype) {
          case mjJNT_SLIDE:
            addToScl(lxpos, lxaxis, qpos[qadr] - qpos0[qadr], 3);
            break;

          case mjJNT_BALL:
          case mjJNT_HINGE: {
            // compute local quaternion rotation
            float qloc[4];
            if (jtype == mjJNT_BALL) {
              copy(qloc, qpos + qadr, 4);
              normalize(qloc, 4);
            } else {
              axisAngle2Quat(qloc, jnt_axis + 3 * jid,
                             qpos[qadr] - qpos0[qadr]);
            }

            // apply rotation
            mulQuat(lxquat, lxquat, qloc);

            // correct for off-center rotation
            float vec[3];
            rotVecQuat(vec, jnt_pos + 3 * jid, lxquat);
            sub(lxpos, lxanchor, vec, 3);
          } break;

          default:
            // TODO: whatever cuda error semantics are
            // mjERROR("unknown joint type %d", jtype);  // SHOULD NOT OCCUR
            break;
        }

        // assign xanchor and xaxis
        copy(xanchor + 3 * jid, lxanchor, 3);
        copy(xaxis + 3 * jid, lxaxis, 3);
      }
    }

    // assign xquat and xpos, construct xmat
    normalize(lxquat, 4);
    copy(xquat + 4 * i, lxquat, 4);
    copy(xpos + 3 * i, lxpos, 3);
    quat2Mat(xmat + 9 * i, lxquat);
  }
}

}  // namespace

ffi::Error LaunchKinematicsKernel(
    cudaStream_t stream, ffi::Buffer<ffi::DataType::F32> qpos0,
    ffi::Buffer<ffi::DataType::S32> body_jntadr,
    ffi::Buffer<ffi::DataType::S32> body_jntnum,
    ffi::Buffer<ffi::DataType::S32> body_parentid,
    ffi::Buffer<ffi::DataType::S32> body_mocapid,
    ffi::Buffer<ffi::DataType::F32> body_pos,
    ffi::Buffer<ffi::DataType::F32> body_quat,
    ffi::Buffer<ffi::DataType::F32> body_ipos,
    ffi::Buffer<ffi::DataType::F32> body_iquat,
    ffi::Buffer<ffi::DataType::S32> jnt_type,
    ffi::Buffer<ffi::DataType::S32> jnt_qposadr,
    ffi::Buffer<ffi::DataType::F32> jnt_axis,
    ffi::Buffer<ffi::DataType::F32> jnt_pos,
    ffi::Buffer<ffi::DataType::F32> geom_pos,
    ffi::Buffer<ffi::DataType::F32> geom_quat,
    ffi::Buffer<ffi::DataType::F32> site_pos,
    ffi::Buffer<ffi::DataType::F32> site_quat,
    ffi::Buffer<ffi::DataType::F32> qpos,
    ffi::Buffer<ffi::DataType::F32> mocap_pos,
    ffi::Buffer<ffi::DataType::F32> mocap_quat,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> xanchor,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> xaxis,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> xmat,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> xpos,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> xquat,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> xipos,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> ximat,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> geom_xpos,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> geom_xmat,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> site_xpos,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> site_xmat) {
  auto nq = qpos.dimensions().back();
  auto n = qpos.element_count() / nq;
  auto njnt = jnt_type.dimensions().back();
  auto nbody = body_jntadr.dimensions().back();
  auto ngeom = geom_pos.dimensions()[-2];
  auto nsite = site_pos.dimensions()[-2];
  auto nmocap = mocap_pos.dimensions()[-2];

  const int blockSize = 256;
  const int gridSize = (n + blockSize - 1) / blockSize;

  kinematicsKernel<<<gridSize, blockSize, /*shared_mem=*/0, stream>>>(
      n, nq, njnt, nbody, ngeom, nsite, nmocap, qpos0.typed_data(),
      body_jntadr.typed_data(), body_jntnum.typed_data(),
      body_parentid.typed_data(), body_mocapid.typed_data(),
      body_pos.typed_data(), body_quat.typed_data(), body_ipos.typed_data(),
      body_iquat.typed_data(), jnt_type.typed_data(), jnt_qposadr.typed_data(),
      jnt_axis.typed_data(), jnt_pos.typed_data(), geom_pos.typed_data(),
      geom_quat.typed_data(), site_pos.typed_data(), site_quat.typed_data(),
      qpos.typed_data(), mocap_pos.typed_data(), mocap_quat.typed_data(),
      xanchor->typed_data(), xaxis->typed_data(), xmat->typed_data(),
      xpos->typed_data(), xquat->typed_data(), xipos->typed_data(),
      ximat->typed_data(), geom_xpos->typed_data(), geom_xmat->typed_data(),
      site_xpos->typed_data(), site_xmat->typed_data());

  // Check for launch time errors. Note that this function may also
  // return error codes from previous, asynchronous launches. This
  // means that an error status returned here could have been caused
  // by a different kernel previously launched by XLA.
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return ffi::Error(
        XLA_FFI_Error_Code_INTERNAL,
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

}  // namespace mujoco::mjx::cuda
