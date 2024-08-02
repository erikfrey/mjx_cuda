"""Run GPU benchmarks."""

import time
from typing import Any, Callable, Sequence, Tuple

from absl import app
import jax
import mujoco
from mujoco import mjx
from . import engine_core_smooth


def _measure(fn, *args) -> Tuple[float, float]:
  """Reports jit time and op time for a function."""

  beg = time.perf_counter()
  compiled_fn = fn.lower(*args).compile()
  end = time.perf_counter()
  jit_time = end - beg

  times = []

  for _ in range(5):
    beg = time.perf_counter()
    result = compiled_fn(*args)
    jax.block_until_ready(result)
    end = time.perf_counter()
    run_time = end - beg
    times.append(run_time)

  return jit_time, sum(times) / len(times)


def benchmark(
    m: mjx.Model,
    step_fn: Callable[..., Any],
    nstep: int = 100,
    batch_size: int = 4096,
    unroll_steps: int = 1,
) -> Tuple[float, float, int]:
  """Benchmark a model."""

  @jax.vmap
  def init(key):
    qpos = 0.01 * jax.random.normal(key, shape=(m.nq,))
    d = mjx.make_data(m).replace(qpos=qpos)
    return d

  key = jax.random.split(jax.random.key(0), batch_size)
  d = jax.jit(init)(key)
  jax.block_until_ready(d)

  @jax.jit
  def unroll(d):
    def fn(d, _):
      d = d.replace(qpos=d.qpos + 0.001)  # ensure real work - avoid caching
      return step_fn(m, d), None

    return jax.lax.scan(fn, d, None, length=nstep, unroll=unroll_steps)

  jit_time, run_time = _measure(unroll, d)
  steps = nstep * batch_size

  return jit_time, run_time, steps


def _main(_: Sequence[str]):
  """Runs testpeed function."""
  m = mujoco.MjModel.from_xml_path(  # pylint:disable=disallowed-name
      'runfile:google3/third_party/mujoco/model/humanoid/humanoid.xml'
  )
  mx = mjx.put_model(m)

  nstep, batch_size, unroll = 100, 4096, 10

  for name, op in (
      ('CUDA', engine_core_smooth.kinematics),
      ('Pure JAX', jax.vmap(mjx.kinematics, in_axes=(None, 0))),
  ):
    jit_time, run_time, steps = benchmark(mx, op, nstep, batch_size, unroll)

    print(f"""
  Summary for {name} rollouts ({steps} steps, batch_size={batch_size} unroll={unroll})

  Total JIT time: {jit_time:.2f} s
  Total simulation time: {run_time:.2f} s
  Total steps per second: { steps / run_time:.0f}
  Total realtime factor: { steps * m.opt.timestep / run_time:.2f} x
  Total time per step: { 1e6 * run_time / steps:.2f} µs""")


def main():
  app.run(_main)


if __name__ == '__main__':
  main()
