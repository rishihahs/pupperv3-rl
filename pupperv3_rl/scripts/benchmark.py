import os
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += (
    ' --xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)
os.environ['XLA_FLAGS'] = xla_flags
os.environ['MUJOCO_GL'] = 'egl'

import time
import jax
import argparse

from brax import envs
from pupperv3_mjx import environment

from pupperv3_rl.config.loader import load_config, prepare_model, setup_environment


def benchmark(batch_size, nstep, visualization_env):
    from mujoco.mjx._src import forward
    from mujoco.mjx._src import io

    unroll_steps = 1
    
    def _measure(fn, *args):
      """Reports jit time and op time for a function."""
    
      beg = time.perf_counter()
      compiled_fn = fn.lower(*args).compile()
      end = time.perf_counter()
      jit_time = end - beg
    
      beg = time.perf_counter()
      result = compiled_fn(*args)
      jax.block_until_ready(result)
      end = time.perf_counter()
      run_time = end - beg
    
      return jit_time, run_time
    
    @jax.pmap
    def init(key):
      key = jax.random.split(key, batch_size // jax.device_count())
    
      @jax.vmap
      def random_init(key):
        return visualization_env.reset(key)
    
      return random_init(key)
    
    key = jax.random.split(jax.random.key(0), jax.device_count())
    d = init(key)
    jax.block_until_ready(d)

    @jax.pmap
    def unroll(d):
      @jax.vmap
      def step(carry, _):
        d, rng = carry
        rng, key = jax.random.split(rng)
        ctrl = jax.random.uniform(key, shape=(visualization_env.sys.nu,))
        d = visualization_env.step(d, ctrl)
        return (d, rng), None
    
      rng = jax.random.split(jax.random.PRNGKey(0), batch_size // jax.device_count())
      d, _ = jax.lax.scan(step, (d, rng), None, length=nstep, unroll=unroll_steps)
    
      return d
    
    
    jit_time, run_time = _measure(unroll, d)
    steps = nstep * batch_size
    
    print(jit_time, steps / run_time)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config if args.config else None)

    model_path = prepare_model(config)
    config.simulation.model_path = model_path
    env = setup_environment(config)

    benchmark(config.training.ppo.num_envs, config.training.ppo.episode_length, env)


if __name__ == "__main__":
    main()
