# based on: https://colab.research.google.com/github/google/jax-cfd/blob/main/notebooks/spectral_forced_turbulence.ipynb

import argparse
import os
import time
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import xarray
import numpy as np

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral

# 64-bit precision
jax.config.update("jax_enable_x64", True)
print("start time: ", datetime.now())

# one example configuration
N = 1024
VISCOSITY = 1e-3
MAX_VELOCITY = 7.0
DOMAIN = ((0, 2 * jnp.pi), (0, 2 * jnp.pi))
FORCING_WAVENUMBER = 4
FORCING_SCALE = 1.0
FINAL_TIME = 50.0
DRAG = 0.1

def get_solver_components():
    """
    Sets up the grid, time step, and equation
    """
    grid = grids.Grid((N, N), domain=DOMAIN)
    
    dt = cfd.equations.stable_time_step(MAX_VELOCITY, 0.5, VISCOSITY, grid)
    
    def kolmogorov_forcing_factory(grid):
        def forcing(v):
            u_var, v_var = v
            _, y = grid.mesh(offset=u_var.offset)
            
            u_force_data = FORCING_SCALE * jnp.sin(FORCING_WAVENUMBER * y)
            v_force_data = jnp.zeros_like(v_var.data)
            
            fx = grids.GridArray(u_force_data, u_var.offset, grid)
            fy = grids.GridArray(v_force_data, v_var.offset, grid)
            return (fx, fy)
        return forcing

    eqn = spectral.equations.NavierStokes2D(
        viscosity=VISCOSITY, 
        grid=grid, 
        drag=DRAG, 
        smooth=True,
        forcing_fn=kolmogorov_forcing_factory
    )

    step_fn = spectral.time_stepping.crank_nicolson_rk4(eqn, dt)
    
    return grid, step_fn, dt

def run_single_simulation(key, grid, step_fn, dt):
    """
    runs a single simulation trajectory for a specific random key.
    """
    integer_steps = int(FINAL_TIME)
    steps_per_sec = int(round(1.0 / dt))
    
    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, steps_per_sec), integer_steps
    )

    # initial conditions
    v0 = cfd.initial_conditions.filtered_velocity_field(key, grid, MAX_VELOCITY, 8)
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)

    # run simulation!
    _, vorticity_hat_trajectory = trajectory_fn(vorticity_hat0)

    # concatenate initial state with trajectory
    all_vorticity_hat = jnp.concatenate(
        [jnp.expand_dims(vorticity_hat0, 0), vorticity_hat_trajectory], axis=0
    )

    # Transform to real space
    all_vorticity = jnp.fft.irfftn(all_vorticity_hat, axes=(1, 2))
    
    return all_vorticity

def save_single_sample(data_array, sample_id, dt, output_dir):
    """ saves a single simulation result to a NetCDF file """
    integer_steps = int(FINAL_TIME)
    spatial_coord = np.arange(N) * 2 * np.pi / N
    time_coord = np.arange(integer_steps + 1)

    ds = xarray.Dataset(
        data_vars={
            "vorticity": (("time", "x", "y"), data_array)
        },
        coords={
            "time": time_coord,
            "x": spatial_coord,
            "y": spatial_coord
        },
        attrs={
            "description": f"2D Navier-Stokes Turbulence - Sample {sample_id}",
            "sample_id": sample_id,
            "viscosity": VISCOSITY,
            "max_velocity": MAX_VELOCITY,
            "dt": float(dt)
        }
    )
    
    filename = os.path.join(output_dir, f"sample_{sample_id:05d}-3072.nc")
    ds.to_netcdf(filename)
    return filename

def generate_dataset(num_samples, batch_size=10, output_file="cfd_dataset.nc"):
    """
    generates N samples and saves to a NetCDF file
    """
    print(f"Initializing setup for {num_samples} samples...")
    grid, step_fn, dt = get_solver_components()
    
    # JIT compile the simulation function for performance
    print("JIT compiling solver...")
    sim_runner = jax.jit(lambda k: run_single_simulation(k, grid, step_fn, dt))
    
    # Pre-compile with a dummy run
    dummy_key = jax.random.PRNGKey(0)
    _ = sim_runner(dummy_key)
    print("Compilation complete.")

    all_simulations = []
    
    start_time = time.time()
    
    master_key = jax.random.PRNGKey(42)
    keys = jax.random.split(master_key, num_samples)

    print(f"Starting generation of {num_samples} samples...   ({datetime.now()})")
    print(f"Grid: {N}x{N}, Time: {FINAL_TIME}s")

    for i in range(num_samples):
        sim_data = sim_runner(keys[i])
        
        all_simulations.append(np.array(sim_data))
        
        if (i + 1) % batch_size == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (num_samples - (i + 1))
            print(f"Generated {i + 1}/{num_samples} samples. "
                  f"Est. remaining: {remaining/60:.1f} min"
                  f" | {datetime.now()}")

    # stack into a single large array: (samples, time, x, y)
    print("Stacking data...")
    dataset_array = np.stack(all_simulations, axis=0)
    
    # create Xarray Dataset
    print("Creating NetCDF dataset...")
    integer_steps = int(FINAL_TIME)
    spatial_coord = np.arange(N) * 2 * np.pi / N
    time_coord = np.arange(integer_steps + 1)
    sample_coord = np.arange(num_samples)

    ds = xarray.Dataset(
        data_vars={
            "vorticity": (("sample", "time", "x", "y"), dataset_array)
        },
        coords={
            "sample": sample_coord,
            "time": time_coord,
            "x": spatial_coord,
            "y": spatial_coord
        },
        attrs={
            "description": "2D Navier-Stokes Turbulence Dataset",
            "viscosity": VISCOSITY,
            "max_velocity": MAX_VELOCITY,
            "dt": float(dt)
        }
    )

    ds.to_netcdf(output_file)
    print(f"Dataset saved to {output_file} ({os.path.getsize(output_file)/1e9:.2f} GB)")

def visualize_sample(file_path, sample_index=None):
    """
    Loads the dataset and visualizes a specific sample.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found. Generate it first.")
        return

    ds = xarray.open_dataset(file_path)
    num_samples = ds.sizes['sample']

    if sample_index is None:
        sample_index = np.random.randint(0, num_samples)
        print(f"No index specified. Visualizing random sample: {sample_index}")
    
    if sample_index >= num_samples:
        print(f"Error: Index {sample_index} out of bounds (max {num_samples-1})")
        return

    da = ds["vorticity"].isel(sample=sample_index)

    plot_times = [2, 6, 10, 15, 20, 25, 40, 50]
    valid_times = [t for t in plot_times if t in da.time.values]
    
    da_plot = da.sel(time=valid_times)

    print(f"Plotting Sample {sample_index}...")
    
    plot = da_plot.plot.imshow(
        col="time",
        col_wrap=len(valid_times),
        cmap=sns.cm.icefire,
        robust=True,
        size=3,
        aspect=1,
        add_colorbar=True
    )

    for ax in plot.axes.flat:
        ax.set_aspect('equal')
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.suptitle(f"Sample {sample_index} Evolution", y=1.05)
    save_path = f"vis/256-1e3-f6-sample_{sample_index}_vis.png"
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="CFD Dataset Generator & Visualizer")
    parser.add_argument("--generate", action="store_true", help="Generate the dataset")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--visualize", action="store_true", help="Visualize a sample from the dataset")
    parser.add_argument("--index", type=int, default=None, help="Index of sample to visualize (default: random)")
    parser.add_argument("--file", type=str, default="cfd_dataset.nc", help="Filename for the dataset")

    args = parser.parse_args()

    if args.generate:
        generate_dataset(args.samples, output_file=args.file, batch_size=1)
    
    if args.visualize:
        visualize_sample(args.file, args.index)

    if not args.generate and not args.visualize:
        print("Please specify --generate or --visualize. Use --help for options.")

if __name__ == "__main__":
    main()