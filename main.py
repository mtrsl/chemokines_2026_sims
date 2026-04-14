import numpy as np
import argparse
from pathlib import Path


def run(chemotaxis, v_max, chi, alpha, D_ratio, Pe, n_cells, CCL21_added, cell_motility, cell_init, rng, output_dir, rep):
    Lx, Ly = 1600, 1400
    Nx, Ny = 161, 141
    dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)

    D_CCL21 = 100.0
    D_CCL19 = D_ratio * D_CCL21
    d = 1.3e-6
    dt = 0.1
    T_total = 5400
    Nt_total = int(T_total / dt)

    u = 0.084 * Pe

    if CCL21_added:
        inlet_conc = 6.0
    else:
        inlet_conc = 0.0

    sigma = 10
    outlet_conc = 0.0

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)

    c1 = np.zeros((Nx, Ny))

    sqrt_term = np.sqrt(u**2 + 4 * D_CCL21 * d)
    lambda1 = (u + sqrt_term) / (2 * D_CCL21)
    lambda2 = (u - sqrt_term) / (2 * D_CCL21)

    e1_0 = np.exp(lambda1 * 0.0)
    e2_0 = np.exp(lambda2 * 0.0)
    e1_L = np.exp(lambda1 * Ly)
    e2_L = np.exp(lambda2 * Ly)

    den = e2_L * e1_0 - e2_0 * e1_L
    if abs(den) < 1e-16:
        A = 0.0
        B = 0.0
    else:
        A = (inlet_conc * e2_L) / den
        B = (inlet_conc * e1_L) / (e1_L * e2_0 - e2_L * e1_0)

    c2_1d_y = A * np.exp(lambda1 * y) + B * np.exp(lambda2 * y)
    c2 = np.tile(c2_1d_y[None, :], (Nx, 1))
    c2[:, 0] = inlet_conc
    c2[:, -1] = outlet_conc


    def step(c1, c2, source):
        c1_new = c1.copy()
        c2_new = c2.copy()

        c1_c = c1[1:-1, 1:-1]
        c2_c = c2[1:-1, 1:-1]

        u_p = max(0.0, u)
        u_m = min(0.0, u)

        lap1 = (
            (c1[2:, 1:-1] - 2 * c1_c + c1[:-2, 1:-1]) / dx**2 +
            (c1[1:-1, 2:] - 2 * c1_c + c1[1:-1, :-2]) / dy**2
        )
        lap2 = (
            (c2[2:, 1:-1] - 2 * c2_c + c2[:-2, 1:-1]) / dx**2 +
            (c2[1:-1, 2:] - 2 * c2_c + c2[1:-1, :-2]) / dy**2
        )

        c1_y_p = (c1[1:-1, 2:] - c1_c) / dy
        c1_y_m = (c1_c - c1[1:-1, :-2]) / dy
        c2_y_p = (c2[1:-1, 2:] - c2_c) / dy
        c2_y_m = (c2_c - c2[1:-1, :-2]) / dy

        adv_y1 = u_p * c1_y_m + u_m * c1_y_p
        adv_y2 = u_p * c2_y_m + u_m * c2_y_p

        c1_new[1:-1, 1:-1] = c1_c + dt * (
            D_CCL19 * lap1 - adv_y1 - d * c1_c + source[1:-1, 1:-1]
        )
        c2_new[1:-1, 1:-1] = c2_c + dt * (
            D_CCL21 * lap2 - adv_y2 - d * c2_c
        )

        c1_new[:, 0] = 0.0
        c1_new[:, -1] = 0.0
        c1_new[0, :] = c1_new[1, :]
        c1_new[-1, :] = c1_new[-2, :]

        c2_new[0, :] = c2_new[1, :]
        c2_new[-1, :] = c2_new[-2, :]
        c2_new[:, 0] = inlet_conc
        c2_new[:, -1] = outlet_conc

        return c1_new, c2_new


    n_bound = n_unbound = n_cells // 2

    if cell_init == "grid":
        nx = int(np.sqrt(n_cells))
        ny = int(np.ceil(n_cells / nx))

        dx_grid = Lx / nx
        dy_grid = (1200 - 200) / ny

        grid_x = np.linspace(0, Lx - dx_grid, nx) + dx_grid / 2
        grid_y = np.linspace(200, 1200 - dy_grid, ny) + dy_grid / 2

        gx, gy = np.meshgrid(grid_x, grid_y)
        cell_x = gx.flatten()[:n_cells]
        cell_y = gy.flatten()[:n_cells]
    else:
        cell_x = rng.uniform(0.0, Lx, n_cells)
        cell_y = rng.uniform(200.0, 1200.0, n_cells)

    initial_cell_x = np.copy(cell_x)
    initial_cell_y = np.copy(cell_y)

    cell_type = np.zeros(n_cells, dtype=int)
    cell_type[n_unbound:] = 1

    sim_dt = 0.1
    start_t = 0
    end_t = T_total
    frames_time = np.arange(start_t, end_t + 1e-9, 10.0)
    num_frames = len(frames_time)
    frame_indices = np.round(frames_time / sim_dt).astype(int)
    frame_set = set(frame_indices)
    density_x_grid = np.linspace(0, Lx, 100)
    density_matrix = np.zeros((num_frames, len(density_x_grid)))
    times_recorded = []
    cell_x_timepoints = []
    cell_y_timepoints = []
    cell_type_timepoints = []
    frames = []

    ccl19_file_name = "CCL19_all_" + str(rep) + ".txt"
    ccl21_file_name = "CCL21_all_" + str(rep) + ".txt"

    ccl19_file = open(output_dir / ccl19_file_name, "w")
    ccl21_file = open(output_dir / ccl21_file_name, "w")

    x0 = x[0]
    y0 = y[0]
    p_off_dt = 5e-3 * dt
    source_radius_x = int(np.ceil(3 * sigma / dx))
    source_radius_y = int(np.ceil(3 * sigma / dy))

    for n in range(Nt_total):
        if n % 100 == 0:
            print(f"t={n * dt}")

        ix = np.clip(np.searchsorted(x, cell_x), 1, Nx - 2)
        iy = np.clip(np.searchsorted(y, cell_y), 1, Ny - 2)
        c_total_local = c1[ix, iy] + c2[ix, iy]

        bound_mask = cell_type == 1
        unbound_mask = ~bound_mask

        rand_off = rng.random(n_cells)
        cell_type[bound_mask & (rand_off < p_off_dt)] = 0

        p_on_dt = 1.83e-3 * c_total_local * dt
        rand_on = rng.random(n_cells)
        cell_type[unbound_mask & (rand_on < p_on_dt)] = 1

        dcell_x = rng.normal(0, np.sqrt(2 * cell_motility * dt), n_cells)
        dcell_y = rng.normal(0, np.sqrt(2 * cell_motility * dt), n_cells)

        c_total = c1 + c2
        grad_x_grid, grad_y_grid = np.gradient(c_total, dx, dy)

        bound_mask = cell_type == 1
        if np.any(bound_mask):
            fx = (cell_x - x0) / dx
            fy = (cell_y - y0) / dy
            ixg = np.floor(fx).astype(int)
            iyg = np.floor(fy).astype(int)
            wx = fx - ixg
            wy = fy - iyg
            ixg = np.clip(ixg, 0, Nx - 2)
            iyg = np.clip(iyg, 0, Ny - 2)

            g00x = grad_x_grid[ixg, iyg]
            g10x = grad_x_grid[ixg + 1, iyg]
            g01x = grad_x_grid[ixg, iyg + 1]
            g11x = grad_x_grid[ixg + 1, iyg + 1]

            g00y = grad_y_grid[ixg, iyg]
            g10y = grad_y_grid[ixg + 1, iyg]
            g01y = grad_y_grid[ixg, iyg + 1]
            g11y = grad_y_grid[ixg + 1, iyg + 1]

            gx_i = (
                (1 - wx) * (1 - wy) * g00x
                + wx * (1 - wy) * g10x
                + (1 - wx) * wy * g01x
                + wx * wy * g11x
            )
            gy_i = (
                (1 - wx) * (1 - wy) * g00y
                + wx * (1 - wy) * g10y
                + (1 - wx) * wy * g01y
                + wx * wy * g11y
            )

            if chemotaxis == "keller_segel":
                dcell_x[bound_mask] += chi * gx_i[bound_mask] * dt
                dcell_y[bound_mask] += chi * gy_i[bound_mask] * dt
            elif chemotaxis == "saturating":
                g0 = v_max / chi
                denom = 1.0 + (np.sqrt(gx_i**2 + gy_i**2) / g0)
                dcell_x[bound_mask] += (chi * gx_i[bound_mask] / denom[bound_mask]) * dt
                dcell_y[bound_mask] += (chi * gy_i[bound_mask] / denom[bound_mask]) * dt


        proposed_x = cell_x + dcell_x
        proposed_y = cell_y + dcell_y

        in_x_bounds = (proposed_x >= 0) & (proposed_x <= Lx)
        cell_x = np.where(in_x_bounds, proposed_x, cell_x)

        proposed_y = np.clip(proposed_y, 200, 1200)
        cell_y = proposed_y

        cell_source = np.zeros((Nx, Ny))
        for i in range(n_cells):
            ix_c = int(np.floor((cell_x[i] - x0) / dx))
            iy_c = int(np.floor((cell_y[i] - y0) / dy))

            ix_min = max(0, ix_c - source_radius_x)
            ix_max = min(Nx - 1, ix_c + source_radius_x)
            iy_min = max(0, iy_c - source_radius_y)
            iy_max = min(Ny - 1, iy_c + source_radius_y)

            x_idx = np.arange(ix_min, ix_max + 1)
            y_idx = np.arange(iy_min, iy_max + 1)
            dxs = x[x_idx] - cell_x[i]
            dys = y[y_idx] - cell_y[i]

            patch = alpha * np.exp(-(dxs[:, None] ** 2 + dys[None, :] ** 2) / sigma**2)
            cell_source[ix_min:ix_max + 1, iy_min:iy_max + 1] += patch

        c1, c2 = step(c1, c2, cell_source)

        if n in frame_set:
            t = n * dt
            ccl19_file.write(f"# step={n}, time={t:.3f} s\n")
            np.savetxt(ccl19_file, c1, fmt="%.6e", delimiter=" ")
            ccl19_file.write("\n")

            ccl21_file.write(f"# step={n}, time={t:.3f} s\n")
            np.savetxt(ccl21_file, c2, fmt="%.6e", delimiter=" ")
            ccl21_file.write("\n")

        if n in frame_set:
            cell_x_timepoints.append(np.copy(cell_x))
            cell_y_timepoints.append(np.copy(cell_y))
            cell_type_timepoints.append(np.copy(cell_type))
            times_recorded.append(n * dt)

    ccl19_file.close()
    ccl21_file.close()

    cell_locations_file_name = "cell_locations_" + str(rep) + ".txt"
    cell_locations_file = output_dir / cell_locations_file_name

    with open(cell_locations_file, "w") as f:
        f.write("CellID,Time(s),x(microns),y(microns),bound\n")
        for t_idx, t in enumerate(times_recorded):
            xs = cell_x_timepoints[t_idx]
            ys = cell_y_timepoints[t_idx]
            types = cell_type_timepoints[t_idx]
            for cell_id, (xv, yv, bound_status) in enumerate(zip(xs, ys, types)):
                f.write(f"{cell_id},{t:.3e},{xv:.3e},{yv:.3e},{int(bound_status)}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--chemotaxis", type=str, choices=['keller_segel', 'saturating'], required=True)
    parser.add_argument("--v_max", type=float)
    parser.add_argument("--chi", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--Pe", type=float, required=True)
    parser.add_argument("--D_ratio", type=float, required=True)
    parser.add_argument("--n_cells", type=int, required=True)
    parser.add_argument("--CCL21_added", type=str.casefold, choices=["true", "false", "1", "0", "yes", "no"], required=True)
    parser.add_argument("--cell_motility", type=float, required=True)
    parser.add_argument("--cell_init", type=str, choices=["grid", "random"], required=True)
    parser.add_argument("--rng_seed", type=int, default=None)
    parser.add_argument("--n_reps", type=int, default=1)

    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    chemotaxis = args.chemotaxis
    v_max = args.v_max
    chi = args.chi
    alpha = args.alpha
    D_ratio = args.D_ratio
    Pe = args.Pe
    n_cells = args.n_cells
    CCL21_added = args.CCL21_added in ("true", "1", "yes")
    cell_motility = args.cell_motility
    cell_init = args.cell_init
    rng_seed = args.rng_seed
    n_reps = args.n_reps

    if args.chemotaxis == "saturating" and args.v_max is None:
        parser.error("--v_max is required when --chemotaxis is 'saturating'")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(rng_seed)

    for rep in range(n_reps):
        run(chemotaxis, v_max, chi, alpha, D_ratio, Pe, n_cells, CCL21_added, cell_motility, cell_init, rng, output_dir, rep)


if __name__ == "__main__":
    main()
