import numpy as np
import argparse


ccl19_file = open("CCL19_all.txt", "w")
ccl21_file = open("CCL21_all.txt", "w")

parser = argparse.ArgumentParser()

parser.add_argument('--chi', type=float)
parser.add_argument('--alpha', type=float)
parser.add_argument('--pe', type=float)
parser.add_argument('--D_ratio', type=float)
parser.add_argument('--n_cells', type=int)
parser.add_argument('--CCL21_added', type=str)
parser.add_argument('--cell_motility', type=float)

args = parser.parse_args()

chi = args.chi
alpha = args.alpha
D_ratio = args.D_ratio
Pe = args.pe
n_cells = args.n_cells
CCL21_added = args.CCL21_added.strip().lower() in ("true", "1", "yes")
cell_motility = args.cell_motility

Lx, Ly = 1600, 1400
Nx, Ny = 161, 141
dx, dy = Lx/(Nx-1), Ly/(Ny-1)

D_CCL21 = 100.0
D_CCL19 = D_ratio*D_CCL21
d = 1.3e-6
dt = 0.1
T_total = 5400
Nt_total = int(T_total/dt)

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

sqrt_term = np.sqrt(u**2 + 4*D_CCL21*d)
lambda1 = (u + sqrt_term) / (2*D_CCL21)
lambda2 = (u - sqrt_term) / (2*D_CCL21)

e1_0 = np.exp(lambda1 * 0.0)
e2_0 = np.exp(lambda2 * 0.0)
e1_L = np.exp(lambda1 * Ly)    
e2_L = np.exp(lambda2 * Ly)

den = (e2_L * e1_0 - e2_0 * e1_L)
if abs(den) < 1e-16:
    A = 0.0
    B = 0.0
else:
    A = (inlet_conc * e2_L) / den
    B = (inlet_conc * e1_L) / (e1_L * e2_0 - e2_L * e1_0)

c2_1d_y = A * np.exp(lambda1 * y) + B * np.exp(lambda2 * y) 
c2 = np.tile(c2_1d_y[None, :], (Nx, 1))  
c2[:, 0]  = inlet_conc                   
c2[:, -1] = outlet_conc                 

def compute_cell_source(x_grid, y_grid, x0, y0, alpha, sigma):
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')  
    source = alpha * np.exp(-((X - x0)**2 + (Y - y0)**2) / sigma**2)
    return source

def step(c1, c2, source):
    c1_new = c1.copy()
    c2_new = c2.copy()
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            lap1 = (
                (c1[i+1, j] - 2*c1[i, j] + c1[i-1, j]) / dx**2 +
                (c1[i, j+1] - 2*c1[i, j] + c1[i, j-1]) / dy**2
            )
            lap2 = (
                (c2[i+1, j] - 2*c2[i, j] + c2[i-1, j]) / dx**2 +
                (c2[i, j+1] - 2*c2[i, j] + c2[i, j-1]) / dy**2
            )

            adv_y1 = (c1[i, j+1] - c1[i, j]) / dy
            adv_y2 = (c2[i, j+1] - c2[i, j]) / dy

            c1_new[i, j] = (
                c1[i, j]
                + dt * (
                    D_CCL19 * lap1
                    - u * adv_y1
                    - d * c1[i, j]
                    + source[i, j]
                )
            )

            c2_new[i, j] = (
                c2[i, j]
                + dt * (
                    D_CCL21 * lap2
                    - u * adv_y2
                    - d * c2[i, j]
                )
            )

    c1_new[:, 0]  = 0.0                
    c1_new[:, -1] = 0.0                  
    c1_new[0, :]  = c1_new[1, :]         
    c1_new[-1, :] = c1_new[-2, :]        

    c2_new[0, :]  = c2_new[1, :]         
    c2_new[-1, :] = c2_new[-2, :]        
    c2_new[:, 0]  = inlet_conc           
    c2_new[:, -1] = outlet_conc         

    return c1_new, c2_new

n_bound = n_unbound = n_cells // 2
rng = np.random.default_rng()

nx = int(np.sqrt(n_cells))
ny = int(np.ceil(n_cells / nx))

dx_grid = Lx / nx               
dy_grid = (1200 - 200) / ny    

grid_x = np.linspace(0, Lx - dx_grid, nx) + dx_grid / 2
grid_y = np.linspace(200, 1200 - dy_grid, ny) + dy_grid / 2

gx, gy = np.meshgrid(grid_x, grid_y)
cell_x = gx.flatten()[:n_cells]
cell_y = gy.flatten()[:n_cells]

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

for n in range(Nt_total):
    
    if n % 100 == 0:
         print(f"t={n*dt}")
    
    for i in range(n_cells):
        ix = np.clip(np.searchsorted(x, cell_x[i]), 1, Nx-2)
        iy = np.clip(np.searchsorted(y, cell_y[i]), 1, Ny-2)
        c_total_local = c1[ix, iy] + c2[ix, iy]

        if cell_type[i] == 1:
            p_off_dt = 5e-3 * dt
            if np.random.rand() < p_off_dt:
                cell_type[i] = 0
        else:
            p_on_dt = 1.83e-3 * c_total_local * dt
            if np.random.rand() < p_on_dt:
                cell_type[i] = 1

    dcell_x = rng.normal(0, np.sqrt(2 * cell_motility * dt), n_cells)
    dcell_y = rng.normal(0, np.sqrt(2 * cell_motility * dt), n_cells)
    
    c_total = c1 + c2
    grad_x_grid, grad_y_grid = np.gradient(c_total, dx, dy)

    for i in range(n_cells):
        if cell_type[i] == 1:

            fx = (cell_x[i] - x[0]) / dx
            fy = (cell_y[i] - y[0]) / dy
            ixg = int(np.floor(fx))
            iyg = int(np.floor(fy))
            wx = fx - ixg
            wy = fy - iyg
            ixg = max(0, min(ixg, Nx - 2))
            iyg = max(0, min(iyg, Ny - 2))

            g00x = grad_x_grid[ixg    , iyg    ]
            g10x = grad_x_grid[ixg + 1, iyg    ]
            g01x = grad_x_grid[ixg    , iyg + 1]
            g11x = grad_x_grid[ixg + 1, iyg + 1]

            g00y = grad_y_grid[ixg    , iyg    ]
            g10y = grad_y_grid[ixg + 1, iyg    ]
            g01y = grad_y_grid[ixg    , iyg + 1]
            g11y = grad_y_grid[ixg + 1, iyg + 1]

            gx_i = (1 - wx) * (1 - wy) * g00x + wx * (1 - wy) * g10x + (1 - wx) * wy * g01x + wx * wy * g11x
            gy_i = (1 - wx) * (1 - wy) * g00y + wx * (1 - wy) * g10y + (1 - wx) * wy * g01y + wx * wy * g11y

            dcell_x[i] += chi * gx_i * dt
            dcell_y[i] += chi * gy_i * dt

    proposed_x = cell_x + dcell_x
    proposed_y = cell_y + dcell_y

    in_x_bounds = (proposed_x >= 0) & (proposed_x <= Lx)
    cell_x = np.where(in_x_bounds, proposed_x, cell_x)

    proposed_y = np.clip(proposed_y, 200, 1200)
    cell_y = proposed_y

    cell_source = np.zeros((Nx, Ny))
    for i in range(n_cells):
        cell_source += compute_cell_source(x, y, cell_x[i], cell_y[i], alpha, sigma)

    c1, c2 = step(c1, c2, cell_source)

    if n in frame_set:
        t = n * dt
        ccl19_file.write(f"# step={n}, time={t:.3f} s\n")
        np.savetxt(ccl19_file, c1, fmt="%.6e", delimiter=" ")
        ccl19_file.write("\n")
    
        ccl21_file.write(f"# step={n}, time={t:.3f} s\n")
        np.savetxt(ccl21_file, c2, fmt="%.6e", delimiter=" ")
        ccl21_file.write("\n")

    if n in frame_indices:
        cell_x_timepoints.append(np.copy(cell_x))
        cell_y_timepoints.append(np.copy(cell_y))
        cell_type_timepoints.append(np.copy(cell_type))
        times_recorded.append(n*dt)

ccl19_file.close()
ccl21_file.close()

output_file = "cell_locations.txt"

with open(output_file, "w") as f:
    f.write("CellID,Time(s),x(microns),y(microns),bound\n")
    for t_idx, t in enumerate(times_recorded):
        xs = cell_x_timepoints[t_idx]
        ys = cell_y_timepoints[t_idx]
        types = cell_type_timepoints[t_idx]
        for cell_id, (xv, yv, bound_status) in enumerate(zip(xs, ys, types)):
            f.write(f"{cell_id},{t:.3e},{xv:.3e},{yv:.3e},{int(bound_status)}\n")
