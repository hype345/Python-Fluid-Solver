import torch
import warnings
from tqdm import tqdm

class Solver3d:
    """
    Solves the 3D incompressible Navier Stokes equations using ...
    
    Solution Steps:
        • 
        • 
        • 
        • 
        • 
        •
        •
    
    Parameters:
        • config {dict}:
            -
        
        • state {pyflow.state.State3d object}

    Optional Parameters:
        •

    """
    def __init__(self, config, state):
        # Matching internal device & dtype with state's
        self.device =  state.device
        self.dtype = state.dtype


















class Solver2d:
    """
    Solves the 2D incompressible Navier Stokes equations using the finite difference projection method.

    PDE: du/dt + (u⋅∇)u = -∇p + 1/Re * ∇^2 u + f
    
    Solution Steps:
        • WENO5 advection
        • 6th order central difference diffusion
        • Incremental Rotational Pressure with a Conjugate Gradient solver

    Boundary Condition:
    - Types:
        • 'dirichlet'
            Value: fixed constant
        • 'neumann'
            Value: fixed slope
        • 'periodic'
            Value: None

    Parameters:
        • config {dict}:
            - 'top_bc_type' (type of BC for y=0) {string}
            - 'top_bc_value' (value of BC for y=0, indexed (y, x)) [dirichlet: m/s], [neumann: 1/s] {float, float}
            - 'bottom_bc_type' (type of BC for y=L) {string}
            - 'bottom_bc_value' (value of BC for y=L, indexed (y, x)) [dirichlet: m/s], [neumann: 1/s] {float, float}
            - 'left_bc_type' (type of BC for x=0) {string}
            - 'left_bc_value' (value of BC for x=0, indexed (y, x)) [dirichlet: m/s], [neumann: 1/s] {float, float}
            - 'right_bc_type' (type of BC for x=L) {string}
            - 'right_bc_value' (value of BC for x=L, indexed (y, x)) [dirichlet: m/s], [neumann: 1/s] {float, float}
            - 'CFL_number' (C <= 1. Smaller is safer) {float}
            - 'cg_steps' (number of steps used for Conjugate Gradient) {int}
            - 'weno_epsilon' (division by zero parameter used in WENO-5) {float}
        
        • state {pyflow.state.State1d object}
            - Note: expected to have the initial conditions already applied, i.e. the .velocity atribute is set to the initial condition.

    Optional Parameters:
        •

    """
    def __init__(self, config, state):
        # Matching internal device & dtype with state's
        self.device =  state.device
        self.dtype = state.dtype

        # Making sure state is in the dimensionless form
        state.dimensionless()

        # Storing the relevant conversion factors
        self.CHARACTERISTIC_TIME = state.CHARACTERISTIC_LENGTH / state.CHARACTERISTIC_SPEED
        self.CHARACTERISTIC_SPEED = state.CHARACTERISTIC_SPEED
        self.CHARACTERISTIC_PRESSURE = state.CHARACTERISTIC_DENSITY * state.CHARACTERISTIC_SPEED**2

        # Storing BC types and nondimensional values
        self.top_bc_type = config.get('top_bc_type', 'periodic')
        self.top_bc_value = None
        if self.top_bc_type == 'dirichlet':
            top_bc_value = config.get('top_bc_value', (0.0, 0.0)) 
            self.top_bc_value = torch.tensor(top_bc_value, device=self.device, dtype=self.dtype) / state.CHARACTERISTIC_SPEED
        elif self.top_bc_type == 'neumann':
            top_bc_value = config.get('top_bc_value', (0.0, 0.0)) 
            self.top_bc_value = torch.tensor(top_bc_value, device=self.device, dtype=self.dtype) * state.CHARACTERISTIC_LENGTH / state.CHARACTERISTIC_SPEED

        self.bottom_bc_type = config.get('bottom_bc_type', 'periodic')
        self.bottom_bc_value = None
        if self.bottom_bc_type == 'dirichlet':
            bottom_bc_value = config.get('bottom_bc_value', (0.0, 0.0)) 
            self.bottom_bc_value = torch.tensor(bottom_bc_value, device=self.device, dtype=self.dtype) / state.CHARACTERISTIC_SPEED
        elif self.bottom_bc_type == 'neumann':
            bottom_bc_value = config.get('bottom_bc_value', (0.0, 0.0))
            self.bottom_bc_value = torch.tensor(bottom_bc_value, device=self.device, dtype=self.dtype) * state.CHARACTERISTIC_LENGTH / state.CHARACTERISTIC_SPEED

        self.left_bc_type = config.get('left_bc_type', 'periodic')
        self.left_bc_value = None
        if self.left_bc_type == 'dirichlet':
            left_bc_value = config.get('left_bc_value', (0.0, 0.0)) 
            self.left_bc_value = torch.tensor(left_bc_value, device=self.device, dtype=self.dtype) / state.CHARACTERISTIC_SPEED
        elif self.left_bc_type == 'neumann':
            left_bc_value = config.get('left_bc_value', (0.0, 0.0))
            self.left_bc_value = torch.tensor(left_bc_value, device=self.device, dtype=self.dtype) * state.CHARACTERISTIC_LENGTH / state.CHARACTERISTIC_SPEED

        self.right_bc_type = config.get('right_bc_type', 'periodic')
        self.right_bc_value = None
        if self.right_bc_type == 'dirichlet':
            right_bc_value = config.get('right_bc_value', (0.0, 0.0)) 
            self.right_bc_value = torch.tensor(right_bc_value, device=self.device, dtype=self.dtype) / state.CHARACTERISTIC_SPEED
        elif self.right_bc_type == 'neumann':
            right_bc_value = config.get('right_bc_value', (0.0, 0.0)) 
            self.right_bc_value = torch.tensor(right_bc_value, device=self.device, dtype=self.dtype) * state.CHARACTERISTIC_LENGTH / state.CHARACTERISTIC_SPEED

        # Asserting that if periodic BCs are chosen on one side the other side must match
        if self.left_bc_type == 'periodic' or self.right_bc_type == 'periodic':
            assert self.left_bc_type == self.right_bc_type, "Must apply periodic boundary conditions to BOTH left & right if used"

        if self.top_bc_type == 'periodic' or self.bottom_bc_type == 'periodic':
            assert self.top_bc_type == self.bottom_bc_type, "Must apply periodic boundary conditions to BOTH top & bottom if used"

        # Checking if there are four neumann pressure BCs (this is the same as checking for 4 dirichlet velocity BCs)
        self.all_neumann = False
        if self.top_bc_type == self.bottom_bc_type == self.left_bc_type == self.right_bc_type == 'dirichlet':
            self.all_neumann = True

        # Storing dimensionless grid spacing
        self.dy = state.d[0]
        self.dx = state.d[1]

        # Storing division by zero safety parameters
        self.w_eps = torch.tensor(config.get('weno_epsilon', 1e-6), device=self.device, dtype=self.dtype)
        self.cg_eps = torch.tensor(torch.finfo(self.dtype).tiny, device=self.device, dtype=self.dtype)

        # Storing Conjugate Gradient steps
        self.cg_steps = config.get('cg_steps', 50)

        # Storing useful computational coefficients
        self.laplacian_coeff = torch.tensor(1.0 / 180.0, device=self.device, dtype=self.dtype)
        self.deriv_coeff = torch.tensor(1.0 / 60.0, device=self.device, dtype=self.dtype)
        self.inv_dx = 1.0 / self.dx
        self.inv_dy = 1.0 / self.dy
        self.inv_dx2 = 1.0 / (self.dx**2)
        self.inv_dy2 = 1.0 / (self.dy**2)
        self.inv_Re = torch.tensor(1.0 / state.Re, device=self.device, dtype=self.dtype)

        # Storing the CFL safety factor
        self.CFL_number = torch.tensor(config.get('CFL_number', 1.0), device=self.device, dtype=self.dtype)

        # Getting velocity & pressure fields
        self.velocity = state.velocity
        self.pressure = state.pressure

        # Getting shape of the grid
        self.Ny = state.N[0].item()
        self.Nx = state.N[1].item()

        # Using torch compile on ssp_rk3 code dynamically based on device
        if "cuda" in str(self.device):
            # Pre-allocate static input buffers to break CUDAGraph memory aliasing
            self.static_velocity = self.velocity.clone()
            self.static_pressure = self.pressure.clone()
            self.static_dt = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            
            self.compiled_ssp_rk3 = torch.compile(self.ssp_rk3, mode="reduce-overhead")
        else:
            self.compiled_ssp_rk3 = torch.compile(self.ssp_rk3, backend="inductor")

    def solve(self, steps=None, duration=None, max_steps=10_000, save_intermediates=False, dimensional=False, every_n_steps=1):
        """
        Solves the viscous Burgers' partial differential equation for a duration or for a fixed number of steps.

        Parameters:
            Note: Must supply either steps or duration but not both.
            • steps (sets number of solution steps) {int}
            • duration (sets solution duration) {float}
                - Note: if dimensional=True this value will be assumed to be in [s] but if dimensional=False it will be a dimensionless
            • max_steps (if using duration sets a hard limit for the number of steps) {int}
                - Note: if save_intermediates=True it will try to pre-allocate solution tensors of shape t[max_steps], sol[max_steps, N] so set to a large number with care
            • save_intermediates (controls whether the entire solution is return or if just the final step) {bool}
            • dimensional (controls whether the return results are dimensional or dimensionless) {bool}
            • every_n_steps (controls after how many steps to save an intermediate value) {int}
        """

        # You need to either supply duration or steps
        assert (steps is not None) or (duration is not None), "Must supply either steps or duration"

        # Both duration and steps shouldn't be supplied together. However, if they are it will default to using duration.
        if (steps is not None) and (duration is not None):
            warnings.warn("Both steps and duration should not be supplied together. Pick one or the other. Will default to using duration and steps will go unused.")


        # If dimensional==True this is the dimensional simulated time in [s].
        # if dimensional==False this is the dimensionless simulated time
        time = 0 

        # Running the simulation until time > duration or step >= max_steps
        if duration is not None:
            # Pre allocating wrost case sizes.
            if save_intermediates:
                allocated_frames = 1 + (max_steps // every_n_steps)
                t = torch.empty((allocated_frames), device=self.device, dtype=self.dtype)
                sol = torch.empty((allocated_frames, 3, self.Ny, self.Nx), device=self.device, dtype=self.dtype)
                t[0] = 0.0
                sol[0, :2] = self.velocity
                sol[0, 2] = self.pressure
                saved_count = 1

            step = 1
            with tqdm(unit="steps", desc="Simulating") as pbar:
                while time <= duration and step <= max_steps:
                    vel, p, dt = self.ssp_rk3_step()
                    if dimensional:
                        time += dt * self.CHARACTERISTIC_TIME
                    else:
                        time += dt
                    if save_intermediates and (step % every_n_steps == 0):
                        t[saved_count] = time
                        sol[saved_count, :2] = vel
                        sol[saved_count, 2] = p
                        saved_count += 1
                    step += 1
                    pbar.update(1)
            if dimensional:
                if save_intermediates:
                    sol[:, :2] = sol[:, :2] * self.CHARACTERISTIC_SPEED
                    sol[:, 2] = sol[:, 2] * self.CHARACTERISTIC_PRESSURE
                    return sol[:saved_count], t[:saved_count]
                else:
                    vel = vel * self.CHARACTERISTIC_SPEED
                    p = p * self.CHARACTERISTIC_PRESSURE
                    return vel, p, time
            else:
                if save_intermediates:
                    return sol[:saved_count], t[:saved_count]
                else:
                    return vel, p, time

        # Running the simulation for a fixed number of steps
        else:
            # Pre allocating sizes.
            if save_intermediates:
                allocated_frames = 1 + (steps // every_n_steps)
                t = torch.empty((allocated_frames), device=self.device, dtype=self.dtype)
                sol = torch.empty((allocated_frames, 3, self.Ny, self.Nx), device=self.device, dtype=self.dtype)
                t[0] = 0.0
                sol[0, :2] = self.velocity
                sol[0, 2] = self.pressure
                saved_count = 1
            for step in tqdm(range(steps), unit="steps", desc="Simulating"):
                vel, p, dt = self.ssp_rk3_step()
                time += dt
                if save_intermediates and ((step+1) % every_n_steps == 0):
                    t[saved_count] = time
                    sol[saved_count, :2] = vel
                    sol[saved_count, 2] = p
                    saved_count += 1

            if dimensional:
                if save_intermediates:
                    sol[:, :2] = sol[:, :2] * self.CHARACTERISTIC_SPEED
                    sol[:, 2] = sol[:, 2] * self.CHARACTERISTIC_PRESSURE
                    t = t * self.CHARACTERISTIC_TIME
                    return sol, t
                else:
                    vel = vel * self.CHARACTERISTIC_SPEED
                    p = p * self.CHARACTERISTIC_PRESSURE
                    time = time * self.CHARACTERISTIC_TIME
                    return vel, p, time
            else:
                if save_intermediates:
                    return sol, t
                else:
                    return vel, p, time

    
    def ssp_rk3_step(self):
        """
        Computes a single forward step using the Strong Stability Preserving Runge-Kutta 3rd Order scheme.
        Updates the current velocity field.
        Updates the currect pressure field
        Returns the new dimensionless velocity field, pressure field, and time step used.
        """

        # Getting an adaptive time step based on the CFL conditions
        dt = self.get_dt(self.velocity)

        if "cuda" in str(self.device):
            # Explicitly copy eager state into static buffers to guarantee memory separation.
            # This prevents PyTorch from seeing the input pointer as the previous output pointer.
            self.static_velocity.copy_(self.velocity)
            self.static_pressure.copy_(self.pressure)
            self.static_dt.copy_(dt)
            
            # Execute compiled function using static buffers
            self.velocity, self.pressure = self.compiled_ssp_rk3(
                self.static_velocity, self.static_pressure, self.static_dt
            )
        else:
            # Computing the SSP-RK3 step natively for CPU
            self.velocity, self.pressure = self.compiled_ssp_rk3(self.velocity, self.pressure, dt)

        return self.velocity, self.pressure, dt
    
    def ssp_rk3(self, velocity, pressure, dt):
        """
        Helper function for ssp_rks that does not update class variables and only does computation
        """
        # First approximation
        u1 = velocity + dt*self.get_rhs(velocity, pressure)
        u1, p1 = self.pressure_substep(u1, pressure, dt)

        # Second approximation
        u2 = ((3.0/4.0) * velocity) + ((1.0/4.0) * (u1 + (dt*self.get_rhs(u1, p1))))
        u2, p2 = self.pressure_substep(u2, p1, dt)

        # Final approximation
        u3 = ((1.0/3.0) * velocity) + ((2.0/3.0) * (u2 + (dt*self.get_rhs(u2, p2))))

        return self.pressure_substep(u3, p2, dt)

    
    def get_dt(self, velocity):
        """
        Uses to CFL limit and safety factor to determine the maximum allowable time step.
        """
        # Getting the velocity field padding values
        top_pad = self.get_padding_top(velocity, self.top_bc_type, self.top_bc_value)
        bot_pad = self.get_padding_bottom(velocity, self.bottom_bc_type, self.bottom_bc_value)
        left_pad = self.get_padding_left(velocity, self.left_bc_type, self.left_bc_value)
        right_pad = self.get_padding_right(velocity, self.right_bc_type, self.right_bc_value)

        # Creating the full padded array 
        vel_pad = torch.nn.functional.pad(velocity, (3, 3, 3, 3), mode='constant', value=0.0)
        vel_pad[:, :3, 3:-3] = top_pad
        vel_pad[:, -3:, 3:-3] = bot_pad
        vel_pad[:, 3:-3, :3] = left_pad
        vel_pad[:, 3:-3, -3:] = right_pad

        # Finding the absolute maximum velocity values
        max_v, max_u = torch.amax(torch.abs(vel_pad), dim=(1, 2))

        # Calculating advection & diffusion limits
        advect_lim = (max_v * self.inv_dy) + (max_u * self.inv_dx)
        diffuse_lim = 2.0*self.inv_Re*(self.inv_dy2 + self.inv_dx2)

        return self.CFL_number / (advect_lim + diffuse_lim)
    
    def get_rhs(self, velocity, pressure):
        """
        Calculates the right hand side of the incompressible Navier Stokes equation.
        """
        # Calculating the divergence of the velocity field
        top_pad = self.get_padding_top(velocity, self.top_bc_type, self.top_bc_value)
        bot_pad = self.get_padding_bottom(velocity, self.bottom_bc_type, self.bottom_bc_value)
        left_pad = self.get_padding_left(velocity, self.left_bc_type, self.left_bc_value)
        right_pad = self.get_padding_right(velocity, self.right_bc_type, self.right_bc_value)

        # Pads the velocity field for WENO
        y_pad = torch.cat([top_pad, velocity, bot_pad], dim=1) # (2, Ny+6, Nx)
        x_pad = torch.cat([left_pad, velocity, right_pad], dim=2) # (2, Ny, Nx+6)

        # Computes the divergence of the flux tensor
        advection = self.weno5(y_pad, x_pad)

        # Computes the diffusion
        diffusion = self.inv_Re * self.central_difference_laplacian(y_pad, x_pad)

        # Computes the pressure gradient
        pressure_grad = self.apply_pressure_gradient(pressure)

        return diffusion - advection - pressure_grad
    
    def pressure_substep(self, velocity, pressure, dt):
        """
        Handles all of the pressure substeps during an SSP-RK3 stage after calculating the intermediate velocity
        1) solve for incremental pressure 
        2) project velocity
        3) update pressure
        """
        # Calculating the divergence of the intermediate velocity field
        div_velocity = self.get_divergence(velocity)

        # Calculating the incremental pressure
        phi = self.pressure_poisson(div_velocity, dt)

        # projecting velocity to zero divergence
        velocity = self.pressure_projection(velocity, phi, dt)

        # Updating pressure
        pressure = self.rotational_incremental_pressure(pressure, phi, div_velocity)

        return velocity, pressure
    
    def pressure_projection(self, velocity, phi, dt):
        """
        Projects the intermediate divergent velocity field to a divergent free one
        """
        # Computes the gradient of the incremental pressure phi (Ny, Nx) --> (2, Ny, Nx)
        grad_phi = self.apply_pressure_gradient(phi)

        return velocity - dt*grad_phi

    def rotational_incremental_pressure(self, pressure, phi, div_velocity):
        """
        Updates the pressure using the rotational incremental pressure formula.
        p_new = p_old + ϕ - 1/Re * (∇⋅u)
        """
        return pressure + phi - self.inv_Re * div_velocity.squeeze(0)

    def apply_pressure_gradient(self, pressure):
        """
        Helper function to pad & compute the gradient of pressure. Expected to be of shape (Ny, Nx)
        """
        # Computes the padded pressure tensors (1, Ny+6, Nx), (1, Ny, Nx+6)
        phi = pressure.unsqueeze(0)
        y_pad, x_pad = self.pad_pressure(phi, phi)

        # Calculates dp/dy and dp/dx (1, Ny, Nx)
        dpdy, dpdx = self.central_difference_derivative(y_pad, x_pad)

        return torch.cat([dpdy, dpdx], dim=0) # (2, Ny, Nx)
    
    def get_divergence(self, velocity):
        """
        Helper function to calculate the divergence of the velocity field
        """
        # Calculating the divergence of the velocity field
        top_pad = self.get_padding_top(velocity, self.top_bc_type, self.top_bc_value)
        bot_pad = self.get_padding_bottom(velocity, self.bottom_bc_type, self.bottom_bc_value)
        left_pad = self.get_padding_left(velocity, self.left_bc_type, self.left_bc_value)
        right_pad = self.get_padding_right(velocity, self.right_bc_type, self.right_bc_value)
        y_pad = torch.cat([top_pad[0:1], velocity[0:1], bot_pad[0:1]], dim=1) # (1, Ny+6, Nx)
        x_pad = torch.cat([left_pad[1:2], velocity[1:2], right_pad[1:2]], dim=2) # (1, Ny, Nx+6)
        dvdy,  dudx = self.central_difference_derivative(y_pad, x_pad)
        return dvdy + dudx # (1, Ny, Nx)

    def pressure_poisson(self, div_velocity, dt):
        """
        Solves the pressure poisson's equation for incremental pressure ϕ. ∇^2ϕ = 1/dt * (∇⋅u).
        """
        rhs = 1.0/dt * div_velocity
        
        # If all pressure BCs are neumann then need mean of rhs to be 0 for a well posed problem.
        if self.all_neumann:
            rhs = rhs - torch.mean(rhs)

        # Uses the conjugate gradient method to solve for phi
        phi = self.conjugate_gradient(self.apply_pressure_laplacian, rhs) # (1, Ny, Nx) 
        
        # Set referance pressure to zero if all neumann BCs because only defined up to a constant
        if self.all_neumann:
            phi = phi - phi[:, 0, 0]

        return phi.squeeze(0) # (Ny, Nx)
    
    def conjugate_gradient(self, A, b):
        """
        Applies the Conjugate Gradient method to solve Ax=b.
        """
        # Initalize the residual, search direction, and x0
        r = b.clone() # equivalent to r = b - A(x) when x0=0
        d = r.clone()
        x = torch.zeros_like(b)

        # Creating 1d views for dot product performance
        r_flat = r.view(-1)

        # Pre-computes the rTr
        r_dot = torch.vdot(r_flat, r_flat)

        # Conjugate gradient loop for the specified number of steps
        # Note the Algorithm should converge mathmatically at most after Nx*Ny steps.
        for i in range(self.cg_steps):
            # Computes & stores A(d) as it is used twice per iteration
            A_d = A(d)

            # Creating 1d views for dot product performance
            d_flat = d.view(-1)
            A_d_flat = A_d.view(-1)

            # Calculates the step size & updates x and r acordingly
            alpha = r_dot / (torch.vdot(d_flat, A_d_flat) + self.cg_eps)

            x = x + alpha * d
            r = r - alpha * A_d

            # Creating 1d views for dot product performance
            r_flat = r.view(-1)

            # Store old r_dot for the denominator of beta
            r_dot_old = r_dot

            # Compute new r_dot for the next iteration's alpha and current beta
            r_dot = torch.vdot(r_flat, r_flat)

            # Computes Beta
            beta =  r_dot / (r_dot_old + self.cg_eps)

            # Computes new direction
            d = r + beta * d
        return x

    def apply_pressure_laplacian(self, phi):
        """
        Calculates the Laplacian of pressure. Expected to be of shape (1, Ny, Nx)
        Creates the Laplacian opperator from a composition of the Divergence and Gradient opperators: L=DG
        """
        # Getting the padded pressure tensors (1, Ny+12, Nx), (1, Ny, Nx+12)
        y_pad, x_pad = self.pad_pressure(phi, phi, amount=6)

        # Calculates dϕ/dy & dϕ/dx (1, Ny+6, Nx), (1, Ny, Nx+6)
        dfdy, dfdx = self.central_difference_derivative(y_pad, x_pad)

        # Calculates d2ϕ/dy2 & d2ϕ/dx2 (1, Ny, Nx), (1, Ny, Nx)
        d2fdy2, d2fdx2 = self.central_difference_derivative(dfdy, dfdx)

        # Computing the Laplacian ∇^2ϕ = d^2/dx^2ϕ + d^2/dy^2ϕ (1, Ny, Nx)
        return d2fdy2 + d2fdx2

    def pad_pressure(self, phi_y, phi_x, amount=3):
        """
        Applies padding to pressure based on velocity BCs.
        velocity periodic --> pressure periodic
        velocity dirichlet --> pressure neumann=0 
        velocity neumann --> pressure dirichlet=0
        Note if all pressure BCs are neumann must set referance value.
        """
        # Defining the zero value as a tensor
        zero_val = torch.tensor([0.0], device=self.device, dtype=self.dtype)

        # Note: phi is unsqueezed to match with velocity pad functions so shape (1, Ny, Nx)
        # Top padding 
        if self.top_bc_type == 'periodic':
            top_pad = self.get_padding_top(phi_y, 'periodic', amount=amount)
        elif self.top_bc_type == 'dirichlet':
            top_pad = self.get_padding_top(phi_y, 'neumann', value=zero_val, amount=amount)
        elif self.top_bc_type == 'neumann': 
            top_pad = self.get_padding_top(phi_y, 'dirichlet', value=zero_val, amount=amount)

        # Bottom padding
        if self.bottom_bc_type == 'periodic':
            bot_pad = self.get_padding_bottom(phi_y, 'periodic', amount=amount)
        elif self.bottom_bc_type == 'dirichlet':
            bot_pad = self.get_padding_bottom(phi_y, 'neumann', value=zero_val, amount=amount)
        elif self.bottom_bc_type == 'neumann':
            bot_pad = self.get_padding_bottom(phi_y, 'dirichlet', value=zero_val, amount=amount)

        y_pad = torch.cat([top_pad, phi_y, bot_pad], dim=1)

        # Left padding
        if self.left_bc_type == 'periodic':
            left_pad = self.get_padding_left(phi_x, 'periodic', amount=amount)
        elif self.left_bc_type == 'dirichlet':
            left_pad = self.get_padding_left(phi_x, 'neumann', value=zero_val, amount=amount)
        elif self.left_bc_type == 'neumann':
            left_pad = self.get_padding_left(phi_x, 'dirichlet', value=zero_val, amount=amount)

        # Right padding
        if self.right_bc_type == 'periodic':
            right_pad = self.get_padding_right(phi_x, 'periodic', amount=amount)
        elif self.right_bc_type == 'dirichlet':
            right_pad = self.get_padding_right(phi_x, 'neumann', value=zero_val, amount=amount)
        elif self.right_bc_type == 'neumann':
            right_pad = self.get_padding_right(phi_x, 'dirichlet', value=zero_val, amount=amount)
        
        x_pad = torch.cat([left_pad, phi_x, right_pad], dim=2)

        return y_pad, x_pad


    def central_difference_derivative(self, y_pad, x_pad):
        """
        Uses a 6th order central difference derivative stencil to compute df/dx and df/dy.
        """
        # Stencil uses three points on each side in y (C, Ny+6, Nx) --> (C, Ny, Nx)
        dfdy = self.deriv_coeff * (
            -1 * y_pad[:, :-6, :] 
            + 9 * y_pad[:, 1:-5, :] 
            - 45 * y_pad[:, 2:-4, :] 
            # 0 * y_pad[:, 3:-3, :] 
            + 45 * y_pad[:, 4:-2, :] 
            - 9 * y_pad[:, 5:-1, :] 
            + 1 * y_pad[:, 6:, :]
        ) * self.inv_dy

        # Stencil uses three points on each side in x (C, Ny, Nx+6) --> (C, Ny, Nx)
        dfdx = self.deriv_coeff * (
            -1 * x_pad[:, :, :-6] 
            + 9 * x_pad[:, :, 1:-5] 
            - 45 * x_pad[:, :, 2:-4] 
            # 0 * x_pad[:, :, 3:-3]
            + 45 * x_pad[:, :, 4:-2] 
            - 9 * x_pad[:, :, 5:-1] 
            + 1 * x_pad[:, :, 6:]
        ) * self.inv_dx

        return dfdy, dfdx
    

    def central_difference_laplacian(self, y_pad, x_pad):
        """
        Uses a 6th order central difference Laplacian stencil to compute d^2u/dx^2 + d^2u/dy^2 and d^2v/dx^2 + d^2v/dy^2.
        """
        # Stencil uses three points on each side in y (C, Ny+6, Nx) --> (C, Ny, Nx)
        d2fdy2 = self.laplacian_coeff * (
              2 * y_pad[:,:-6,:]      
            - 27 * y_pad[:,1:-5,:]    
            + 270 * y_pad[:,2:-4,:]   
            - 490 * y_pad[:,3:-3,:]   
            + 270 * y_pad[:,4:-2,:]   
            - 27 * y_pad[:,5:-1,:]    
            + 2 * y_pad[:,6:,:]       
        ) * self.inv_dy2

        # Stencil uses three points on each side in x (C, Ny, Nx+6) --> (C, Ny, Nx)
        d2fdx2 = self.laplacian_coeff * (
              2 * x_pad[:,:,:-6]      
            - 27 * x_pad[:,:,1:-5]    
            + 270 * x_pad[:,:,2:-4]   
            - 490 * x_pad[:,:,3:-3]   
            + 270 * x_pad[:,:,4:-2]   
            - 27 * x_pad[:,:,5:-1]    
            + 2 * x_pad[:,:,6:]       
        ) * self.inv_dx2

        return d2fdy2 + d2fdx2

    def weno5(self, y_pad, x_pad):
        """
        Uses Jiang-Shu WENO-5 to calculate the divergence of F where is F is the flux tensor: (u^2, uv; uv v^2)
        """
        #-----------------------------
        # Calculating dv^2/dy & duv/dy
        #-----------------------------
        # Applies Lax-Friedrichs flux splitting on padded velocity field (2, Ny+6, Nx)
        fp, fm = self.flux_splitting_y(y_pad)

        # Uses WENO-5 interpolation with a left biased stencil on f+ to calculate f+_{i+1/2} (2, Ny+1, Nx)
        f_p_half = self.weno5_interpolation(
            fp[:,:-5,:], fp[:,1:-4,:], fp[:,2:-3,:], fp[:,3:-2,:], fp[:,4:-1,:]
        )
        
        # Uses WENO-5 interpolation with a right biased stencil on f- to calculate f-_{i-1/2} (2, Ny+1, Nx)
        f_m_half = self.weno5_interpolation(
            fm[:,5:,:], fm[:,4:-1,:], fm[:,3:-2,:], fm[:,2:-3,:], fm[:,1:-4,:]
        )

        # Calculates the total flux using f = (f+) + (f-) at all interfaces (2, Ny+1, Nx)
        f_half = f_p_half + f_m_half

        # Calculates the derivative (2, Ny+1, Nx) --> (2, Ny, Nx)
        # dfdy[0]=dv2dy, dfdy[1]=duvdy
        dfdy = (f_half[:,1:,:] - f_half[:,:-1,:]) * self.inv_dy 

        #-----------------------------
        # Calculating duv/dx & du^2/dx
        #-----------------------------
        # Applies Lax-Friedrichs flux splitting on padded velocity field (2, Ny, Nx+6)
        fp, fm = self.flux_splitting_x(x_pad)

        # Uses WENO-5 interpolation with a left biased stencil on f+ to calculate f+_{i+1/2} (2, Ny, Nx+1)
        f_p_half = self.weno5_interpolation(
            fp[:,:,:-5], fp[:,:,1:-4], fp[:,:,2:-3], fp[:,:,3:-2], fp[:,:,4:-1]
        )
        
        # Uses WENO-5 interpolation with a right biased stencil on f- to calculate f-_{i-1/2} (2, Ny, Nx+1)
        f_m_half = self.weno5_interpolation(
            fm[:,:,5:], fm[:,:,4:-1], fm[:,:,3:-2], fm[:,:,2:-3], fm[:,:,1:-4]
        )

        # Calculates the total flux using f = (f+) + (f-) at all interfaces (2, Ny, Nx+1)
        f_half = f_p_half + f_m_half

        # Calculates the derivatives (2, Ny, Nx+1) --> (2, Ny, Nx)
        # dfdx[0]=duvdx, dfdx[1]=du2dx
        dfdx = (f_half[:,:,1:] - f_half[:,:,:-1]) * self.inv_dx

        return dfdy + dfdx

    def flux_splitting_y(self, y_pad):
        """
        Uses Lax-Friedrichs flux splitting on the y-directional flux vector find f+ & f- vectors where df+/dv >= 0 and df-/dv <= 0.
        """
        # Seperating u,v components (2, Ny+6, Nx) --> (Ny+6,Nx), (Ny+6, Nx)
        v_pad, u_pad = y_pad

        # Maximum absolute eigenvalue of the Jacobian of the y-directional flux vector
        alpha = torch.max(torch.abs(2*v_pad))

        # y-component
        f_plus_y = 0.5*(v_pad**2 + alpha*v_pad)
        f_minus_y = 0.5*(v_pad**2 - alpha*v_pad)
        # x-component
        f_plus_x = 0.5*(u_pad*v_pad + alpha*u_pad)
        f_minus_x = 0.5*(u_pad*v_pad - alpha*u_pad)
        # Batching for WENO (2, Ny+6, Nx)
        fp = torch.stack([f_plus_y, f_plus_x])
        fm = torch.stack([f_minus_y, f_minus_x])

        return fp, fm

    def flux_splitting_x(self, x_pad):
        """
        Uses Lax-Friedrichs flux splitting on the x-directional flux vector to find f+ & f- vectors where df+/du >= 0 and df-/du <= 0.
        """
        # Seperating u,v components(2, Ny, Nx+6) --> (Ny,Nx+6), (Ny,Nx+6)
        v_pad, u_pad = x_pad

        # Maximum absolute eigenvalue of the Jacobian of the x-directional flux vector
        alpha = torch.max(torch.abs(2*u_pad))

        # y-component
        f_plus_y = 0.5*(u_pad*v_pad + alpha*v_pad)
        f_minus_y = 0.5*(u_pad*v_pad - alpha*v_pad)
        # x-component
        f_plus_x = 0.5*(u_pad**2 + alpha*u_pad)
        f_minus_x = 0.5*(u_pad**2 - alpha*u_pad)
        # Batching for WENO (2, Ny, Nx+6)
        fp = torch.stack([f_plus_y, f_plus_x])
        fm = torch.stack([f_minus_y, f_minus_x])

        return fp, fm
    
    def weno5_interpolation(self, v1, v2, v3, v4, v5):
        """
        Core Jiang-Shu WENO-5 interpolation using a left-biased stencil (interpolates right face).
        Reverse inputs to use a right-biased stencil (interpolates left face).
        """
        # Smoothness indicators
        beta0 = (13.0 / 12.0) * (v1 - 2*v2 + v3)**2 + (1.0 / 4.0) * (v1 - 4*v2 + 3*v3)**2
        beta1 = (13.0 / 12.0) * (v2 - 2*v3 + v4)**2 + (1.0 / 4.0) * (v2 - v4)**2
        beta2 = (13.0 / 12.0) * (v3 - 2*v4 + v5)**2 + (1.0 / 4.0) * (3*v3 - 4*v4 + v5)**2

        # Linear weights
        d0, d1, d2 = 0.1, 0.6, 0.3

        # Unnormalized alpha weights
        alpha0 = d0 / (self.w_eps + beta0)**2
        alpha1 = d1 / (self.w_eps + beta1)**2
        alpha2 = d2 / (self.w_eps + beta2)**2

        sum_alpha = alpha0 + alpha1 + alpha2

        # Normalized WENO weights
        w0 = alpha0 / sum_alpha
        w1 = alpha1 / sum_alpha
        w2 = alpha2 / sum_alpha

        # Polynomial reconstructions
        p0 = (1.0 / 3.0) * v1 - (7.0 / 6.0) * v2 + (11.0 / 6.0) * v3
        p1 = -(1.0 / 6.0) * v2 + (5.0 / 6.0) * v3 + (1.0 / 3.0) * v4
        p2 = (1.0 / 3.0) * v3 + (5.0 / 6.0) * v4 - (1.0 / 6.0) * v5

        return w0 * p0 + w1 * p1 + w2 * p2

    def get_padding_top(self, f, bc_type, value=None, amount=3):
        """
        Computes the values for the 3 top padding cells to give the desired boundary condition at y=0.
        """
        # Copy values from bottom edge to top padding
        if bc_type == 'periodic':
            return f[:, -amount:, :]

        # Linear anti-symmetric reflection: Forces the average of ghost and interior cells to equal the boundary value.
        elif bc_type == 'dirichlet':
            return 2 * value.view(-1, 1, 1) - f[:, :amount, :].flip(1)
        
        # Linear symmetric extrapolation: Sets ghost values such that the central difference across the interface matches the gradient.
        elif bc_type == 'neumann':
            offsets = (2*torch.arange(amount, device=self.device, dtype=self.dtype)+1).view(1, -1, 1)
            return f[:, :amount, :].flip(1) - (offsets * value.view(-1, 1, 1) * self.dy)

    def get_padding_bottom(self, f, bc_type, value=None, amount=3):
        """
        Computes the values for the 3 bottom padding cells to give the desired boundary condition at y=L.
        """
        # Copy values from top edge to bottom padding
        if bc_type == 'periodic':
            return f[:, :amount, :]
        
        # Linear anti-symmetric reflection: Forces the average of ghost and interior cells to equal the boundary value.
        elif bc_type == 'dirichlet':
            return 2 * value.view(-1, 1, 1) - f[:, -amount:, :].flip(1)
        
        # Linear symmetric extrapolation: Sets ghost values such that the central difference across the interface matches the gradient.
        elif bc_type == 'neumann':
            offsets = (2*torch.arange(amount, device=self.device, dtype=self.dtype)+1).view(1, -1, 1)
            return f[:, -amount:, :].flip(1) + (offsets * value.view(-1, 1, 1) * self.dy)

    def get_padding_left(self, f, bc_type, value=None, amount=3):
        """
        Computes the values for the 3 left padding cells to give the desired boundary condition at x=0.
        """
        # Copy values from right edge to left padding
        if bc_type == 'periodic':
            return f[:, :, -amount:]

        # Linear anti-symmetric reflection: Forces the average of ghost and interior cells to equal the boundary value.
        elif bc_type == 'dirichlet':
            return 2 * value.view(-1, 1, 1) - f[:, :, :amount].flip(2)
        
        # Linear symmetric extrapolation: Sets ghost values such that the central difference across the interface matches the gradient.
        elif bc_type == 'neumann':
            offsets = (2*torch.arange(amount, device=self.device, dtype=self.dtype)+1).view(1, 1, -1)
            return f[:, :, :amount].flip(2) - (offsets * value.view(-1, 1, 1) * self.dx)

    def get_padding_right(self, f, bc_type, value=None, amount=3):
        """
        Computes the values for the 3 right padding cells to give the desired boundary condition at x=L.
        """
        # Copy values from left edge to right padding
        if bc_type == 'periodic':
            return f[:, :, :amount]
        
        # Linear anti-symmetric reflection: Forces the average of ghost and interior cells to equal the boundary value.
        elif bc_type == 'dirichlet':
            return 2 * value.view(-1, 1, 1) - f[:, :, -amount:].flip(2)
        
        # Linear symmetric extrapolation: Sets ghost values such that the central difference across the interface matches the gradient.
        elif bc_type == 'neumann':
            offsets = (2*torch.arange(amount, device=self.device, dtype=self.dtype)+1).view(1, 1, -1)
            return f[:, :, -amount:].flip(2) + (offsets * value.view(-1, 1, 1) * self.dx)
        



















class Solver1d:
    """
    Solves the 1D viscous Burgers' equation using the finite difference method.
    
    PDE: du/dt + u*du/dx = 1/Re * d^2u/dx^2 (dimensionless form)
        ---> du/dt + dF/dx = 0, F = u^2/2 + 1/Re * du/dx (conservation form)

    Finite Difference Method Scheme:
        • Use WENO-5 to calculate df/dx where f=u^2/2
        • Use a 6th order center differnce stencil to calculate 1/Re * d^2u/dx^2
        • Use SSP-RK3 to step in time with an adaptive CFL based dt

    Boundary Condition:
        - Types:
            • 'dirichlet'
                Value: fixed constant
            • 'neumann'
                Value: fixed slope
            • 'periodic'
                Value: None
    
    Parameters:
        • config {dict}:
            - 'left_bc_type' (type of BC for x=0) {string}
            - 'left_bc_value' (value of BC for x=0) [dirichlet: m/s], [neumann: 1/s] {float}
            - 'right_bc_type' (type of BC for x=L) {string}
            - 'right_bc_value' (value of BC for x=L) [dirichlet: m/s], [neumann: 1/s] {float}
            - 'CFL_number' (C <= 1. Smaller is safer) {float}
            - 'epsilon' (division by zero parameter used in WENO-5) {float}
        
        • state {pyflow.state.State1d object}
            - Note: expected to have the initial conditions already applied, i.e. the .velocity atribute is set to the initial condition.
    """
    def __init__(self, config, state):
        # Matching internal device & dtype with state's
        self.device = state.device
        self.dtype = state.dtype

        # Making sure state is in the dimensionless form
        state.dimensionless()

        # Storing the relevant conversion factors
        self.CHARACTERISTIC_TIME = state.CHARACTERISTIC_LENGTH / state.CHARACTERISTIC_SPEED
        self.CHARACTERISTIC_SPEED = state.CHARACTERISTIC_SPEED

        # Storing BC types and nondimensional values
        self.left_bc_type = config.get('left_bc_type', 'periodic')
        if self.left_bc_type == 'dirichlet':
            left_bc_value = config.get('left_bc_value', 0.0) / state.CHARACTERISTIC_SPEED
            self.left_bc_value = torch.tensor(left_bc_value, device=self.device, dtype=self.dtype)
        elif self.left_bc_type == 'neumann':
            left_bc_value = (config.get('left_bc_value', 0.0) * state.CHARACTERISTIC_LENGTH) / state.CHARACTERISTIC_SPEED
            self.left_bc_value = torch.tensor(left_bc_value, device=self.device, dtype=self.dtype)

        self.right_bc_type = config.get('right_bc_type', 'periodic')
        if self.right_bc_type == 'dirichlet':
            right_bc_value = config.get('right_bc_value', 0.0) / state.CHARACTERISTIC_SPEED
            self.right_bc_value = torch.tensor(right_bc_value, device=self.device, dtype=self.dtype)
        elif self.right_bc_type == 'neumann':
            right_bc_value = (config.get('right_bc_value', 0.0) * state.CHARACTERISTIC_LENGTH) / state.CHARACTERISTIC_SPEED
            self.left_bc_value = torch.tensor(left_bc_value, device=self.device, dtype=self.dtype)

        # Asserting that if periodic BCs are chosen on one side the other side must match
        if self.left_bc_type == 'periodic' or self.right_bc_type == 'periodic':
            assert self.left_bc_type == self.right_bc_type, "Must apply periodic boundary conditions to BOTH left & right if used"

        # Storing dimensionless grid spacing
        self.dx = state.d[0]

        # Storing WENO-5 division by zero safety parameter
        self.eps = torch.tensor(config.get('epsilon', 1e-6), device=self.device, dtype=self.dtype)

        # Storing useful computational coefficients
        self.laplacian_coeff = torch.tensor(1.0 / 180.0, device=self.device, dtype=self.dtype)
        self.inv_dx = 1.0 / self.dx
        self.inv_dx2 = 1.0 / (self.dx**2)
        self.inv_Re = torch.tensor(1.0 / state.Re, device=self.device, dtype=self.dtype)

        # Storing the CFL safety factor
        self.CFL_number = torch.tensor(config.get('CFL_number', 1.0), device=self.device, dtype=self.dtype)

        # Getting velocity field 
        self.u = state.velocity

        # Getting shape of the field
        self.N = state.N.item()

    def solve(self, steps=None, duration=None, max_steps=10_000, save_intermediates=False, dimensional=False, every_n_steps=1):
        """
        Solves the viscous Burgers' partial differential equation for a duration or for a fixed number of steps.

        Parameters:
            Note: Must supply either steps or duration but not both.
            • steps (sets number of solution steps) {int}
            • duration (sets solution duration) {float}
                - Note: if dimensional=True this value will be assumed to be in [s] but if dimensional=False it will be a dimensionless
            • max_steps (if using duration sets a hard limit for the number of steps) {int}
                - Note: if save_intermediates=True it will try to pre-allocate solution tensors of shape t[max_steps], sol[max_steps, N] so set to a large number with care
            • save_intermediates (controls whether the entire solution is return or if just the final step) {bool}
            • dimensional (controls whether the return results are dimensional or dimensionless) {bool}
            • every_n_steps (controls after how many steps to save an intermediate value) {int}
        """

        # You need to either supply duration or steps
        assert (steps is not None) or (duration is not None), "Must supply either steps or duration"

        # Both duration and steps shouldn't be supplied together. However, if they are it will default to using duration.
        if (steps is not None) and (duration is not None):
            warnings.warn("Both steps and duration should not be supplied together. Pick one or the other. Will default to using duration and steps will go unused.")


        # If dimensional==True this is the dimensional simulated time in [s].
        # if dimensional==False this is the dimensionless simulated time
        time = 0 

        # Running the simulation until time > duration or step >= max_steps
        if duration is not None:

            # Pre allocating wrost case sizes.
            if save_intermediates:
                allocated_frames = 1 + (max_steps // every_n_steps)
                t = torch.empty((allocated_frames), device=self.device, dtype=self.dtype)
                sol = torch.empty((allocated_frames, self.N), device=self.device, dtype=self.dtype)
                t[0] = 0.0
                sol[0] = self.u
                saved_count = 1

            step = 1
            with tqdm(unit="steps", desc="Simulating") as pbar:
                while time <= duration and step <= max_steps:
                    u, dt = self.ssp_rk3_step()
                    if dimensional:
                        time += dt * self.CHARACTERISTIC_TIME
                    else:
                        time += dt
                    if save_intermediates and (step % every_n_steps == 0):
                        t[saved_count] = time
                        sol[saved_count] = u
                        saved_count += 1
                    step += 1
                    pbar.update(1)
            if dimensional:
                if save_intermediates:
                    sol = sol * self.CHARACTERISTIC_SPEED
                    return sol[:saved_count], t[:saved_count]
                else:
                    u = u * self.CHARACTERISTIC_SPEED
                    return u, time
            else:
                if save_intermediates:
                    return sol[:saved_count], t[:saved_count]
                else:
                    return u, time

        # Running the simulation for a fixed number of steps
        else:
            # Pre allocating sizes.
            if save_intermediates:
                allocated_frames = 1 + (steps // every_n_steps)
                t = torch.empty((allocated_frames), device=self.device, dtype=self.dtype)
                sol = torch.empty((allocated_frames, self.N), device=self.device, dtype=self.dtype)
                t[0] = 0.0
                sol[0] = self.u
                saved_count = 1
            for step in tqdm(range(steps), unit="steps", desc="Simulating"):
                u, dt = self.ssp_rk3_step()
                time += dt
                if save_intermediates and ((step+1) % every_n_steps ==0):
                    t[saved_count] = time
                    sol[saved_count] = u
                    saved_count += 1

            if dimensional:
                if save_intermediates:
                    sol = sol * self.CHARACTERISTIC_SPEED
                    t = t * self.CHARACTERISTIC_TIME
                    return sol, t
                else:
                    u = u * self.CHARACTERISTIC_SPEED
                    time = time * self.CHARACTERISTIC_TIME
                    return u, time
            else:
                if save_intermediates:
                    return sol, t
                else:
                    return u, time
            
    def ssp_rk3_step(self):
        """
        Computes a single forward step using the Strong Stability Preserving Runge-Kutta 3rd Order scheme.
        Updates the current velocity field.
        Returns the new dimensionless velocity field and the dimensionless time step used.
        """

        # Getting an adaptive time step based on the CFL conditions
        dt = self.get_dt(self.u)

        # First approximation
        u1 = self.u + dt*self.get_rhs(self.u)

        # Second approximation
        u2 = ((3.0/4.0) * self.u) + ((1.0/4.0) * (u1 + (dt*self.get_rhs(u1))))

        # Final approximation
        self.u = ((1.0/3.0) * self.u) + ((2.0/3.0) * (u2 + (dt*self.get_rhs(u2))))

        return self.u, dt


    def get_dt(self, u):
        """
        Uses to CFL limit and safety factor to determine the maximum allowable time step.
        """
        # Padding velocity field to account for boundary values
        left_padding = self.get_padding_left(u)
        right_padding = self.get_padding_right(u)
        u_pad = torch.cat([left_padding, u, right_padding], dim=0)

        return self.CFL_number / ((torch.max(torch.abs(u_pad))*self.inv_dx) + (2.0*self.inv_Re*self.inv_dx2))

    def get_rhs(self, u):
        """
        Calculates the right hand side of the viscous Burgers' partial differential equation.
        """

        # Padding velocity field to enforce BCs (N) --> (N+6)
        left_padding = self.get_padding_left(u)
        right_padding = self.get_padding_right(u)
        u_pad = torch.cat([left_padding, u, right_padding], dim=0)

        # Calculating advection: d/dx(u^2/2) (N)
        advection = self.weno5(u_pad)

        # Calculating diffusion: 1/Re * d^2/dx^2(u) (N)
        diffusion = self.inv_Re * self.central_difference_laplacian(u_pad)
        
        # Calculating RHS: du/dt = -d/dx(u^2/2) + 1/Re * d^2/dx^2(u) (N)
        RHS = diffusion - advection

        return RHS
    
    def central_difference_laplacian(self, u_pad):
        """
        Uses a 6th order central difference Laplacian stencil to compute d^2u/dx^2.
        """
        # Stecil uses 3 points on each side (N+6) --> (N)
        laplacian = self.laplacian_coeff * (
              2 * u_pad[:-6]      # u_{i-3}
            - 27 * u_pad[1:-5]    # u_{i-2}
            + 270 * u_pad[2:-4]   # u_{i-1}
            - 490 * u_pad[3:-3]   # u_{i}
            + 270 * u_pad[4:-2]   # u_{i+1}
            - 27 * u_pad[5:-1]    # u_{i+2}
            + 2 * u_pad[6:]       # u_{i+3}
        ) * self.inv_dx2
        
        return laplacian

    def weno5(self, u_pad):
        """
        Uses Jiang-Shu WENO-5 to calculate the df/dx where f= u^2/2.
        """

        # Applies Lax-Friedrichs flux splitting on padded velocity field (N+6)
        f_plus, f_minus = self.flux_splitting(u_pad)

        # Uses WENO-5 interpolation with a left biased stencil on f+ to calculate f+_{i+1/2} (N+1)
        f_p_half = self.weno5_interpolation(
            f_plus[:-5], f_plus[1:-4], f_plus[2:-3], f_plus[3:-2], f_plus[4:-1]
        )
        
        # Uses WENO-5 interpolation with a right biased stencil on f- to calculate f-_{i-1/2} (N+1)
        f_m_half = self.weno5_interpolation(
            f_minus[5:], f_minus[4:-1], f_minus[3:-2], f_minus[2:-3], f_minus[1:-4]
        )

        # Calculates the total flux using f = (f+) + (f-) at all interfaces (N+1)
        f_half = f_p_half + f_m_half

        # Calculates the derivative of f w.r.t. x (N+1) --> (N)
        dF = (f_half[1:] - f_half[:-1]) * self.inv_dx

        return dF

    def weno5_interpolation(self, v1, v2, v3, v4, v5):
        """
        Core Jiang-Shu WENO-5 interpolation using a left-biased stencil (interpolates right face).
        Reverse inputs to use a right-biased stencil (interpolates left face).
        """
        # Smoothness indicators
        beta0 = (13.0 / 12.0) * (v1 - 2*v2 + v3)**2 + (1.0 / 4.0) * (v1 - 4*v2 + 3*v3)**2
        beta1 = (13.0 / 12.0) * (v2 - 2*v3 + v4)**2 + (1.0 / 4.0) * (v2 - v4)**2
        beta2 = (13.0 / 12.0) * (v3 - 2*v4 + v5)**2 + (1.0 / 4.0) * (3*v3 - 4*v4 + v5)**2

        # Linear weights
        d0, d1, d2 = 0.1, 0.6, 0.3

        # Unnormalized alpha weights
        alpha0 = d0 / (self.eps + beta0)**2
        alpha1 = d1 / (self.eps + beta1)**2
        alpha2 = d2 / (self.eps + beta2)**2

        sum_alpha = alpha0 + alpha1 + alpha2

        # Normalized WENO weights
        w0 = alpha0 / sum_alpha
        w1 = alpha1 / sum_alpha
        w2 = alpha2 / sum_alpha

        # Polynomial reconstructions
        p0 = (1.0 / 3.0) * v1 - (7.0 / 6.0) * v2 + (11.0 / 6.0) * v3
        p1 = -(1.0 / 6.0) * v2 + (5.0 / 6.0) * v3 + (1.0 / 3.0) * v4
        p2 = (1.0 / 3.0) * v3 + (5.0 / 6.0) * v4 - (1.0 / 6.0) * v5

        return w0 * p0 + w1 * p1 + w2 * p2
        
    def flux_splitting(self, u_pad):
        """
        Uses Lax-Friedrichs flux splitting to find f+ & f- where df+/du >= 0 and df-/du <= 0.
        """
        # LF algorithm says to calculate max|df/du|. Here f=u^2/2 --> df/du=u --> max|u|
        alpha = torch.max(torch.abs(u_pad))

        f_plus = 0.5 * ((0.5*u_pad**2) + (alpha*u_pad))
        f_minus = 0.5 * ((0.5*u_pad**2) - (alpha*u_pad))
        return f_plus, f_minus
    
    def get_padding_left(self, u):
        """
        Computes the values for the left padding cells to give the desired boundary condition at x=0.
        """
        # Copy values from right edge to left padding
        if self.left_bc_type == 'periodic':
            return u[-3:]

        # Linear anti-symmetric reflection: Forces the average of ghost and interior cells to equal the boundary value.
        elif self.left_bc_type == 'dirichlet':
            return 2 * self.left_bc_value - u[:3].flip(0)
        
        # Linear symmetric extrapolation: Sets ghost values such that the central difference across the interface matches the gradient.
        elif self.left_bc_type == 'neumann':
            offsets = torch.tensor([1.0, 3.0, 5.0], device=self.device, dtype=self.dtype)
            return u[:3].flip(0) - (offsets * self.left_bc_value * self.dx)

    def get_padding_right(self, u):
        """
        Computes the values for the right padding cells to give the desired boundary condition at x=L.
        """
        # Copy values from left edge to right padding
        if self.right_bc_type == 'periodic':
            return u[:3]
        
        # Linear anti-symmetric reflection: Forces the average of ghost and interior cells to equal the boundary value.
        elif self.right_bc_type == 'dirichlet':
            return 2 * self.right_bc_value - u[-3:].flip(0)
        
        # Linear symmetric extrapolation: Sets ghost values such that the central difference across the interface matches the gradient.
        elif self.right_bc_type == 'neumann':
            offsets = torch.tensor([1.0, 3.0, 5.0], device=self.device, dtype=self.dtype)
            return u[-3:].flip(0) + (offsets * self.right_bc_value * self.dx)


    


