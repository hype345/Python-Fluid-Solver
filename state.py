import torch

class State:
    def __init__(self, config):
        """
            Parent class of State1d, State2d, and State3d. 
            Not intended for direct use.
            Use State1d, State2d, or State3d instead.
        """
        # Storing device & dtype
        requested_device = config.get('device', "cuda")
        if requested_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            message = "Using device: cpu"
            if requested_device == "cuda":
                message = "No CUDA device available; defaulting to cpu"
            print(message)

        self.dtype = getattr(torch, config.get('dtype', "float32"))
        
        # Storing physical constants
        self.rho = config.get('rho', 1.2) # [kg/m^3]
        self.nu = config.get('nu', 1.5e-5) # [m^2/s]

        self.CHARACTERISTIC_LENGTH = config.get('CHARACTERISTIC_LENGTH', 1.0) # [m]
        self.CHARACTERISTIC_SPEED = config.get('CHARACTERISTIC_SPEED', 0.01) # [m/s]
        self.CHARACTERISTIC_DENSITY = self.rho # [kg/m^3]
        
        self.Re = (self.CHARACTERISTIC_SPEED * self.CHARACTERISTIC_LENGTH) / self.nu
        print(f"Reynolds Number: {self.Re:.1f}")

        # Flag for whether physical variables are dimensional or not
        self.is_dimensional = True

    def dimensionless(self):
        """
        Converts all internal parameters from a dimensional form to a dimensionless form.
        """
        if self.is_dimensional == True:
            self.rho = self.rho / self.CHARACTERISTIC_DENSITY
            self.nu = self.nu / (self.CHARACTERISTIC_SPEED * self.CHARACTERISTIC_LENGTH)

            self.velocity = self.velocity / self.CHARACTERISTIC_SPEED
            self.pressure = self.pressure / (self.CHARACTERISTIC_DENSITY * self.CHARACTERISTIC_SPEED**2)

            self.L = self.L / self.CHARACTERISTIC_LENGTH
            self.d = self.d / self.CHARACTERISTIC_LENGTH
            self.coordinates = self.coordinates / self.CHARACTERISTIC_LENGTH

            self.is_dimensional = False

    def dimensional(self):
        """
        Converts all internal parameters from a dimensionless form to a dimensional form.
        """
        if self.is_dimensional == False:
            self.rho = self.rho * self.CHARACTERISTIC_DENSITY
            self.nu = self.nu * (self.CHARACTERISTIC_SPEED * self.CHARACTERISTIC_LENGTH)

            self.velocity = self.velocity * self.CHARACTERISTIC_SPEED
            self.pressure = self.pressure * (self.CHARACTERISTIC_DENSITY * self.CHARACTERISTIC_SPEED**2)

            self.L = self.L * self.CHARACTERISTIC_LENGTH
            self.d = self.d * self.CHARACTERISTIC_LENGTH
            self.coordinates = self.coordinates * self.CHARACTERISTIC_LENGTH

            self.is_dimensional = True

class State3d(State):
    def __init__(self, config, u0=None, p0=None):
        """
        3D fluid system state used for solving the 3D incompressible Navier Stokes equations.

        Coordinate System:
            - Origin top left
            - z-axis depth (+ into screen)
            - y-axis height (+ down)
            - x-axis width (+ right)
            - Right handed system

        Internal indexing (z, y, x).

        Internal coordinates are cell centered representing [0, L].

        Parameters:
            - config {dict}:
                • 'dtype' (pytorch data type used) {string}
                • 'device' (pytorch device used) {string}
                • 'Nz' (dim-0 cell count) {int}
                • 'Ny' (dim-1 cell count) {int}
                • 'Nx' (dim-2 cell count) {int}
                • 'DOMAIN_DEPTH' (z) [m] {float}
                • 'DOMAIN_HEIGHT' (y) [m] {float}
                • 'DOMAIN_WIDTH' (x) [m] {float}
                • 'rho' (fluid density) [kg/m^3] {float}
                • 'nu' (fluid kinematic viscosity) [m^2/s] {float}
                • 'CHARACTERISTIC_LENGTH' (dimensionlal length conversion factor) [m] {float}
                • 'CHARACTERISTIC_SPEED' (dimensionlal speed conversion factor) [m/s] {float}
        """
        super().__init__(config)

        # Getting cell counts & physical lengths
        Nz = config.get('Nz', 256)
        Ny = config.get('Ny', 256)
        Nx = config.get('Nx', 256)

        DOMAIN_DEPTH = config.get('DOMAIN_DEPTH', 1.0) # [m] z-direction (away)
        DOMAIN_HEIGHT = config.get('DOMAIN_HEIGHT', 1.0) # [m] y-direction (down)
        DOMAIN_WIDTH = config.get('DOMAIN_WIDTH', 1.0) # [m] x-direction (right)

        # Initializing & storing dimensional fields
        if u0 is not None:
            self.velocity = u0.to(device=self.device, dtype=self.dtype) # [m/s]
            assert u0.shape == (3, self.Nz, self.Ny, self.Nx), f"The shape of u0 must be (3, Nz={self.Nz}, Ny={self.Ny}, Nx={self.Nx}) to match with Nz, Ny, and Nx from config"
        else:
            self.velocity = torch.zeros((3, Nz, Ny, Nx), dtype=self.dtype, device=self.device) # [m/s]

        if p0 is not None:
            self.pressure = p0.to(device=self.device, dtype=self.dtype) # [Kg/m*s^2]
            assert p0.shape == (self.Ny, self.Nx), f"The shape of p0 must be (Nz={self.Nz}, Ny={self.Ny}, Nx={self.Nx}) to match with Nz, Ny, and Nx from config"
        else:
            self.pressure = torch.zeros((Ny, Nx), dtype=self.dtype, device=self.device) # [Kg/m*s^2]

        # Initializing & Storing dimensional cell centered coordinates
        dz = DOMAIN_DEPTH / Nz
        dy = DOMAIN_HEIGHT / Ny
        dx = DOMAIN_WIDTH / Nx
        
        z = torch.linspace(0, DOMAIN_DEPTH - dz, steps=Nz, dtype=self.dtype, device=self.device) + (dz / 2.0)
        y = torch.linspace(0, DOMAIN_HEIGHT - dy, steps=Ny, dtype=self.dtype, device=self.device) + (dy / 2.0)
        x = torch.linspace(0, DOMAIN_WIDTH - dx, steps=Nx, dtype=self.dtype, device=self.device) + (dx / 2.0)
        
        Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')

        self.coordinates = torch.stack((Z, Y, X), dim=0)

        # Storing dimensional grid parameters in single tensor format 
        self.N = torch.tensor([Nz, Ny, Nx], dtype=torch.int32, device=self.device)
        self.L = torch.tensor([DOMAIN_DEPTH, DOMAIN_HEIGHT, DOMAIN_WIDTH], dtype=self.dtype, device=self.device)
        self.d = torch.tensor([dz, dy, dx], dtype=self.dtype, device=self.device)

        # Storing dimension of state
        self.dim = 3

class State2d(State):
    def __init__(self, config, u0=None, p0=None):
        """
        2D fluid system state used for solving the 2D incompressible Navier Stokes equations.

        Coordinate System:
            - Origin top left
            - y-axis height (+ down)
            - x-axis width (+ right)
            - Right handed system

        Internal indexing (y, x).

        Internal coordinates are cell centered representing [0, L].

        Parameters:
            - config {dict}:
                • 'dtype' (pytorch data type used) {string}
                • 'device' (pytorch device used) {string}
                • 'Ny' (dim-0 cell count) {int}
                • 'Nx' (dim-1 cell count) {int}
                • 'DOMAIN_HEIGHT' (y) [m] {float}
                • 'DOMAIN_WIDTH' (x) [m] {float}
                • 'rho' (fluid density) [kg/m^3] {float}
                • 'nu' (fluid kinematic viscosity) [m^2/s] {float}
                • 'CHARACTERISTIC_LENGTH' (dimensionlal length conversion factor) [m] {float}
                • 'CHARACTERISTIC_SPEED' (dimensionlal speed conversion factor) [m/s] {float}
        """
        super().__init__(config)

        # Getting cell counts & physical lengths
        Ny = config.get('Ny', 256)
        Nx = config.get('Nx', 256)

        DOMAIN_HEIGHT = config.get('DOMAIN_HEIGHT', 1.0) # [m] y-direction (down)
        DOMAIN_WIDTH = config.get('DOMAIN_WIDTH', 1.0) # [m] x-direction (right)

        # Initializing & storing dimensional fields
        if u0 is not None:
            self.velocity = u0.to(device=self.device, dtype=self.dtype) # [m/s]
            assert u0.shape == (2, self.Ny, self.Nx), f"The shape of u0 must be (2, Ny={self.Ny}, Nx={self.Nx}) to match with Ny and Nx from config"
        else:
            self.velocity = torch.zeros((2, Ny, Nx), dtype=self.dtype, device=self.device) # [m/s]

        if p0 is not None:
            self.pressure = p0.to(device=self.device, dtype=self.dtype) # [Kg/m*s^2]
            assert p0.shape == (self.Ny, self.Nx), f"The shape of p0 must be (Ny={self.Ny}, Nx={self.Nx}) to match with Ny and Nx from config"
        else:
            self.pressure = torch.zeros((Ny, Nx), dtype=self.dtype, device=self.device) # [Kg/m*s^2]

        # Initializing & Storing dimensional cell centered coordinates
        dy = DOMAIN_HEIGHT / Ny
        dx = DOMAIN_WIDTH / Nx
        
        y = torch.linspace(0, DOMAIN_HEIGHT - dy, steps=Ny, dtype=self.dtype, device=self.device) + (dy / 2.0)
        x = torch.linspace(0, DOMAIN_WIDTH - dx, steps=Nx, dtype=self.dtype, device=self.device) + (dx / 2.0)
        
        Y, X = torch.meshgrid(y, x, indexing='ij')

        self.coordinates = torch.stack((Y, X), dim=0)

        # Storing dimensional grid parameters in single tensor format 
        self.N = torch.tensor([Ny, Nx], dtype=torch.int32, device=self.device)
        self.L = torch.tensor([DOMAIN_HEIGHT, DOMAIN_WIDTH], dtype=self.dtype, device=self.device)
        self.d = torch.tensor([dy, dx], dtype=self.dtype, device=self.device)

        # Storing dimension of state
        self.dim = 2

class State1d(State):
    def __init__(self, config, u0=None):
        """
        1D fluid system state used for solving the 1D viscous Burgers' equation.

        Coordinate System:
            - x-axis width (+ right)
            - Right handed system

        Internal indexing (x).

        Internal coordinates are cell centered representing [0, L].

        Parameters:
            - config {dict}:
                • 'dtype' (pytorch data type used) {string}
                • 'device' (pytorch device used) {string}
                • 'Nx' (dim-0 cell count) {int}
                • 'DOMAIN_WIDTH' (x) [m] {float}
                • 'nu' (fluid kinematic viscosity) [m^2/s] {float}
                • 'CHARACTERISTIC_LENGTH' (dimensionlal length conversion factor) [m] {float}
                • 'CHARACTERISTIC_SPEED' (dimensionlal speed conversion factor) [m/s] {float}
            -u0 (initial velocity field) [m/s] {torch.tensor}
        """
        super().__init__(config)

        # Removing density class variables they do not exist for 1d Burgers'
        del self.rho
        del self.CHARACTERISTIC_DENSITY

        # Getting cell counts & physical lengths
        Nx = config.get('Nx', 256)

        DOMAIN_WIDTH = config.get('DOMAIN_WIDTH', 1.0) # [m] x-direction (right)

        # Initializing & storing dimensional fields
        if u0 is not None:
            self.velocity = u0.to(device=self.device, dtype=self.dtype) # [m/s]
            assert self.Nx == u0.shape[0], f"The shape of initial condition must be Nx={self.Nx} to match Nx from config"
        else:
            self.velocity = torch.zeros(Nx, dtype=self.dtype, device=self.device) # [m/s]

        # Initializing & Storing dimensional cell centered coordinates
        dx = DOMAIN_WIDTH / Nx
        
        self.coordinates = torch.linspace(0, DOMAIN_WIDTH - dx, steps=Nx, dtype=self.dtype, device=self.device) + (dx / 2.0)

        # Storing dimensional grid parameters in single tensor format 
        self.N = torch.tensor([Nx], dtype=torch.int32, device=self.device)
        self.L = torch.tensor([DOMAIN_WIDTH], dtype=self.dtype, device=self.device)
        self.d = torch.tensor([dx], dtype=self.dtype, device=self.device)

        # Storing dimension of state
        self.dim = 1

    # Redefining conversions for 1d because there is no longer pressure
    def dimensionless(self):
        """
        Converts all internal parameters from a dimensional form to a dimensionless form.
        """
        if self.is_dimensional == True:
            self.nu = self.nu / (self.CHARACTERISTIC_SPEED * self.CHARACTERISTIC_LENGTH)

            self.velocity = self.velocity / self.CHARACTERISTIC_SPEED

            self.L = self.L / self.CHARACTERISTIC_LENGTH
            self.d = self.d / self.CHARACTERISTIC_LENGTH
            self.coordinates = self.coordinates / self.CHARACTERISTIC_LENGTH

            self.is_dimensional = False

    def dimensional(self):
        """
        Converts all internal parameters from a dimensionless form to a dimensional form.
        """
        if self.is_dimensional == False:
            self.nu = self.nu * (self.CHARACTERISTIC_SPEED * self.CHARACTERISTIC_LENGTH)

            self.velocity = self.velocity * self.CHARACTERISTIC_SPEED

            self.L = self.L * self.CHARACTERISTIC_LENGTH
            self.d = self.d * self.CHARACTERISTIC_LENGTH
            self.coordinates = self.coordinates * self.CHARACTERISTIC_LENGTH

            self.is_dimensional = True


   