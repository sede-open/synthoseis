# Synthoseis Repository Research Documentation

## Overview

Synthoseis is an open-source, Python-based tool for generating pseudo-random synthetic seismic data designed to train deep learning networks for seismic interpretation. This document provides an extensive analysis of the codebase, focusing on the key aspects of the synthetic seismic data generation pipeline.

**Reference Paper:** Merrifield et al. (2022) - "Synthetic seismic data for training deep learning networks", Interpretation Journal

---

## 1. THE EXACT PIPELINE FOR CREATING SYNTHETIC SEISMIC DATA

### Main Entry Point
**File:** `main.py:18-83`

The complete workflow is orchestrated in the `build_model()` function:

### Detailed Step-by-Step Pipeline

#### **Step 1: Configuration and Setup** (Lines 21-24)
```python
p = Parameters(user_json, runid=run_id, test_mode=test_mode)
p.setup_model(rpm_factors=rpm_factors)
p.hdf_setup(os.path.join(p.temp_folder, "model_data.hdf"))
```
- Loads user parameters from JSON config file
- Sets up HDF5 storage for intermediate data
- Creates output directories
- **Implementation:** `datagenerator/Parameters.py`

#### **Step 2: Build Unfaulted Depth Horizons** (Lines 26-27)
```python
depth_maps, onlap_list, fan_list, fan_thicknesses = build_unfaulted_depth_maps(p)
facies = create_facies_array(p, depth_maps, onlap_list, fan_list)
```
- Builds a stack of 2D depth horizons from base upward
- Each layer has pseudo-random thickness (between `thickness_min` and `thickness_max`)
- Assigns lithology (sand/shale) to each layer
- Creates basin floor fans and onlap surfaces
- **Implementation:** `datagenerator/Horizons.py:1450-1577`

#### **Step 3: Build Unfaulted Geologic Models** (Lines 29-31)
```python
geo_models = Geomodel(p, depth_maps, onlap_list, facies)
geo_models.build_unfaulted_geomodels()
```
- Converts 2D horizon stack into 3D geologic age volume
- Each voxel assigned a relative geologic age (layer index)
- Creates onlap segmentation volumes
- **Implementation:** `datagenerator/Geomodels.py:87-123`

#### **Step 4: Generate and Apply Faults** (Lines 33-38)
```python
f = Faults(p, depth_maps, onlap_list, geo_models, fan_list, fan_thicknesses)
f.apply_faulting_to_geomodels_and_depth_maps()
f.build_faulted_property_geomodels(facies)
```
- Generates random fault patterns (various styles: self-branching, stair-case, relay-ramp, horst-graben)
- Calculates fault displacements using ellipsoid geometry
- Applies displacements to geologic age and depth volumes
- Builds faulted lithology and net-to-gross volumes
- **Implementation:** `datagenerator/Faults.py:107-289, 756-1244`

#### **Step 5: Create Closures (Traps) and Fill with Fluids** (Lines 40-43)
```python
closures = Closures(p, f, facies, onlap_list)
closures.create_closures()
closures.write_closure_volumes_to_disk()
```
- Uses flood-fill algorithm to identify structural closures (4-way, 3-way faulted, stratigraphic)
- Randomly fills closures with oil, gas, or brine
- Creates fluid indicator volumes (`oil_closures`, `gas_closures`)
- **Implementation:** `datagenerator/Closures.py:868-1167`

#### **Step 6: Build Elastic Property Models** (Line 54)
```python
seismic.build_elastic_properties("inv_vel")
```
- Calculates 3D volumes of density (ρ), P-wave velocity (Vp), and S-wave velocity (Vs)
- Uses depth-dependent trend lines from rock physics model
- Applies fluid substitution in hydrocarbon-filled closures
- **Implementation:** `datagenerator/Seismic.py:83-97, 693-1090`

#### **Step 7: Compute Seismic Volumes** (Line 55)
```python
seismic.build_seismic_volumes()
```
- Calculates reflection coefficients using Zoeppritz equation for each incident angle
- Adds angle-dependent weighted noise
- Applies Butterworth bandpass filtering or wavelet convolution
- Applies geophysical augmentations (lateral smoothing, amplitude balancing)
- Writes final seismic volumes to disk
- **Implementation:** `datagenerator/Seismic.py:183-261`

### Complete Pipeline Summary Table

| Step | Operation | File | Key Lines | Output |
|------|-----------|------|-----------|--------|
| 1 | Parameter setup | `Parameters.py` | 18-24 | HDF setup, directories |
| 2 | Horizon generation | `Horizons.py` | 1450-1577 | 2D depth maps, facies array |
| 3 | Geologic age model | `Geomodels.py` | 87-123 | 3D age volume, onlap segments |
| 4 | Fault generation | `Faults.py` | 756-1244 | Faulted age, depth, lithology |
| 5 | Closure detection | `Closures.py` | 868-1167 | Fluid indicator volumes |
| 6 | Elastic properties | `Seismic.py` | 693-1090 | 3D ρ, Vp, Vs volumes |
| 7 | Seismic modeling | `Seismic.py` | 183-380 | Angle-stack seismic data |

---

## 2. HOW THE GEOLOGIC ELASTIC MODEL IS GENERATED

### 2.1 Rock Property Trend Lines Definition

**File:** `rockphysics/rpm_example.py:35-80`

The rock property trends are **1D depth-dependent polynomial functions** for each facies/fluid combination:

#### Trend Definitions (Polynomial Functions)

**Shale Properties:**
```python
@staticmethod
def calc_shale_rho(z):  # Density in g/cc
    return 7.7e-12 * z**3 + -8.8e-08 * z**2 + 0.0004 * z + 1.957

@staticmethod
def calc_shale_vp(z):  # P-wave velocity in m/s
    return -0.00013 * z**2 + 1.13 * z + 1580

@staticmethod
def calc_shale_vs(z):  # S-wave velocity in m/s
    return -0.0001 * z**2 + 0.96 * z + 279
```

**Brine Sand Properties:**
```python
@staticmethod
def calc_brine_sand_rho(z):
    return -7.8e-09 * z**2 + 0.00012 * z + 2.021

@staticmethod
def calc_brine_sand_vp(z):
    return -1.34e-05 * z**2 + 0.49 * z + 2317

@staticmethod
def calc_brine_sand_vs(z):
    return -1.0785e-05 * z**2 + 0.391 * z + 1007
```

**Oil Sand Properties:**
```python
@staticmethod
def calc_oil_sand_rho(z):
    return -9.23e-09 * z**2 + 0.00014 * z + 1.916

@staticmethod
def calc_oil_sand_vp(z):
    return -8.876e-06 * z**2 + 0.505 * z + 1998

@staticmethod
def calc_oil_sand_vs(z):
    return -1.126e-05 * z**2 + 0.391 * z + 1036
```

**Gas Sand Properties:**
```python
@staticmethod
def calc_gas_sand_rho(z):
    return -1.818e-08 * z**2 + 0.000247 * z + 1.612

@staticmethod
def calc_gas_sand_vp(z):
    return -3.216e-06 * z**2 + 0.4796 * z + 1996

@staticmethod
def calc_gas_sand_vs(z):
    return -1.0687e-05 * z**2 + 0.3662 * z + 1135
```

**Key Observations:**
- Trends are polynomial functions (mostly quadratic or cubic) in depth
- Separate trends for each fluid type (brine, oil, gas) in sands
- Oil and gas sands have lower density and Vp but similar Vs compared to brine sands
- This reflects the physical effect of hydrocarbon substitution on rock properties

### 2.2 Converting 1D Trends to 3D Parameter Cubes

**File:** `datagenerator/Seismic.py:693-1090`

**Key Method:** `build_property_models_randomised_depth()`

#### The Transformation Process:

**Step 1: Layer-by-Layer Processing** (Lines 753-754)
```python
for z in range(1, integer_faulted_age.max()):
    __i, __j, __k = np.where(integer_faulted_age == z)
```
- Loops through each unique geologic age (layer) value
- Finds all voxels belonging to that layer

**Step 2: Depth Randomization for Heterogeneity** (Lines 766-776, 1098-1119)
```python
delta_z_layer = self.get_delta_z_layer(z, layer_half_range, __k)
delta_z_rho, delta_z_vp, delta_z_vs = self.get_delta_z_properties(z, property_half_range)
```
- Adds random depth shifts to create lateral heterogeneity
- Two levels of randomization:
  - `delta_z_layer`: Random shift applied to entire layer (35-125 samples from triangular distribution)
  - `delta_z_property`: Independent random shifts for ρ, Vp, Vs (5-20 samples)
- Prevents perfectly flat, unrealistic property variations

**Step 3: Query Trend Functions** (Lines 778-810, 803-810)
```python
_k_rho = np.array(k + delta_z_layer + delta_z_rho).clip(0, cube_shape[-1] - 10)
mixed_properties = self.calculate_shales(
    xyz=(i, j, k),
    shifts=(_k_rho, _k_vp, _k_vs),
    props=mixed_properties,
    rpm=rpm,
    depth=depth,
    z=z,
)
```
- Uses randomized depth indices to query trend functions
- Each voxel gets properties based on its **actual depth** plus random offset
- This creates realistic lateral variations in rock properties

**Step 4: Apply Fluid Substitution** (Lines 813-999)
```python
# Brine sands (no fluid substitution)
i, j, k = np.where((net_to_gross > 0.0) & (oil_closures == 0.0) & (gas_closures == 0.0))

# Oil sands - use oil trends
i, j, k = np.where((net_to_gross > 0.0) & (oil_closures == 1.0) & (gas_closures == 0.0))
mixed_properties = self.calculate_sands(..., fluid="oil", ...)

# Gas sands - use gas trends
i, j, k = np.where((net_to_gross > 0.0) & (gas_closures == 1.0))
mixed_properties = self.calculate_sands(..., fluid="gas", ...)
```
- Checks fluid indicator volumes (`oil_closures`, `gas_closures`)
- Uses appropriate trend functions based on fluid type

#### Summary of the Trend → 3D Cube Transformation:

```
Input: 1D polynomial trends f(z) → Output: 3D cube of properties

Process:
1. For each voxel (i, j, k):
   - Get actual depth from faulted_depth volume: z_actual = depth[i,j,k]
   - Apply random offset: z_random = z_actual + delta_z_random
   - Query trend function: property[i,j,k] = f(z_random)
2. For voxels with mixed lithology (Net-to-Gross between 0 and 1):
   - Calculate shale properties at z_random
   - Calculate sand properties at z_random (with appropriate fluid)
   - Mix using Backus or inverse-velocity averaging
```

### 2.3 Facies/Lithology Assignment

**File:** `datagenerator/Horizons.py:1331-1448`

#### Two Methods for Facies Assignment:

**Method 1: Binomial Distribution** (Lines 1339-1345)
```python
def sand_shale_facies_binomial_dist(self):
    sand_layer = np.random.binomial(1, self.cfg.sand_layer_pct, size=self.max_layers)
    self.facies = np.hstack((np.array((-1.0)), sand_layer))  # -1 for water
```
- Simple random selection based on sand percentage
- Each layer independently assigned as sand or shale
- Quick but unrealistic (no spatial correlation)

**Method 2: Markov Chain Process** (Lines 1390-1448) - **Preferred Method**
```python
class MarkovChainFacies:
    def _transition_matrix(self):
        beta = 1 / self.sand_thickness  # Probability of leaving sand state
        alpha = sand_fraction / (sand_thickness * (1 - sand_fraction))
        self.transition = np.array([[1-alpha, alpha], [beta, 1-beta]])
    
    def generate_states(self, current_state, num):
        # Generate sequence of facies using Markov transitions
```
- Creates realistic layer sequences
- Transition probabilities control:
  - `alpha`: Probability of transitioning from shale to sand
  - `beta`: Probability of transitioning from sand to shale
- Results in sand layers with realistic thickness distribution
- Better represents actual geologic layering patterns

#### Net-to-Gross (N/G) Variation

**File:** `Horizons.py:612-686`

- N/G represents fraction of sand in each voxel
- For pure sand: N/G = 1.0
- For pure shale: N/G = 0.0
- For mixed lithology: N/G varies between 0 and 1
- Variable shale N/G can be enabled in config

### 2.4 Property Mixing for Partial Voxels

**File:** `rockphysics/RockPropertyModels.py:80-134`

When voxels have N/G between 0 and 1 (mixed lithology), properties must be mixed:

#### **Method 1: Inverse Velocity Mixing** (Lines 99-111) - Default
```python
def inverse_velocity_mixing(self):
    self.rho = arithmetic_mean(shales.rho, sands.rho, net_to_gross)
    self.vp = harmonic_mean(shales.vp, sands.vp, net_to_gross)  # Slowness averaging
    self.vs = harmonic_mean(shales.vs, sands.vs, net_to_gross)
```
- Density: Arithmetic average
- Velocities: Harmonic average (slowness averaging)
- Physically appropriate for vertical ray paths through layered media

#### **Method 2: Backus Moduli Mixing** (Lines 113-133)
```python
def backus_moduli_mixing(self):
    lambda_mix = harmonic_mean(shales.lam, sands.lam, net_to_gross)
    mu_mix = harmonic_mean(shales.mu, sands.mu, net_to_gross)
    self.rho = arithmetic_mean(shales.rho, sands.rho, net_to_gross)
    self.vp = moduli.vp(lam=lambda_mix, mu=mu_mix, rho=self.rho)
    self.vs = moduli.vs(mu=mu_mix, rho=self.rho)
```
- Uses Backus averaging for finely layered media
- More physically rigorous for anisotropic effects
- Mixes elastic moduli (λ, μ) using harmonic average

---

## 3. HOW THE FORWARD SEISMIC PROBLEM IS COMPUTED

### 3.1 Physics/Algorithm Used: **1D Convolutional Modeling**

**Method:** Convolutional seismic modeling with Zoeppritz equation

This is NOT full wave-equation modeling (no wave propagation physics). It's a simplified but effective approach for generating training data.

#### Why This Method Was Chosen:

1. **Computational Efficiency**: Much faster than full wave-equation modeling (seconds vs hours per model)
2. **Sufficiently Realistic**: Captures key seismic features for DL training:
   - AVO effects (angle-dependent amplitudes)
   - Wavelet bandwidth effects
   - Noise characteristics
3. **Scalability**: Can generate thousands of training examples quickly
4. **Controllability**: Easy to adjust parameters (angles, bandwidth, noise)

### 3.2 Reflection Coefficient Calculation

**File:** `datagenerator/Seismic.py:1379-1475`

#### Three Methods Available in the `RFC` Class:

**Method 1: Normal Incidence Reflectivity** (Lines 1398-1401)
```python
def normal_incidence_rfc(self):
    z0 = self.rho_upper * self.vp_upper  # Acoustic impedance upper
    z1 = self.rho_lower * self.vp_lower  # Acoustic impedance lower
    return (z1 - z0) / (z1 + z0)
```
- Simple impedance-based formula
- Only valid for θ = 0°

**Method 2: Shuey Approximation** (Lines 1403-1424)
```python
def shuey(self):
    """3-term Shuey AVO approximation"""
    r0 = 0.5 * (d_vp/avg_vp + d_rho/avg_rho)  # Normal incidence term
    g = 0.5 * (d_vp/avg_vp) - 2*avg_vs**2/avg_vp**2 * (d_rho/avg_rho + 2*d_vs/avg_vs)
    f = 0.5 * (d_vp/avg_vp)
    return r0 + g*sin_squared + f*(tan_squared - sin_squared)
```
- Approximate AVO equation
- Good for angles up to ~30-40°
- Commonly used in industry AVO analysis

**Method 3: Zoeppritz Equation** (Lines 1426-1467) - **Primary Method Used**
```python
def zoeppritz_reflectivity(self):
    """Calculate PP reflectivity using exact Zoeppritz equation"""
    theta = self._degrees_to_radians(dtype="complex")
    p = np.sin(theta) / self.vp_upper  # Ray parameter
    
    # Calculate transmission and reflection angles
    theta2 = np.arcsin(p * self.vp_lower)  # P-wave transmission angle
    phi1 = np.arcsin(p * self.vs_upper)   # Converted S-wave reflection angle
    phi2 = np.arcsin(p * self.vs_lower)   # S-wave transmission angle
    
    # Compute Zoeppritz matrix elements (exact solution)
    a = self.rho_lower * (1 - 2*np.sin(phi2)**2) - self.rho_upper * (1 - 2*np.sin(phi1)**2)
    b = self.rho_lower * (1 - 2*np.sin(phi2)**2) + 2*self.rho_upper * np.sin(phi1)**2
    c = self.rho_upper * (1 - 2*np.sin(phi1)**2) + 2*self.rho_lower * np.sin(phi2)**2
    d = 2*(self.rho_lower * self.vs_lower**2 - self.rho_upper * self.vs_upper**2)
    
    # ... complex matrix solution for PP reflectivity
    zoep_pp = (f * (b*(cos(theta)/vp_upper) - c*(cos(theta2)/vp_lower)) 
               - h * p**2 * (...)) * (1/d)
    
    return np.real(zoep_pp)
```
- Exact solution for plane-wave reflection at an interface
- Accounts for P-to-S conversions
- Valid for all angles (up to critical angle)
- Complex values indicate critical angle exceeded (energy loss)

#### Main RFC Calculation Loop (Lines 308-380)

```python
def create_rfc_volumes(self):
    theta = np.asarray(self.angles).reshape((-1, 1))
    zoep = np.zeros((nx, ny, nz-1, n_angles), dtype='complex64')
    
    # Trace-by-trace calculation
    for i in trange(self.rho.shape[0]):  # xlines
        for j in range(self.rho.shape[1]):  # inlines
            rho1, rho2 = _rho[i,j,:-1], _rho[i,j,1:]
            vp1, vp2 = _vp[i,j,:-1], _vp[i,j,1:]
            vs1, vs2 = _vs[i,j,:-1], _vs[i,j,1:]
            rfc = RFC(vp1, vs1, rho1, vp2, vs2, rho2, theta)
            zoep[i,j,:,:] = rfc.zoeppritz_reflectivity().T
    
    # Remove imaginary parts (critical angle artifacts)
    zoep = np.real(zoep).astype('float64')
```
- Computes RFC for every interface in every trace
- RFC calculated for all incident angles simultaneously
- Results stored as 4D volume (angle × x × y × z)

### 3.3 Wavelet/Convolution Application

#### **Method 1: Butterworth Bandpass Filter** (Lines 472-530, 1477-1499)

```python
def apply_bandlimits(self, data, frequencies=None):
    """Apply Butterworth bandpass filter to seismic data"""
    low = np.random.uniform(self.cfg.bandwidth_low[0], self.cfg.bandwidth_low[1])
    high = np.random.uniform(self.cfg.bandwidth_high[0], self.cfg.bandwidth_high[1])
    order = self.cfg.bandwidth_ord
    
    b, a = derive_butterworth_bandpass(low, high, dt, order=4)
    y = filtfilt(b, a, data, method="gust", irlen=approx_impulse_len)
    return y

def derive_butterworth_bandpass(lowcut, highcut, digitisation, order=4):
    nyq = 0.5 * digitisation
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
```
- Uses scipy's Butterworth filter design
- Forward-backward filtering (filtfilt) for zero phase shift
- Bandwidth randomized from config ranges
- Example: 10-60 Hz typical

#### **Method 2: Wavelet Convolution** (Lines 99-155)

```python
def bandlimit_volumes_wavelets(self, n_wavelets=10):
    # Load pre-computed filter specifications
    fs_nr = np.load(self.cfg.wavelets[0])  # Near stack filter spec
    fs_md = np.load(self.cfg.wavelets[1])  # Mid stack filter spec
    fs_fr = np.load(self.cfg.wavelets[2])  # Far stack filter spec
    
    # Generate wavelets with randomized amplitude spectra
    low_freq_pctile = np.random.uniform(0, 100)
    mid_freq_pctile = np.random.uniform(0, 100)
    high_freq_pctile = np.random.uniform(0, 100)
    
    f_nr, a_nr, w_nr = generate_wavelet(fs_nr, low_med_high_percentiles)
    
    # Convolve with reflectivity
    for idx in range(rfc_bandlimited_wavelet.shape[0]):
        rfc_bandlimited_wavelet[idx,...] = apply_wavelet(
            self.rfc_noise_added[idx,...], wavelets[idx]
        )
```
- Pre-computed wavelet filter specifications loaded from .npy files
- Randomized amplitude spectra using percentile interpolation
- Convolution via FFT for efficiency

#### Wavelet Generation (wavelets.py:136-182)

```python
def generate_wavelet(filter_spec, percentiles):
    # Uses Ricker wavelet as basis
    # Modifies spectrum based on filter_spec and random percentiles
    # Returns frequency response, amplitude spectrum, and time-domain wavelet
```

### 3.4 Angle Stack Generation

**File:** `datagenerator/Seismic.py:308-380`

```python
# Input angles from config (e.g., [7, 15, 24] degrees)
self.angles = self.cfg.incident_angles

# Each angle gets its own RFC cube
# Then each is processed independently:
- Apply bandpass/wavelet
- Add noise (angle-dependent)
- Apply augmentations
```

#### Full Stack Creation (Lines 173-180, 547-549)

```python
def stack_substacks(cube_list):
    """Stack cubes by taking mean across angle dimension"""
    return np.mean(cube_list, axis=0)

# Create fullstack from near, mid, far stacks
rfc_fullstack = self.stack_substacks([near, mid, far])
```

### 3.5 Noise Addition

**File:** `datagenerator/Seismic.py:402-470`

#### Hilterman-Weighted Noise Model:

```python
def add_weighted_noise(self, depth_maps):
    # Create base noise cubes
    noise_0deg = self.noise_3d(cube_shape)  # Noise for 0° angle
    noise_45deg = self.noise_3d(cube_shape)  # Noise for 45° angle
    
    # Angle-dependent weighting (Hilterman model)
    for x, ang in enumerate(self.angles):
        weighted_noise = noise_0deg * (cos(ang)**2) + noise_45deg * (sin(ang)**2)
        
        # Scale to achieve target S/N ratio
        data_std = data_below_seafloor.std()
        noise_std = weighted_noise.std()
        std_ratio = 10 ** (snr_db / 20.0)  # Convert dB to linear ratio
        noise3d_cubes[x,...] = weighted_noise * (data_std / noise_std) / std_ratio
```

**Physical Basis:**
- Noise amplitude varies with angle
- Near angles: dominated by random amplitude noise (cos² weight)
- Far angles: dominated by coherent noise (sin² weight)
- S/N ratio controlled from config (triangular distribution)

### 3.6 Complete Forward Modeling Summary

```
Forward Modeling Pipeline:

Input: 3D volumes of ρ, Vp, Vs + Incident angles

1. Compute Reflection Coefficients:
   For each trace (i, j):
     For each interface k:
       For each angle θ:
         RPP[i,j,k,θ] = Zoeppritz(ρ1, Vp1, Vs1, ρ2, Vp2, Vs2, θ)

2. Add Noise:
   Create weighted noise: N[θ] = N₀·cos²(θ) + N₄₅·sin²(θ)
   Scale to achieve target S/N ratio
   Add: RFC_noisy[θ] = RPP[θ] + N[θ]

3. Bandlimit/Convolve:
   Apply Butterworth bandpass filter (randomized bandwidth)
   OR convolve with pre-designed wavelets
   Result: seismic[θ]

4. Post-process:
   Apply lateral smoothing filter
   Scale amplitudes (normalize to 100 std)
   Apply augmentations (stretch, RMO)

Output: Angle-stack seismic volumes + Fullstack
```

---

## 4. GENERAL PRINCIPLES OF THE CODE ARCHITECTURE

### 4.1 Module Structure

| Module | File | Purpose | Key Methods |
|--------|------|---------|-------------|
| **Parameters** | `Parameters.py` | Configuration management | `setup_model()`, `hdf_setup()` |
| **Horizons** | `Horizons.py` | Layer generation | `build_unfaulted_depth_maps()`, `create_facies_array()` |
| **Geomodels** | `Geomodels.py` | 3D geologic model | `build_unfaulted_geomodels()`, `create_geologic_age_3d()` |
| **Faults** | `Faults.py` | Fault generation | `build_faults()`, `apply_faulting()` |
| **Closures** | `Closures.py` | Trap detection | `create_closures()`, `flood_fill()` |
| **Seismic** | `Seismic.py` | Forward modeling | `build_elastic_properties()`, `create_rfc_volumes()` |
| **RockPhysics** | `RockPropertyModels.py` | Property trends | `select_rpm()`, mixing methods |
| **RPMExample** | `rpm_example.py` | Trend definitions | Polynomial depth trends |

### 4.2 Design Patterns

#### **Borg Pattern** (Parameters.py:18-24)
```python
class _Borg:
    _shared_state = {}  # All instances share same state
    
class Parameters(_Borg):
    def __init__(self):
        self.__dict__ = self._shared_state
```
- Ensures single source of truth for parameters
- All modules access same configuration object
- Avoids passing parameters repeatedly

#### **Factory Pattern** (Faults.py:736-754)
```python
def fault_parameters(self):
    fault_mode = self._get_fault_mode()
    return fault_mode()  # Returns appropriate fault generator
```
- Different fault styles generated by different methods
- fault_mode determines which generator to use
- Styles: random, self-branching, stair-case, relay-ramp, horst-graben

#### **Strategy Pattern** (RockPropertyModels.py:143-167)
```python
def select_rpm(cfg):
    if cfg.project == "example":
        rpm = RPMExample(cfg)
    elif cfg.project == "tagilsk":
        rpm = RPMTagilsk(cfg)
    return rpm
```
- Allows swapping rock physics models
- User selects RPM via config file
- Each RPM defines its own depth trends

#### **Multiple Inheritance**
```python
class Faults(Horizons, Geomodel):  # Inherits from both
class SeismicVolume(Geomodel):     # Inherits geologic model methods
class Closures(Horizons, Geomodel, Parameters):  # Triple inheritance
```
- Shares functionality across related classes
- Avoids code duplication

### 4.3 Data Storage Architecture

#### HDF5-Based Storage (Parameters.py:953-1010)

```python
def hdf_setup(self, hdf_name):
    self.h5file = tables.open_file(hdf_filename, "w")
    self.h5file.create_group("/", "ModelData")

def hdf_init(self, dset_name, shape):
    atom = tables.FloatAtom()  # Or UInt8Atom, IntAtom
    new_array = self.h5file.create_carray(group, dset_name, atom, shape)
```

**Benefits:**
- Chunked array storage (efficient for large 3D volumes)
- Blosc compression (fast, low memory)
- All intermediate volumes accessible during computation
- Final cleanup deletes temp folder

### 4.4 Key Geologic Concepts Implemented

#### **Geologic Age Model** (Geomodels.py:154-240)
- Concept: Each voxel assigned relative geologic age
- Age = horizon index at that depth
- Interpolation between horizons for smooth age variation
- Used for layer-based property assignment

#### **Fault Displacement** (Faults.py:756-1244)
- Ellipsoidal fault geometry model
- Displacement varies across fault surface (maximum at center)
- Heuristic displacement profile: D(x,y) = D_max × f(ellipsoid)
- Interpolation/stretching applied to age and depth volumes

#### **Closure Detection** (Closures.py:180-507)
- Flood-fill algorithm identifies connected regions
- Closure types:
  - **4-way (simple)**: Four-way dip closure
  - **3-way (faulted)**: Fault forms one side
  - **Stratigraphic (onlap)**: Onlap surface forms base
- Minimum voxel count filters remove tiny closures

#### **Fluid Substitution**
- Oil/gas closures identified before property calculation
- Different trend lines used for hydrocarbon sands
- No Gassmann equation - direct empirical trends
- Simplified but effective for training data

### 4.5 Augmentation Pipeline

**File:** `datagenerator/Augmentation.py`

#### **TZ Stretch** (Lines 9-82)
- Simulates time-depth conversion errors
- Random vertical stretching/compression
- Applied to simulate processing artifacts

#### **Uniform Stretch** (Lines 85-143)
- Random spatial scaling in all dimensions
- Creates additional variety in training data

### 4.6 Configuration System

**Example Config:** `config/example.json`

**Key Parameters:**
- `cube_shape`: [X, Y, Z] dimensions
- `incident_angles`: [7, 15, 24] degrees
- `thickness_min/max`: Layer thickness range
- `bandwidth_low/high`: Seismic frequency range
- `signal_to_noise_ratio_db`: [left, mode, right] dB values
- `min/max_number_faults`: Fault count range
- `closure_types`: ["simple", "faulted", "onlap"]
- `sand_layer_fraction`: Average sand percentage
- `sand_layer_thickness`: Average sand layer thickness

### 4.7 Key Algorithms Summary

| Algorithm | Location | Purpose | Complexity |
|-----------|----------|---------|------------|
| **Horizon deposition** | Horizons.py:1450-1498 | Build layers from base | O(n_layers × XY) |
| **Geologic age interpolation** | Geomodels.py:154-240 | 3D age model | O(XYZ) |
| **Ellipsoid fault displacement** | Faults.py:756-1244 | Generate fault offsets | O(n_faults × XY) |
| **Flood-fill closure detection** | Closures.py:180-507 | Find traps | O(XYZ × n_closures) |
| **Zoeppritz reflectivity** | Seismic.py:308-380 | Compute RFC | O(XYZ × n_angles) |
| **Butterworth bandpass** | Seismic.py:472-530 | Bandlimit seismic | O(XYZ × filter_order) |
| **Markov facies generation** | Horizons.py:1390-1448 | Layer sequences | O(n_layers) |

---

## 5. SUMMARY AND KEY INSIGHTS

### Strengths of the Approach

1. **Realistic Geologic Modeling**
   - Proper layering with Markov sequences
   - Multiple fault styles
   - Basin floor fans and onlap surfaces
   - Structural closure detection

2. **Physically-Based Properties**
   - Compaction trends (depth-dependent polynomials)
   - Fluid effects on rock properties
   - Proper mixing of lithologies (Backus/inverse-velocity)

3. **Rigorous Seismic Modeling**
   - Exact Zoeppritz equation (not approximations)
   - AVO effects captured
   - Realistic noise model (Hilterman weighting)
   - Proper bandwidth effects

4. **Scalable and Efficient**
   - Fast convolutional modeling
   - HDF5 chunked storage
   - Can generate thousands of training examples

### Limitations

1. **Simplified Physics**
   - No wave propagation (no multiples, diffractions)
   - No anisotropy (VTI effects)
   - No attenuation (Q effects)
   - No realistic wavelet estimation from field data

2. **Simplified Geology**
   - No salt kinematics
   - No realistic channel geometries (deprecated)
   - Simplified fluid substitution (empirical trends, not Gassmann)
   - No diagenetic effects beyond compaction

3. **1D Approach**
   - Each trace computed independently
   - No lateral wave effects
   - No spatial correlation of noise

### Why This Works for Deep Learning Training

The simplifications are **intentional and appropriate** for DL training:

- **Diversity over perfection**: Thousands of varied examples > few perfect ones
- **Feature capture**: Key features (AVO, faults, closures) well represented
- **Label accuracy**: Ground truth labels perfectly known (age, facies, fluids)
- **Speed**: Hours to generate training set, not months

The approach is well-suited for training networks to identify:
- Faults and fault blocks
- Structural closures (traps)
- Fluid indicators (AVO anomalies)
- Lithology classification

---

## 6. KEY FILE LOCATIONS QUICK REFERENCE

| Component | File Path | Key Lines | Description |
|-----------|-----------|-----------|-------------|
| Entry point | `main.py` | 18-83 | Full pipeline orchestration |
| Config loader | `Parameters.py` | 18-24 | Borg pattern, HDF setup |
| Horizon builder | `Horizons.py` | 1450-1577 | Layer deposition, thickness |
| Facies assignment | `Horizons.py` | 1339-1448 | Markov/binomial methods |
| Geologic age | `Geomodels.py` | 154-240 | 3D age interpolation |
| Fault generation | `Faults.py` | 756-1244 | Ellipsoid displacement |
| Closure detection | `Closures.py` | 180-507, 868-1167 | Flood-fill algorithm |
| Rock trends | `rpm_example.py` | 35-80 | Polynomial depth trends |
| RPM selection | `RockPropertyModels.py` | 143-167 | Strategy pattern |
| Property mixing | `RockPropertyModels.py` | 99-133 | Backus/inverse-velocity |
| Elastic model | `Seismic.py` | 693-1090 | Trend→3D conversion |
| Zoeppritz RFC | `Seismic.py` | 1379-1467 | Exact reflectivity |
| RFC volumes | `Seismic.py` | 308-380 | Trace-by-trace calculation |
| Butterworth | `Seismic.py` | 472-530, 1477-1499 | Bandpass filtering |
| Wavelet convolution | `Seismic.py` | 99-155 | FFT-based |
| Noise addition | `Seismic.py` | 402-470 | Hilterman weighting |
| Augmentation | `Augmentation.py` | 9-143 | TZ/uniform stretch |

---

## 7. RUNNING THE CODE

### Quick Start Example

```bash
conda activate synthoseis
python main.py --config config/example.json --num_runs 1 --run_id seismic_example
```

### Output Files

Generated in `project_folder/run_id/`:
- `seismicCubes_RFC_7_degrees.npy` - Near stack seismic
- `seismicCubes_RFC_15_degrees.npy` - Mid stack seismic
- `seismicCubes_RFC_24_degrees.npy` - Far stack seismic
- `seismicCubes_RFC_fullstack.npy` - Full stack seismic
- `qc_volume_age_volume.npy` - Geologic age labels
- `qc_volume_lithology.npy` - Lithology labels (0=shale, 1=sand, 2=salt)
- `qc_volume_oil_closures.npy` - Oil indicator
- `qc_volume_gas_closures.npy` - Gas indicator
- Various QC plots (PNG files)

---

## Conclusion

Synthoseis provides a sophisticated pipeline for generating synthetic seismic training data. The approach balances computational efficiency with physical realism, using:

- **Convolutional modeling** with exact Zoeppritz reflectivity
- **Empirical compaction trends** for rock properties
- **Markov processes** for realistic layering
- **Structural geology algorithms** for faults and closures
- **Angle-dependent noise** for realistic seismic characteristics

This combination produces diverse, physically plausible training data with perfect ground truth labels, ideal for deep learning applications in seismic interpretation.