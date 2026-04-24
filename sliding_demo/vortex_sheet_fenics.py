#%%
# Discontinuous Diffeomorphic Image Registration with Groupoid
# ==========================================================
#
# This script demonstrates the discontinuous diffeomorphic image registration
# with FEniCS-based velocity-momentum relationship solver.
#
# Key features:
# - Discontinuous EPDiff equation with sliding boundary conditions
# - FEniCS solver for I(u) = (Id - Δ)^R u + I_Γ(u) = m

import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

import torch
from mermaid.data_wrapper import AdaptVal
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage.util as sku
import os
import matplotlib

import mermaid.example_generation as eg
import mermaid.module_parameters as pars
import mermaid.multiscale_optimizer as MO
import mermaid.registration_networks as RN
import mermaid.forward_models as FM
import mermaid.utils as us

#%%
# Configuration for Discontinuous Registration
# ===========================================

use_map = True
model_name = 'lddmm_discontinuous'  # the new discontinuous model
map_low_res_factor = 1   # 

# Optimizer settings
optimizer_name = 'sgd' #'lbfgs_ls' #'adam' # 
nr_of_iterations = 31
visualize = True
visualize_step = 10

# Create parameter structure
params = pars.ParameterDict()

params['model']['registration_model']['env']['reg_factor'] = 1.0
params['optimizer']['single_scale']['rel_ftol'] = 1e-10
params['optimizer']['single_scale']['abs_ftol'] = 1e-10

# Configure FEniCS discontinuous smoother
params['smoother']['type'] = 'fenics_discontinuous'
params['smoother']['mask_path'] = '../data/Toy_Template_gra_mask.png'

# Configure forward model for discontinuous EPDiff
params['model']['registration_model']['smoother']['type'] = 'fenics_discontinuous'
params['model']['registration_model']['smoother']['mask_path'] = params['smoother']['mask_path']
params['model']['registration_model']['smoother']['fenics_solver_params'] = {}
params['model']['registration_model']['forward_model']['smoother'] = {}
params['model']['registration_model']['forward_model']['smoother']['type'] = 'fenics_discontinuous'
params['forward_model']['smoother_for_forward_model'] = True
params['forward_model']['use_fenics_discontinuous'] = True

params['model']['registration_model']['smoother']['fenics_solver_params'] = {}
params['model']['registration_model']['smoother']['fenics_solver_params']['R'] = 1               
params['model']['registration_model']['smoother']['fenics_solver_params']['alpha'] = 0.1
params['model']['registration_model']['smoother']['fenics_solver_params']['penalty_parameter'] = 50.0

#%%
# Load Images and Mask
# Load source and target images
print("Loading images...")
I0 = skio.imread('../data/Toy_Template_gra.png')
I0 = sku.img_as_float32(I0[np.newaxis, np.newaxis, ...])

I1 = skio.imread('../data/Toy_Reference_gra.png')
I1 = sku.img_as_float32(I1[np.newaxis, np.newaxis, ...])

# Load mask for interface definition
try:
    mask = skio.imread(params['smoother']['mask_path'])
    if len(mask.shape) > 2:
        mask = mask[..., 0]
    mask = sku.img_as_float32(mask[np.newaxis, np.newaxis, ...])
    print(f"Mask loaded successfully: shape {mask.shape}")
except:
    print("Warning: Could not load mask, will use default")
    mask = None

sz = np.array(I0.shape)
spacing = 1. / (sz[2::] - 1)

# Ensure mask has the same resolution as images
if mask is not None:
    # Get expected shape from source image
    expected_shape = sz[2:]  # (H, W)
    mask_shape = mask.shape[2:]  # Extract (H, W) from mask

    if mask_shape[0] != expected_shape[0] or mask_shape[1] != expected_shape[1]:
        print(f"Resizing mask from {mask_shape} to {tuple(expected_shape)}...")

        import skimage.transform as skt
        mask_resized = skt.resize(mask[0, 0, :, :], tuple(expected_shape),
                                 order=0,  # Nearest neighbor for binary mask
                                 preserve_range=True,
                                 anti_aliasing=False)
        mask = sku.img_as_float32(mask_resized[np.newaxis, np.newaxis, ...])
        print(f"Mask resized to {mask.shape}")

    MASK_np = mask[0, 0, :, :]
else:
    MASK_np = None

print(f"Image size: {sz}")
print(f"Spacing: {spacing}")

#%%
# Convert to PyTorch
# =================

ISource = AdaptVal(torch.from_numpy(I0.copy()))
ITarget = AdaptVal(torch.from_numpy(I1))

if mask is not None:
    IMask = AdaptVal(torch.from_numpy(mask))
    FM.EPDiffDiscontinuousMap.SHARED_MASK_TENSOR = IMask
else:
    IMask = None

print("Images converted to PyTorch tensors")

#%%
# Setup Discontinuous Registration Optimizer
# =========================================

print("Setting up discontinuous LDDMM optimizer...")

# Create single-scale optimizer
so = MO.SingleScaleRegistrationOptimizer(sz, spacing, use_map, map_low_res_factor, params)

# Add the new discontinuous model
try:
    # Use the new LDDMMDiscontinuousMapNet
    so.add_model('lddmm_discontinuous_map',
                 RN.LDDMMDiscontinuousMapNet,
                 RN.LDDMMShootingVectorMomentumMapLoss,  # Use existing loss for now
                 use_map=True)

    so.set_model('lddmm_discontinuous_map')
    print("✓ Discontinuous LDDMM model registered successfully")

except Exception as e:
    print(f"Warning: Could not register discontinuous model: {e}")
    print("Falling back to MultiK model with FEniCS smoother...")

    # Fallback to MultiK with FEniCS smoother
    so.add_model('lddmm_shooting_mapmultik',
                 RN.LDDMMShootingVectorMomentumMapMultiKNet,
                 RN.LDDMMShootingVectorMomentumMapMultiKLoss,
                 use_map=True)
    so.set_model('lddmm_shooting_mapmultik')

# Configure optimizer
so.set_optimizer_by_name(optimizer_name)
so.set_visualization(visualize)
so.set_visualize_step(visualize_step)
so.set_number_of_iterations(nr_of_iterations)

# Set images
so.set_source_image(ISource)
so.set_target_image(ITarget)

# Set recording for analysis
so.set_recording_step(1)

print("Optimizer configuration completed")

#%%
# Run Discontinuous Registration
# =============================

print("Starting discontinuous diffeomorphic registration...")
print("This will use FEniCS to solve I(u) = (Id - Δ)^R u + I_Γ(u) = m")
print("=" * 60)

try:
    so.optimize()
    print("=" * 60)
    print("✓ Registration completed successfully!")

except Exception as e:
    print("=" * 60)
    print("✗ REGISTRATION FAILED")
    print(e)
    print("=" * 60)

#%%
# Analyze Results
# ==============

print("\nAnalyzing results...")
save_path = '../result/'
method = 'Vortexsheet_fenics_rectangle' 
os.makedirs(save_path, exist_ok=True)

# Get energy history
energy_history = so.get_history()

# Plot energy evolution
if energy_history is not None and len(energy_history) > 0:
    # Check if we have the expected keys
    if 'iter' in energy_history and 'energy' in energy_history:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(energy_history['iter'], energy_history['energy'])
        plt.title('Total Energy')
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.grid(True)

        if 'similarity_energy' in energy_history:
            plt.subplot(1, 3, 2)
            plt.plot(energy_history['iter'], energy_history['similarity_energy'])
            plt.title('Similarity Energy')
            plt.xlabel('Iteration')
            plt.ylabel('Energy')
            plt.grid(True)

        if 'regularization_energy' in energy_history:
            plt.subplot(1, 3, 3)
            plt.plot(energy_history['iter'], energy_history['regularization_energy'])
            plt.title('Regularization Energy (DVect norm)')
            plt.xlabel('Iteration')
            plt.ylabel('Energy')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('../result/Vortexsheet_fenics_wheel_energies.png', dpi=300)
        plt.show()

        # Print final energies
        if len(energy_history['energy']) > 0:
            final_energy = energy_history['energy'][-1]
            print(f"Final total energy: {final_energy:.6f}")

            if 'similarity_energy' in energy_history and len(energy_history['similarity_energy']) > 0:
                final_sim = energy_history['similarity_energy'][-1]
                print(f"Final similarity energy: {final_sim:.6f}")

            if 'regularization_energy' in energy_history and len(energy_history['regularization_energy']) > 0:
                final_reg = energy_history['regularization_energy'][-1]
                print(f"Final regularization energy: {final_reg:.6f}")
    else:
        print(f"Energy history format: {list(energy_history.keys()) if isinstance(energy_history, dict) else type(energy_history)}")
        print("Energy history does not contain expected keys")
else:
    print("No energy history available - optimization may not have completed properly")

#%%
# Define visualization functions
# ===============================================================

def tensor_to_numpy_viz(x):
    """Convert tensor to numpy array, handling both CPU and CUDA tensors"""
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            return x.detach().cpu().numpy()
        else:
            return x.detach().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)

def show_image(I, title):
    I_np = tensor_to_numpy_viz(I)
    plt.imshow(I_np[0,0,:,:], cmap='gray')
    plt.axis('off')
    plt.title(title)

def show_warped_with_grid(I, phi, title):
    I_np = tensor_to_numpy_viz(I)
    phi_np = tensor_to_numpy_viz(phi)
    plt.imshow(I_np[0,0,:,:] ,cmap='gray')
    m, n = phi_np.shape[2:]
    downsample1 = 1
    downsample2 = 1
    plt.contour(phi_np[0, 0, :, :],
            np.linspace(-2, 2, 1*int(m/downsample1)),
            colors='r',
            linestyles='solid',
            linewidths=0.5)
    plt.contour(phi_np[0, 1, :, :],
            np.linspace(-2, 2, 1*int(n/downsample2)),
            colors='r',
            linestyles='solid',
            linewidths=0.5)
    plt.axis('off')
    plt.title(title)

def show_quiver_2D(I, phi, title):
    I_np = tensor_to_numpy_viz(I)
    phi_np = tensor_to_numpy_viz(phi)
    plt.imshow(I_np[0,0,:,:], cmap='gray')
    m, n = phi_np.shape[1:]
    x = np.linspace(0, m-1, m)
    y = np.linspace(0, n-1, n)
    downsample1 = 2
    downsample2 = 2
    X = x[0::downsample1]
    Y = y[0::downsample2]
    U = phi_np[0, 0::downsample1, 0::downsample2]
    V = phi_np[1, 0::downsample1, 0::downsample2]
    plt.quiver(Y, X, V, -U, color = 'red', scale = 3, width=0.002, pivot='tip')
    plt.axis('off')
    plt.title(title)

#%%
# Visualize Registration Results

try:
    # Get results from optimizer history
    h = so.get_history()

    if 'recording' in h and len(h['recording']) > 0:
        # Get data from recording (like in vortex_sheet.py)
        source_img = h['recording'][-1]['iS']
        target_img = h['recording'][-1]['iT']
        source_img_wp = h['recording'][-1]['iW']
        phi = h['recording'][-1]['phiWarped']
    else:
        # Fallback to direct computation
        phi = so.get_map()
        source_img = ISource
        target_img = ITarget
        if phi is not None:
            source_img_wp = us.compute_warped_image_multiNC(ISource, phi, spacing, 1, zero_boundary=False)
        else:
            source_img_wp = ISource
            phi = AdaptVal(torch.from_numpy(us.identity_map_multiN(sz, spacing)))

    # Compute identity map and displacement
    phi0 = us.identity_map_multiN(sz, spacing) #identityMap

    # Convert phi to numpy if it's a tensor
    if hasattr(phi, 'detach'):
        phi_np = phi.detach().cpu().numpy()
    else:
        phi_np = phi

    disp = phi0 - phi_np

    # Main visualization
    plt.figure(figsize=(15,10))
    plt.subplot(231); show_image(source_img, 'Source image')
    plt.subplot(232); show_image(target_img, 'Target image')
    plt.subplot(233); show_image(source_img_wp, 'Warped image')
    plt.subplot(234); show_warped_with_grid(source_img_wp, phi_np, 'Deformed grid')
    plt.subplot(235); show_quiver_2D(source_img_wp, disp[0, :, :, :], 'Deformation field')
    plt.savefig(os.path.join(save_path, method + '_result.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()


    # Individual visualizations
    # Save deformed image
    plt.figure()
    show_image(source_img_wp,'')
    plt.savefig(os.path.join(save_path, method + '_deformed_image.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()


    # Save deformed grid
    plt.figure()
    show_warped_with_grid(source_img_wp, phi_np, '')
    plt.savefig(os.path.join(save_path, method + '_deformed_grid.png'), format='png', dpi=300, bbox_inches='tight')
    plt.show()


    # Save quiver with mask overlay
    plt.figure()
    show_quiver_2D(source_img_wp, disp[0, :, :, :], '')
    if MASK_np is not None:
        plt.contour(MASK_np, levels=[0.5], colors='yellow', linewidths=2)
    plt.savefig(os.path.join(save_path, method + '_quiver.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot differences
    # Convert to numpy for computation
    target_img_np = tensor_to_numpy_viz(target_img)
    source_img_np = tensor_to_numpy_viz(source_img)
    source_img_wp_np = tensor_to_numpy_viz(source_img_wp)

    diff_before = target_img_np - source_img_np
    diff_after = target_img_np - source_img_wp_np
    vmin = diff_before.min()
    vmax = diff_before.max()
    norm = matplotlib.colors.Normalize(vmin, vmax)

    # Before registration
    plt.figure()
    h1 = plt.imshow(diff_before[0,0,:,:], cmap='coolwarm', norm = norm)
    plt.axis('off')
    plt.colorbar(h1)
    plt.savefig(os.path.join(save_path, method + '_diff_before.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()

    # After registration
    plt.figure()
    h2 = plt.imshow(diff_after[0,0,:,:], cmap='coolwarm', norm = norm)
    plt.axis('off')
    plt.colorbar(h2)
    plt.savefig(os.path.join(save_path, method + '_diff_after.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved files with prefix: {method}")

except Exception as e:
    print("=" * 60)
    print("✗ VISUALIZATION FAILED:", e)
    print("=" * 60)
