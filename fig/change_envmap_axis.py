import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from scipy.spatial.transform import Rotation

def rotate_equirectangular(image, pitch_deg=0, yaw_deg=0, roll_deg=0):
    """
    Rotates an equirectangular environment map.
    :param image: Input image (numpy array)
    :param pitch_deg: Rotate around X axis (Up/Down)
    :param yaw_deg: Rotate around Y axis (Left/Right)
    :param roll_deg: Rotate around Z axis (Tilt) - assuming Y-up view for naming
    """
    h, w = image.shape[:2]
    
    # Create a meshgrid of pixel coordinates
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    xv, yv = np.meshgrid(x, y)

    # Normalize coordinates to [-1, 1] for U and [-pi/2, pi/2] for Latitude
    # Standard Equirectangular:
    # U = phi / (2*pi) -> phi = U * 2*pi
    # V = theta / pi   -> theta = V * pi
    # But usually theta is 0 at North Pole.
    
    # Map pixels to spherical coordinates (phi, theta)
    # phi ranges [0, 2pi], theta ranges [0, pi]
    phi = (xv / w) * 2 * np.pi
    theta = (yv / h) * np.pi

    # Convert Spherical to Cartesian (Standard Y-up convention for calculation)
    # x = sin(theta) * cos(phi)
    # y = cos(theta)            <- Up axis
    # z = sin(theta) * sin(phi)
    
    # Note: We use a temporary coordinate system where Y is Up (Theta=0 maps to Y=1)
    # to perform the rotation easily.
    
    x_cart = np.sin(theta) * np.cos(phi)
    y_cart = np.cos(theta)
    z_cart = np.sin(theta) * np.sin(phi)
    
    # Stack into vectors
    vectors = np.dstack((x_cart, y_cart, z_cart)) # Shape (h, w, 3)
    
    # Define Rotation
    # We want to rotate the VIEW direction, which is the inverse of rotating the world.
    # But usually users want to "Rotate the World". 
    # If we want to move Top to Front (Rotate -90 on X), we apply that matrix to the vectors?
    # No, we need the Inverse mapping: For each pixel in TARGET, where does it come from in SOURCE?
    
    rot = Rotation.from_euler('xyz', [pitch_deg, yaw_deg, roll_deg], degrees=True)
    rot_matrix = rot.as_matrix() # 3x3
    
    # For target pixel P_out, its vector is V. We want to find sample vector V_in.
    # If World rotates by R, then V_new = R * V_old.
    # We are iterating over V_new (the target pixels). We need V_old.
    # V_old = R_inv * V_new.
    
    rot_inv = rot.inv()
    
    # Reshape for efficient matrix multiplication
    vectors_flat = vectors.reshape(-1, 3)
    vectors_rotated = rot_inv.apply(vectors_flat)
    vectors_rotated = vectors_rotated.reshape(h, w, 3)
    
    # Convert back to Spherical
    x_r = vectors_rotated[:,:,0]
    y_r = vectors_rotated[:,:,1]
    z_r = vectors_rotated[:,:,2]
    
    # Theta (0 to pi)
    # y = cos(theta) -> theta = arccos(y)
    # Clip to handle float errors
    theta_new = np.arccos(np.clip(y_r, -1.0, 1.0))
    
    # Phi (0 to 2pi)
    # z = sin(theta)sin(phi), x = sin(theta)cos(phi)
    # atan2(z, x) gives range [-pi, pi]
    phi_new = np.arctan2(z_r, x_r)
    # Normalize phi to [0, 2pi]
    phi_new = np.mod(phi_new, 2*np.pi)
    
    # Map back to Pixel Coordinates (u, v)
    u_src = (phi_new / (2 * np.pi)) * w
    v_src = (theta_new / np.pi) * h
    
    # Remap
    # map_x = u_src, map_y = v_src
    new_image = cv2.remap(image, u_src.astype(np.float32), v_src.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    
    return new_image

# Load the user image (simulation)
input_path = "/mnt/data/youkeyao/FovLight/fig/env.exr" # Assuming the user image is saved here
image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
image= np.clip(image, 0.0, 1.0)

if image is not None:
    # Apply rotation: -90 degrees on X axis.
    # This moves the "Top" (Y) to "Front" (Z).
    # If the user's "Y-up" means they want the content currently at Z-pole to be at Y-pole,
    # and the image currently has content at Top (Pole)... 
    # Let's assume they want the standard "Fix Z-up import" rotation.
    
    # Rotate -90 degrees around X
    rotated_image = rotate_equirectangular(image, pitch_deg=-90)
    rotated_image = (rotated_image * 255).astype(np.uint8)
    
    cv2.imwrite("output_y_up.png", rotated_image)
else:
    print("Image not found")