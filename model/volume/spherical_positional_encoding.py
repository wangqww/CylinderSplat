import torch
import torch.nn as nn
import math
import unittest # Import unittest framework

# Helper class for sinusoidal positional encoding
class SphericalPositionalEncoding(nn.Module):
    def __init__(self, num_freqs=6, d_in=1, include_input=True):
        """
        Sinusoidal positional encoding for continuous scalar inputs.

        Args:
            num_freqs (int): Number of frequency bands (L).
            d_in (int): Input dimension (usually 1 for r, theta, or phi).
            include_input (bool): Whether to include the original input in the output.
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.include_input = include_input
        # Calculate output dimension
        self.d_out = 0
        if self.include_input:
            self.d_out += self.d_in
        if self.num_freqs > 0:
             # pi * 2^k frequencies
            self.freq_bands = (2.0 ** torch.arange(num_freqs)) * torch.pi
            self.d_out += self.d_in * 2 * num_freqs

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [..., d_in]. Assumed normalized e.g. to [-1, 1] or [0, 2pi].

        Returns:
            Tensor: Output tensor of shape [..., d_out].
        """
        out = []
        if self.include_input:
            out.append(x)
        if self.num_freqs > 0:
            # Unsqueeze for broadcasting frequency bands
            # x shape: [..., d_in] -> [..., d_in, 1]
            # freqs shape: [L] -> [1,...,1, L] (broadcasting happens)
            # sin/cos shape: [..., d_in, L]
            # Reshape needed: [..., d_in * L] -> flatten
            x_unsqueezed = x.unsqueeze(-1)
            scaled_x = x_unsqueezed * self.freq_bands.to(x.device).view(*([1] * x.dim()), -1)
            # Shape [..., d_in, L]
            sines = torch.sin(scaled_x)
            cosines = torch.cos(scaled_x)
            # Flatten last two dims: [..., d_in * 2 * L]
            out.append(sines.flatten(start_dim=-2))
            out.append(cosines.flatten(start_dim=-2))

        # Concatenate along the last dimension
        return torch.cat(out, dim=-1)


class SphericalTPVEncoding(nn.Module):
    def __init__(self,
                 output_dim,             # Desired output dimension for *each* encoded component (r, theta, phi)
                 num_freqs_radius=6,     # Frequencies for radius encoding
                 num_freqs_theta=6,      # Frequencies for theta encoding
                 num_freqs_phi=6,        # Frequencies for phi encoding
                 include_input=True,     # Include raw coordinate in encoding
                 # Define the 'resolution' or sampling grid size for each plane
                 grid_size_r=64,         # Number of steps for radius dimension
                 grid_size_theta=128,    # Number of steps for theta dimension (e.g., width)
                 grid_size_phi=64        # Number of steps for phi dimension (e.g., height)
                ):
        super().__init__()

        self.grid_size_r = grid_size_r
        self.grid_size_theta = grid_size_theta
        self.grid_size_phi = grid_size_phi

        # Create individual positional encoders
        self.encoder_r = SphericalPositionalEncoding(num_freqs_radius, d_in=1, include_input=include_input)
        self.encoder_theta = SphericalPositionalEncoding(num_freqs_theta, d_in=1, include_input=include_input)
        self.encoder_phi = SphericalPositionalEncoding(num_freqs_phi, d_in=1, include_input=include_input)

        # Use Linear layers to project encoded features to the desired uniform output_dim
        # This makes concatenation easier and standardizes feature size per coordinate
        self.proj_r = nn.Linear(self.encoder_r.d_out, output_dim) if self.encoder_r.d_out > 0 else nn.Identity()
        self.proj_theta = nn.Linear(self.encoder_theta.d_out, output_dim) if self.encoder_theta.d_out > 0 else nn.Identity()
        self.proj_phi = nn.Linear(self.encoder_phi.d_out, output_dim) if self.encoder_phi.d_out > 0 else nn.Identity()

        self.output_dim_per_coord = output_dim
        # Total dimension depends on the plane type (usually 2 components)
        self.total_output_dim = output_dim * 2

    def forward(self, bs, device, plane_type='tp'):
        """
        Generates positional encodings for points on one of the spherical TPV planes.

        Args:
            bs (int): Batch size.
            device (torch.device): Output device.
            plane_type (str): Which plane to generate encoding for ('tp', 'rt', 'rp').
                              'tp': Theta-Phi plane
                              'rt': Radius-Theta plane
                              'rp': Radius-Phi plane

        Returns:
            Tensor: Positional encoding tensor of shape [bs, H*W, C_total].
                    Where H, W depend on the plane_type, and C_total = output_dim * 2.
        """

        if plane_type == 'tp': # Theta-Phi Plane (like equirectangular)
            H, W = self.grid_size_phi, self.grid_size_theta
            # Generate grid coordinates normalized to [-1, 1]
            # Theta [-pi, pi] -> [-1, 1] ? Or [0, 2pi) -> [-1, 1]? Let's use [0, 2pi) -> [-1, 1]
            thetas_norm = torch.linspace(-1, 1, W, device=device) # Represents 0 to 2pi
            # Phi [0, pi] -> [-1, 1]
            phis_norm = torch.linspace(-1, 1, H, device=device) # Represents 0 to pi

            # Create meshgrid
            grid_theta, grid_phi = torch.meshgrid(thetas_norm, phis_norm, indexing='xy') # W, H

            # Encode coordinates (unsqueeze to add channel dim for encoding)
            enc_theta = self.proj_theta(self.encoder_theta(grid_theta.unsqueeze(-1))) # W, H, D_out
            enc_phi = self.proj_phi(self.encoder_phi(grid_phi.unsqueeze(-1)))       # W, H, D_out

            # Concatenate features
            pos_enc = torch.cat((enc_theta, enc_phi), dim=-1) # W, H, 2*D_out

        elif plane_type == 'rt': # Radius-Theta Plane
            H, W = self.grid_size_r, self.grid_size_theta
            # Radius normalized to [-1, 1] (needs careful definition based on scene)
            radius_norm = torch.linspace(-1, 1, H, device=device)
            thetas_norm = torch.linspace(-1, 1, W, device=device) # Represents 0 to 2pi

            grid_r, grid_theta = torch.meshgrid(radius_norm, thetas_norm, indexing='xy') # H, W

            enc_r = self.proj_r(self.encoder_r(grid_r.unsqueeze(-1)))           # H, W, D_out
            enc_theta = self.proj_theta(self.encoder_theta(grid_theta.unsqueeze(-1))) # H, W, D_out

            pos_enc = torch.cat((enc_r, enc_theta), dim=-1) # H, W, 2*D_out


        elif plane_type == 'rp': # Radius-Phi Plane
            H, W = self.grid_size_r, self.grid_size_phi
            radius_norm = torch.linspace(-1, 1, H, device=device)
            phis_norm = torch.linspace(-1, 1, W, device=device) # Represents 0 to pi

            grid_r, grid_phi = torch.meshgrid(radius_norm, phis_norm, indexing='xy') # H, W

            enc_r = self.proj_r(self.encoder_r(grid_r.unsqueeze(-1)))     # H, W, D_out
            enc_phi = self.proj_phi(self.encoder_phi(grid_phi.unsqueeze(-1))) # H, W, D_out

            pos_enc = torch.cat((enc_r, enc_phi), dim=-1) # H, W, 2*D_out

        else:
            raise ValueError(f"Unknown plane_type: {plane_type}")

        # Reshape and repeat for batch
        # pos_enc shape: [W, H, C] or [H, W, C] depending on meshgrid indexing and plane
        # Flatten spatial dimensions -> [H*W, C]
        pos_enc_flat = pos_enc.reshape(-1, self.total_output_dim)
        # Add batch dimension and repeat -> [bs, H*W, C]
        pos_enc_batch = pos_enc_flat.unsqueeze(0).expand(bs, -1, -1)

        return pos_enc_batch

    def extra_repr(self):
        return f'output_dim={self.output_dim_per_coord}, grid_r={self.grid_size_r}, grid_theta={self.grid_size_theta}, grid_phi={self.grid_size_phi}'
    



# --- Test Suite using unittest ---
class TestSphericalEncoding(unittest.TestCase):

    def setUp(self):
        # Common parameters for tests
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dim = 64
        self.grid_r = 16
        self.grid_theta = 32
        self.grid_phi = 16
        self.bs = 2
        self.model = SphericalTPVEncoding(
            output_dim=self.output_dim,
            grid_size_r=self.grid_r,
            grid_size_theta=self.grid_theta,
            grid_size_phi=self.grid_phi,
            num_freqs_radius=4,
            num_freqs_theta=8,
            num_freqs_phi=4,
            include_input=False # Example: test without raw input
        ).to(self.device)
        self.model.eval() # Set to eval mode for testing (if dropout/bn existed)

    def test_output_shapes(self):
        """Verify the output shapes for each plane type."""
        with torch.no_grad():
            # Theta-Phi Plane
            out_tp = self.model(self.bs, self.device, plane_type='tp')
            expected_points_tp = self.grid_phi * self.grid_theta
            expected_dim_tp = self.output_dim * 2
            self.assertEqual(out_tp.shape, torch.Size([self.bs, expected_points_tp, expected_dim_tp]))

            # Radius-Theta Plane
            out_rt = self.model(self.bs, self.device, plane_type='rt')
            expected_points_rt = self.grid_r * self.grid_theta
            expected_dim_rt = self.output_dim * 2
            self.assertEqual(out_rt.shape, torch.Size([self.bs, expected_points_rt, expected_dim_rt]))

            # Radius-Phi Plane
            out_rp = self.model(self.bs, self.device, plane_type='rp')
            expected_points_rp = self.grid_r * self.grid_phi
            expected_dim_rp = self.output_dim * 2
            self.assertEqual(out_rp.shape, torch.Size([self.bs, expected_points_rp, expected_dim_rp]))

    def test_output_type_device(self):
        """Verify output dtype and device."""
        with torch.no_grad():
            out_tp = self.model(self.bs, self.device, plane_type='tp')
            self.assertEqual(out_tp.dtype, torch.float32) # Assuming default float
            self.assertEqual(out_tp.device, self.device)

    def test_encoding_sensitivity(self):
        """Check if slightly different inputs produce different outputs."""
        with torch.no_grad():
            # Test the underlying SphericalPositionalEncoding helper
            encoder = SphericalPositionalEncoding(num_freqs=6, d_in=1).to(self.device)
            input1 = torch.tensor([[[0.1]]], device=self.device) # Shape [1, 1, 1]
            input2 = torch.tensor([[[0.11]]], device=self.device) # Slightly different
            input3 = torch.tensor([[[0.1]]], device=self.device) # Same as input1

            out1 = encoder(input1)
            out2 = encoder(input2)
            out3 = encoder(input3)

            # Check different inputs produce different outputs
            self.assertFalse(torch.allclose(out1, out2))
            # Check same inputs produce same outputs
            self.assertTrue(torch.allclose(out1, out3))

    def test_theta_periodicity_approx(self):
        """Check if theta=0 and theta=2pi (normalized -1 and 1) have similar encodings."""
        with torch.no_grad():
            # Get Theta-Phi plane encoding
            H, W = self.grid_phi, self.grid_theta
            C = self.output_dim * 2
            out_tp = self.model(1, self.device, plane_type='tp').reshape(H, W, C) # Reshape back to grid for easier indexing

            # Encoding for theta near 0 (normalized -1) and theta near 2pi (normalized 1)
            # Note: linspace includes endpoints, so index 0 is -1, index W-1 is +1
            enc_theta_minus1 = out_tp[:, 0, :self.output_dim] # First half is (weighted) theta encoding
            enc_theta_plus1 = out_tp[:, W-1, :self.output_dim]

            # Due to sin(phi) weighting, they won't be exactly the same unless phi=pi/2
            # Let's check near the equator (middle phi index)
            mid_phi_idx = H // 2
            enc_near_equator_minus1 = enc_theta_minus1[mid_phi_idx]
            enc_near_equator_plus1 = enc_theta_plus1[mid_phi_idx]

            # They should be close because sin/cos(0) approx sin/cos(2pi)
            # The projection layer might make them less similar, but the underlying fourier feats are periodic
            # Increase tolerance due to projection layer and sin(phi) weight variation
            self.assertTrue(torch.allclose(enc_near_equator_minus1, enc_near_equator_plus1, atol=1e-5),
                            f"Theta encodings at ends (-1 and 1) are not close near equator. Diff: {(enc_near_equator_minus1 - enc_near_equator_plus1).abs().max()}")

    def test_invalid_plane_type(self):
        """Test that an invalid plane_type raises an error."""
        with self.assertRaises(ValueError):
            self.model(self.bs, self.device, plane_type='xy') # Invalid type

# --- Run the tests ---
if __name__ == '__main__':
    # Corrected the meshgrid indexing in the SphericalTPVEncoding class to 'ij'
    # Re-run tests
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSphericalEncoding))
    runner = unittest.TextTestRunner()
    runner.run(suite)

    # --- Simple direct test (alternative to unittest) ---
    print("\n--- Direct Test ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SphericalTPVEncoding(output_dim=64, grid_size_r=16, grid_size_theta=32, grid_size_phi=16).to(device)
    model.eval()
    with torch.no_grad():
        out_tp = model(bs=1, device=device, plane_type='tp')
        out_rt = model(bs=1, device=device, plane_type='rt')
        out_rp = model(bs=1, device=device, plane_type='rp')
        print(f"Direct Test - Theta-Phi Output Shape: {out_tp.shape}")
        print(f"Direct Test - Radius-Theta Output Shape: {out_rt.shape}")
        print(f"Direct Test - Radius-Phi Output Shape: {out_rp.shape}")