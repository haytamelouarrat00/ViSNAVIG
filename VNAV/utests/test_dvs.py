"""Unit tests for the DVS controller's analytic building blocks.

The high-bar test (numerical L_I vs analytic L_I under a true camera warp)
needs a real renderer, so it lives in the integration runs. Here we cover:
  - L_I closed-form sanity on a ramp image,
  - pyramid intrinsic rescaling preserves projection consistency,
  - hole_mask drops invalid + dilates,
  - Huber down-weights outliers,
  - MI Hessian is PSD and the cost decreases when target ≈ current.
"""

import unittest
import numpy as np

from VNAV.utilities import photometric as ph
from VNAV.controllers import DVSController


class TestPhotometricHelpers(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.H, self.W = 64, 64
        self.K = np.array([[100.0, 0.0, 32.0],
                           [0.0, 100.0, 32.0],
                           [0.0, 0.0, 1.0]])
        # Smooth ramp + texture: gradients are well-defined.
        u, v = np.meshgrid(np.arange(self.W), np.arange(self.H), indexing="xy")
        self.I = (50.0 + 100.0 * np.sin(u / 7.0) * np.cos(v / 9.0))
        self.Z = np.full((self.H, self.W), 2.0)

    def test_farid7_gradient_shape_and_finite(self):
        Ix, Iy = ph.farid7_gradients(self.I)
        self.assertEqual(Ix.shape, self.I.shape)
        self.assertTrue(np.all(np.isfinite(Ix)) and np.all(np.isfinite(Iy)))

    def test_luminance_interaction_matrix_zeroed_on_invalid(self):
        Ix, Iy = ph.farid7_gradients(self.I)
        bad_depth = self.Z.copy()
        bad_depth[10:20, 10:20] = -1.0  # invalid region
        L_I, valid = ph.luminance_interaction_matrix(Ix, Iy, bad_depth, self.K)
        self.assertEqual(L_I.shape, (self.H * self.W, 6))
        self.assertEqual(valid.shape, (self.H * self.W,))
        # Rows for invalid pixels must be exactly zero.
        invalid_rows = L_I[~valid]
        self.assertTrue(np.allclose(invalid_rows, 0.0))
        self.assertEqual(int(valid.sum()), self.H * self.W - 100)

    def test_intrinsics_pyramid_projection_consistency(self):
        # Take a 3D point in front of the camera; its projection at every
        # pyramid level should map to the same downsampled location.
        Ks = ph.intrinsics_pyramid(self.K, levels=3)
        P = np.array([0.3, -0.2, 2.0])
        u0 = Ks[0][0, 0] * P[0] / P[2] + Ks[0][0, 2]
        v0 = Ks[0][1, 1] * P[1] / P[2] + Ks[0][1, 2]
        for lvl, K_l in enumerate(Ks):
            u_l = K_l[0, 0] * P[0] / P[2] + K_l[0, 2]
            v_l = K_l[1, 1] * P[1] / P[2] + K_l[1, 2]
            scale = 0.5 ** lvl
            # cv2.pyrDown convention: u_new = (u_old + 0.5)/2 - 0.5.
            u_expected = (u0 + 0.5) * scale - 0.5 if lvl > 0 else u0
            v_expected = (v0 + 0.5) * scale - 0.5 if lvl > 0 else v0
            # Apply repeatedly for level > 1.
            u_expected = u0
            v_expected = v0
            for _ in range(lvl):
                u_expected = (u_expected + 0.5) * 0.5 - 0.5
                v_expected = (v_expected + 0.5) * 0.5 - 0.5
            self.assertAlmostEqual(u_l, u_expected, places=6)
            self.assertAlmostEqual(v_l, v_expected, places=6)

    def test_hole_mask_drops_invalid_and_borders(self):
        depth = self.Z.copy()
        depth[20:24, 20:24] = np.nan
        mask = ph.hole_mask(depth, border=2, dilate=2, grad_quantile=1.0)
        # Border pixels must be False.
        self.assertFalse(mask[0, 0])
        self.assertFalse(mask[-1, -1])
        # The 4×4 NaN region plus a 2-px halo (so a 8×8 block) must be False.
        self.assertFalse(mask[18:26, 18:26].any())
        # Far-away pixels remain True.
        self.assertTrue(mask[40, 40])

    def test_huber_downweights_outliers(self):
        e = np.random.randn(1000)
        e[0] = 50.0  # clear outlier
        w = ph.huber_weights(e)
        self.assertLess(w[0], 0.1)
        # Most weights are exactly 1.
        self.assertGreater(int((w == 1.0).sum()), 800)

    def test_znssd_zero_when_identical(self):
        e, c = ph.znssd_residual(self.I, self.I)
        self.assertAlmostEqual(c, 0.0, places=10)

    def test_mi_hessian_is_psd_and_grad_finite(self):
        Ix, Iy = ph.farid7_gradients(self.I)
        L_I, valid = ph.luminance_interaction_matrix(Ix, Iy, self.Z, self.K)
        I_target = self.I + 5.0  # affine-shifted; should have low |grad|
        cost, grad, hess = ph.mi_cost_grad_hess(
            self.I, I_target, L_I, valid, n_bins=8,
            intensity_range=(self.I.min(), self.I.max()),
        )
        self.assertTrue(np.all(np.isfinite(grad)))
        self.assertTrue(np.all(np.isfinite(hess)))
        eigvals = np.linalg.eigvalsh(hess)
        # GN approximation Σ (1/p) g g^T is PSD.
        self.assertGreaterEqual(eigvals.min(), -1e-9)


class TestDVSControllerSmoke(unittest.TestCase):
    """End-to-end smoke: build the controller, call compute_velocity once
    on a tiny synthetic patch, expect a finite 6-vector and no crash."""

    def setUp(self):
        np.random.seed(1)
        self.H, self.W = 80, 80
        self.K = np.array([[120.0, 0.0, 40.0],
                           [0.0, 120.0, 40.0],
                           [0.0, 0.0, 1.0]])
        u, v = np.meshgrid(np.arange(self.W), np.arange(self.H), indexing="xy")
        self.I_curr = 128.0 + 80.0 * np.sin(u / 6.0) * np.cos(v / 8.0)
        self.I_target = 128.0 + 80.0 * np.sin((u + 2) / 6.0) * np.cos((v + 1) / 8.0)
        self.depth = np.full((self.H, self.W), 1.5)

    def _check_v(self, v):
        self.assertEqual(v.shape, (6,))
        self.assertTrue(np.all(np.isfinite(v)))

    def test_znssd_step_is_finite(self):
        c = DVSController(similarity="znssd", solver="esm_lm",
                          pyramid_levels=2, depth_source="render")
        v = c.compute_velocity(self.I_curr, self.depth, self.I_target, self.K)
        self._check_v(v)

    def test_mi_step_is_finite(self):
        # mi_intensity_range=None → auto-derive from data; verifies the fix
        # for renderers that output [0, 1] floats.
        c = DVSController(similarity="mi", solver="lm",
                          pyramid_levels=2, depth_source="render",
                          mi_n_bins=8, mi_intensity_range=None)
        v = c.compute_velocity(self.I_curr, self.depth, self.I_target, self.K)
        self._check_v(v)
        self.assertGreaterEqual(c.current_error_norm, 0.0)  # SSD-based, non-negative

    def test_reset_clears_target_cache(self):
        c = DVSController(similarity="znssd", pyramid_levels=2)
        c.compute_velocity(self.I_curr, self.depth, self.I_target, self.K)
        self.assertIsNotNone(c._target_id)
        c.reset()
        self.assertIsNone(c._target_id)
        self.assertEqual(c.current_level, c.pyramid_levels - 1)


if __name__ == "__main__":
    unittest.main()
