// Multigrid padding that adheres to matlab/TOP3D_XL.m (lines 1763–1784)
// - Computes base levels L by rounding(numVoxels/8) iteratively until < coarsestResolutionControl
// - Enforces L = max(3, L)
// - Pads each dimension to the nearest multiple of 2^L
// - Returns numLevelsForHierarchy = L + 1 (to match MATLAB's post-increment)

#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>

namespace top3d {

struct PaddingResult {
	std::uint64_t adjustedNelx;
	std::uint64_t adjustedNely;
	std::uint64_t adjustedNelz;
	int baseLevels;               // L in MATLAB before +1
	int numLevelsForHierarchy;    // L + 1 (as set in TOP3D_XL.m after padding)
};

inline int compute_base_levels(std::uint64_t numVoxels,
					   std::uint64_t coarsestResolutionControl = 50000,
					   int minLevels = 3) {
	int L = 0;
	while (numVoxels >= coarsestResolutionControl) {
		++L;
		// MATLAB: round(numVoxels/8) → halves away from zero; for integers: (n+4)/8
		numVoxels = (numVoxels + 4u) / 8u;
	}
	return std::max(minLevels, L);
}

inline std::uint64_t pad_up_to_multiple_of_pow2(std::uint64_t n, int L) {
	if (L <= 0) return n;
	const std::uint64_t step = (1ull << L);
	return ((n + step - 1ull) / step) * step;
}

// Compute adjusted dims only, without touching volume data.
inline PaddingResult compute_adjusted_dims(std::uint64_t nelx,
							 std::uint64_t nely,
							 std::uint64_t nelz,
							 std::uint64_t numSolidVoxels,
							 std::uint64_t coarsestResolutionControl = 50000,
							 int minLevels = 3) {
	const int L = compute_base_levels(numSolidVoxels, coarsestResolutionControl, minLevels);
	const std::uint64_t ax = pad_up_to_multiple_of_pow2(nelx, L);
	const std::uint64_t ay = pad_up_to_multiple_of_pow2(nely, L);
	const std::uint64_t az = pad_up_to_multiple_of_pow2(nelz, L);
	return PaddingResult{ax, ay, az, L, L + 1};
}

// Pad a 3D logical/byte volume in MATLAB memory order (y,x,z) with zeros (false).
// Inputs:
// - volumeMatlabOrder: length == nely*nelx*nelz
// - nely, nelx, nelz: original sizes (MATLAB stores as [nely, nelx, nelz])
// - numSolidVoxels: count of nonzero entries in the original volume (same as MATLAB numel(find(...)))
// Returns padded volume and metadata.
struct PaddedVolume {
	std::vector<std::uint8_t> data; // MATLAB order (y,x,z)
	std::uint64_t nely;
	std::uint64_t nelx;
	std::uint64_t nelz;
	int baseLevels;
	int numLevelsForHierarchy;
};

inline PaddedVolume pad_volume_matlab_order(const std::vector<std::uint8_t>& volumeMatlabOrder,
						   std::uint64_t nely,
						   std::uint64_t nelx,
						   std::uint64_t nelz,
						   std::uint64_t numSolidVoxels,
						   std::uint64_t coarsestResolutionControl = 50000,
						   int minLevels = 3) {
	const PaddingResult dims = compute_adjusted_dims(nelx, nely, nelz, numSolidVoxels,
		coarsestResolutionControl, minLevels);

	const std::uint64_t anelx = dims.adjustedNelx;
	const std::uint64_t anely = dims.adjustedNely;
	const std::uint64_t anelz = dims.adjustedNelz;

	std::vector<std::uint8_t> out(anely * anelx * anelz, static_cast<std::uint8_t>(0));

	// Copy original region [0..nely), [0..nelx), [0..nelz) into padded volume.
	for (std::uint64_t iz = 0; iz < nelz; ++iz) {
		for (std::uint64_t ix = 0; ix < nelx; ++ix) {
			// MATLAB linear index: iy + nely*(ix + nelx*iz)
			const std::uint64_t in_plane_offset = nely * (ix + nelx * iz);
			const std::uint64_t out_plane_offset = anely * (ix + anelx * iz);
			const std::uint8_t* src = volumeMatlabOrder.data() + in_plane_offset;
			std::uint8_t* dst = out.data() + out_plane_offset;
			std::copy(src, src + nely, dst);
		}
	}

	return PaddedVolume{std::move(out), anely, anelx, anelz, dims.baseLevels, dims.numLevelsForHierarchy};
}

} // namespace top3d



