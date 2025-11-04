#ifndef IO_EXPORT_HPP
#define IO_EXPORT_HPP

#include <vector>
#include <string>
#include <array>

namespace ioexp {

// Write binary STL from triangle soup
bool write_stl_binary(const std::string& path,
					  const std::vector<std::array<float,3>>& vertices,
					  const std::vector<std::array<uint32_t,3>>& faces);

// Minimal NIfTI-1 single-file (.nii) float32 writer
bool write_nifti_float32(const std::string& path,
						  int dimX, int dimY, int dimZ,
						  float sx, float sy, float sz,
						  const std::vector<float>& data);

} // namespace ioexp

#endif


