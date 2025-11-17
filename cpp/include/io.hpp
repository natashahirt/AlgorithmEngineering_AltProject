#pragma once
#include "core.hpp"
#include <string>
#include <vector>

namespace top3d {

void ensure_out_dir(const std::string& outDir);
std::string out_dir_for_cwd();
std::string out_stl_dir_for_cwd();
std::string generate_unique_filename(const std::string& mode);
void export_surface_stl(const Problem&, const std::vector<double>& xPhys, const std::string& path, float iso=0.5f);

} // namespace top3d