#include "core.hpp"
#include "io.hpp"
#include "geometry_out/export_stl.hpp"
#include "geometry_out/voxel_surface.hpp"
#include <filesystem>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <array>
#include <vector>

namespace top3d {

// forward declarations for exports
// implemented below

// output directory helpers
void ensure_out_dir(const std::string& outDir) {
	std::error_code ec; (void)std::filesystem::create_directories(outDir, ec);
}

std::string out_dir_for_cwd() {
	namespace fs = std::filesystem;
	std::error_code ec;
	fs::path cwd = fs::current_path(ec);
	if (ec) return std::string("./out/");
	fs::path tryLocal = cwd / "out";
	if (fs::exists(tryLocal, ec)) return tryLocal.string() + "/";
	fs::path tryParent = cwd.parent_path() / "out";
	if (fs::exists(tryParent, ec)) return tryParent.string() + "/";
	return (cwd / "out").string() + "/";
}

std::string out_stl_dir_for_cwd() {
	return out_dir_for_cwd() + "stl/";
}

std::string out_log_dir_for_cwd() {
	return out_dir_for_cwd() + "log/";
}

std::string out_comments_dir_for_cwd() {
	return out_dir_for_cwd() + "comments/";
}

std::string generate_unique_tag(const std::string& mode) {
	// Generate datetime string matching bash format: YYYYMMDD_HHMMSS
	auto now = std::chrono::system_clock::now();
	auto time_t = std::chrono::system_clock::to_time_t(now);
	std::tm* tm_buf = std::localtime(&time_t);
	
	std::ostringstream oss;
	oss << std::setfill('0') << std::setw(4) << (tm_buf->tm_year + 1900)
		<< std::setw(2) << (tm_buf->tm_mon + 1)
		<< std::setw(2) << tm_buf->tm_mday << "_"
		<< std::setw(2) << tm_buf->tm_hour
		<< std::setw(2) << tm_buf->tm_min
		<< std::setw(2) << tm_buf->tm_sec;
	std::string datetime = oss.str();
	
	// Get SLURM_JOB_ID from environment if available
	const char* job_id = std::getenv("SLURM_JOB_ID");
	std::string job_id_str = job_id ? std::string(job_id) : "";
	
	// Format: MODE_DATETIME_JOBID.stl (or MODE_DATETIME.stl if no job ID)
	if (!job_id_str.empty()) {
		return mode + "_" + datetime + "_" + job_id_str;
	} else {
		return mode + "_" + datetime;
	}
}

std::string generate_unique_filename(const std::string& mode) {
	return generate_unique_tag(mode) + ".stl";
}

// ===== Export helpers =====
void export_surface_stl(const Problem& pb, const std::vector<float>& xPhys, const std::string& path, float iso) {
	int ny = pb.mesh.resY, nx = pb.mesh.resX, nz = pb.mesh.resZ;
	std::vector<float> vol(ny*nx*nz, 0.0f);
	for (int e=0; e<pb.mesh.numElements; ++e) vol[pb.mesh.eleMapBack[e]] = static_cast<float>(xPhys[e]);
	std::vector<std::array<float,3>> verts; std::vector<std::array<uint32_t,3>> faces;
	voxsurf::extract_faces(vol, ny, nx, nz, iso, verts, faces);
	ioexp::write_stl_binary(path, verts, faces);
}

} // namespace top3d
