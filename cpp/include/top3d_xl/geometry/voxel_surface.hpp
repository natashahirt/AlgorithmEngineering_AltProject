#ifndef VOXEL_SURFACE_HPP
#define VOXEL_SURFACE_HPP

#include <vector>
#include <array>
#include <cstdint>

namespace voxsurf {

// Extract surface triangles from a binary/scalar grid using face extraction at threshold.
// vol is [ny][nx][nz] row-major by y,x,z.
inline void extract_faces(const std::vector<float>& vol, int ny, int nx, int nz, float iso,
			   std::vector<std::array<float,3>>& outVerts,
			   std::vector<std::array<uint32_t,3>>& outFaces) {
	outVerts.clear(); outFaces.clear();
	auto at = [&](int y,int x,int z){ return vol[y + ny*x + ny*nx*z]; };
	auto add_face = [&](int axis, float y0, float x0, float z0){
		uint32_t base = static_cast<uint32_t>(outVerts.size());
		std::array<float,3> v0, v1, v2, v3;
		switch(axis){
			case 0: // plane x = const (vary y,z)
				v0 = { x0,     y0,     z0     };
				v1 = { x0,     y0+1.0f, z0     };
				v2 = { x0,     y0+1.0f, z0+1.0f };
				v3 = { x0,     y0,     z0+1.0f };
				break;
			case 1: // plane y = const (vary x,z)
				v0 = { x0,     y0,     z0     };
				v1 = { x0+1.0f, y0,     z0     };
				v2 = { x0+1.0f, y0,     z0+1.0f };
				v3 = { x0,     y0,     z0+1.0f };
				break;
			default: // plane z = const (vary x,y)
				v0 = { x0,     y0,     z0     };
				v1 = { x0+1.0f, y0,     z0     };
				v2 = { x0+1.0f, y0+1.0f, z0     };
				v3 = { x0,     y0+1.0f, z0     };
				break;
		}
		outVerts.push_back(v0); outVerts.push_back(v1); outVerts.push_back(v2); outVerts.push_back(v3);
		outFaces.push_back({base+0, base+1, base+2});
		outFaces.push_back({base+0, base+2, base+3});
	};
	for (int z=0; z<nz; ++z) {
		for (int x=0; x<nx; ++x) {
			for (int y=0; y<ny; ++y) {
				bool inside = at(y,x,z) >= iso;
				// -X face
				if (x==0) {
					if (inside) add_face(0, (float)y, (float)x, (float)z);
				} else {
					bool neigh = at(y,x-1,z) >= iso;
					if (inside && !neigh) add_face(0, (float)y, (float)x, (float)z);
				}
				// +X face
				if (x==nx-1) {
					if (inside) add_face(0, (float)y, (float)(x+1), (float)z);
				} else {
					bool neigh = at(y,x+1,z) >= iso;
					if (inside && !neigh) add_face(0, (float)y, (float)(x+1), (float)z);
				}
				// -Y face
				if (y==0) {
					if (inside) add_face(1, (float)y, (float)x, (float)z);
				} else {
					bool neigh = at(y-1,x,z) >= iso;
					if (inside && !neigh) add_face(1, (float)y, (float)x, (float)z);
				}
				// +Y face
				if (y==ny-1) {
					if (inside) add_face(1, (float)(y+1), (float)x, (float)z);
				} else {
					bool neigh = at(y+1,x,z) >= iso;
					if (inside && !neigh) add_face(1, (float)(y+1), (float)x, (float)z);
				}
				// -Z face
				if (z==0) {
					if (inside) add_face(2, (float)y, (float)x, (float)z);
				} else {
					bool neigh = at(y,x,z-1) >= iso;
					if (inside && !neigh) add_face(2, (float)y, (float)x, (float)z);
				}
				// +Z face
				if (z==nz-1) {
					if (inside) add_face(2, (float)y, (float)x, (float)(z+1));
				} else {
					bool neigh = at(y,x,z+1) >= iso;
					if (inside && !neigh) add_face(2, (float)y, (float)x, (float)(z+1));
				}
			}
		}
	}
}

}

#endif



