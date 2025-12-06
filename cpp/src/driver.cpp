#include "core.hpp"
#include "filter.hpp"
#include "multigrid/multigrid.hpp"
#include "io.hpp"
#include "fea.hpp"
#include "solver.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace top3d {

void TOP3D_XL_GLOBAL(int nely, int nelx, int nelz, float V0, int nLoop, float rMin, int simulation_count) {
	auto tSimulationsStart = std::chrono::steady_clock::now();
	for (int i = 0; i < simulation_count; i++) {
		auto tStartTotal = std::chrono::steady_clock::now();
		
		Problem pb;
		InitialSettings(pb.params);
		double CsolidRef = 0.0;
		
		std::cout << "\n==========================Displaying Inputs==========================\n";
		std::cout << "Simulation: " << i << "\n";
		std::cout << std::fixed << std::setprecision(4);
		std::cout << "..............................................Volume Fraction: " << std::setw(6) << V0 << "\n";
		std::cout << "..........................................Filter Radius: " << std::setw(6) << rMin << " Cells\n";
		std::cout << std::scientific << std::setprecision(4);
		std::cout << "................................................Cell Size: " << std::setw(10) << pb.params.cellSize << "\n";
		std::cout << std::fixed;
		std::cout << ".................................................#CG Iterations: " << std::setw(4) << pb.params.cgMaxIt << "\n";
		std::cout << std::scientific << std::setprecision(4);
		std::cout << "...........................................Youngs Modulus: " << std::setw(10) << pb.params.youngsModulus << "\n";
		std::cout << "....................................Youngs Modulus (Min.): " << std::setw(10) << pb.params.youngsModulusMin << "\n";
		std::cout << "...........................................Poissons Ratio: " << std::setw(10) << pb.params.poissonRatio << "\n";
		std::cout << std::fixed << std::setprecision(6);
		
		auto tStart = std::chrono::steady_clock::now();
		CreateVoxelFEAmodel_Cuboid(pb, nely, nelx, nelz);
		ApplyBoundaryConditions(pb);
		PDEFilter pfFilter = SetupPDEFilter(pb, rMin);
		PDEFilterWorkspace filter_ws;
		// Build MG hierarchy and fixed masks once; reuse across solves
		std::cout << "Building Multigrid Hierarchy..." << std::endl;
		mg::MGPrecondConfig mgcfgStatic_tv; mgcfgStatic_tv.nonDyadic = true; mgcfgStatic_tv.maxLevels = 5; mgcfgStatic_tv.weight = 0.6;
		mg::MGHierarchy H; std::vector<std::vector<uint8_t>> fixedMasks; mg::build_static_once(pb, mgcfgStatic_tv, H, fixedMasks);
		
		auto tModelTime = std::chrono::duration<float>(std::chrono::steady_clock::now() - tStart).count();
		std::cout << "Preparing Voxel-based FEA Model Costs " << std::setw(10) << std::setprecision(1) << tModelTime << "s\n";
		
		// Initialize design
		for (float& x: pb.density) x = static_cast<float>(V0);

		const int ne = pb.mesh.numElements;
		std::vector<float> x = pb.density;
		std::vector<float> xPhys = x;
		std::vector<double> ce(ne, 0.0);
		std::vector<double> eleMod(ne, static_cast<double>(pb.params.youngsModulus));

		// OPTIMIZATION: Reuse U and bFree allocations
		DOFData U;
		U.ux.assign(pb.mesh.numNodes, 0.0);
		U.uy.assign(pb.mesh.numNodes, 0.0);
		U.uz.assign(pb.mesh.numNodes, 0.0);
		std::vector<double> bFree; 
		restrict_to_free(pb, pb.F, bFree);

		std::vector<double> uFreeWarm; // warm-start buffer for PCG
		// PCG workspace reused across solves
		PCGFreeWorkspace pcg_ws;

		// Solve fully solid for reference
		{
			tStart = std::chrono::steady_clock::now();
			std::vector<double> uFree; uFree.assign(bFree.size(), 0.0);
			// Preconditioner: reuse static MG context, per-iter diagonals and SIMP-modulated coarsest
			auto M = mg::make_diagonal_preconditioner_from_static(pb, H, fixedMasks, eleMod, mgcfgStatic_tv);
			int pcgIters = PCG_free(pb, eleMod, bFree, uFree, pb.params.cgTol, pb.params.cgMaxIt, M, pcg_ws);
			scatter_from_free(pb, uFree, U);
			double Csolid = ComputeCompliance(pb, eleMod, U, ce);
			CsolidRef = Csolid;
			// Seed warm start for first optimization iteration
			uFreeWarm = uFree;
			auto tSolveTime = std::chrono::duration<float>(std::chrono::steady_clock::now() - tStart).count();
			std::cout << std::scientific << std::setprecision(6);
			std::cout << "Compliance of Fully Solid Domain: " << std::setw(16) << Csolid << "\n";
			std::cout << std::fixed;
			std::cout << " It.: " << std::setw(4) << 0 << " Solver Time: " << std::setw(4) << std::setprecision(0) << tSolveTime << "s.\n\n";
			std::cout << std::setprecision(6);
		}

		int loop=0;
		float change=1.0f;
		float sharpness = 1.0f;
		// Aggregation for summary
		float sumCG = 0.0f, sumOpt = 0.0f, sumFilter = 0.0f, sumIter = 0.0f;
		float objFirst = 0.0f, objLast = 0.0f;
		
		while (loop < nLoop && change > 1e-4 && sharpness > 0.01) {
			auto tPerIter = std::chrono::steady_clock::now();
			++loop;
			
			// Update modulus via SIMP
			for (int e=0;e<ne;e++) {
				float rho = std::clamp(xPhys[e], 0.0f, 1.0f);
				float powp = static_cast<float>(std::pow(rho, pb.params.simpPenalty));
				eleMod[e] = pb.params.youngsModulusMin + powp * (pb.params.youngsModulus - pb.params.youngsModulusMin);
			}
			
			// Solve KU=F
			auto tSolveStart = std::chrono::steady_clock::now();
			// Reuse outer U, bFree

			// Ensure warm-start vector matches current system size
			if (uFreeWarm.size() != bFree.size()) uFreeWarm.assign(bFree.size(), 0.0f);
			// Reuse static MG context for current SIMP-modified modulus
			auto M = mg::make_diagonal_preconditioner_from_static(pb, H, fixedMasks, eleMod, mgcfgStatic_tv);
			int pcgIters = PCG_free(pb, eleMod, bFree, uFreeWarm, pb.params.cgTol, pb.params.cgMaxIt, M, pcg_ws);
			scatter_from_free(pb, uFreeWarm, U);
			auto tSolveTime = std::chrono::duration<float>(std::chrono::steady_clock::now() - tSolveStart).count();
			
			// Compliance and sensitivities
			auto tOptStart = std::chrono::steady_clock::now();
			double C = ComputeCompliance(pb, eleMod, U, ce);
			// Normalized reporting to mirror MATLAB: ceNorm = ce / CsolidRef; cObj = sum(Ee*ceNorm); Cdisp = cObj*CsolidRef
			double cObjNorm = 0.0;
			if (CsolidRef > 0) {
				for (int e=0;e<ne;e++) cObjNorm += eleMod[e] * (ce[e] / CsolidRef);
			} else {
				for (int e=0;e<ne;e++) cObjNorm += eleMod[e] * ce[e];
			}
			double Cdisp = (CsolidRef > 0 ? cObjNorm * CsolidRef : C);
			std::vector<float> dc(ne, 0.0f);
			for (int e=0;e<ne;e++) {
				float rho = std::clamp(xPhys[e], 0.0f, 1.0f);
				float dEdrho = pb.params.simpPenalty * static_cast<float>(std::pow(rho, pb.params.simpPenalty-1.0f)) * (pb.params.youngsModulus - pb.params.youngsModulusMin);
				float ceNorm = (CsolidRef > 0 ? static_cast<float>(ce[e] / CsolidRef) : static_cast<float>(ce[e]));
				dc[e] = - dEdrho * ceNorm;
			}
			
			// PDE filter on dc (ft=1): filter(x.*dc)./max(1e-3,x)
			auto tFilterStart = std::chrono::steady_clock::now();
			{
				std::vector<float> xdc(ne);
				for (int e=0;e<ne;e++) xdc[e] = x[e]*dc[e];
				std::vector<float> dc_filt; ApplyPDEFilter(pb, pfFilter, xdc, dc_filt, filter_ws);
				for (int e=0;e<ne;e++) dc[e] = dc_filt[e] / std::max(1.0e-3f, x[e]);
			}
			auto tFilterTime = std::chrono::duration<float>(std::chrono::steady_clock::now() - tFilterStart).count();
			
			// OC (Optimality Criteria)
			float l1=0.0f, l2=1e9f;
			float move=0.2f;
			std::vector<float> xnew(ne, 0.0f);
			while ((l2-l1)/(l1+l2) > 1e-6) {
				float lmid = 0.5f*(l1+l2);
				for (int e=0;e<ne;e++) {
					float val = std::sqrt(std::max(1e-30f, -dc[e]/lmid));
					float xe = std::clamp(x[e]*val, x[e]-move, x[e]+move);
					xe = std::clamp(xe, 0.0f, 1.0f);
					xnew[e] = std::max(1.0e-3f, xe);
				}
				float vol = std::accumulate(xnew.begin(), xnew.end(), 0.0f, [](float s, float v){ return s + v; }) / static_cast<float>(ne);
				if (vol - V0 > 0) l1 = lmid; else l2 = lmid;
			}
			change = 0.0f; for (int e=0;e<ne;e++) change = std::max(change, std::abs(xnew[e]-x[e]));
			x.swap(xnew);
			xPhys = x; // no filter in this minimal port

			sharpness = 4.0f * std::accumulate(xPhys.begin(), xPhys.end(), 0.0f, 
				[](float sum, float val) { return sum + val * (1.0f - val); }) / static_cast<float>(ne);
			auto tOptTime = std::chrono::duration<float>(std::chrono::steady_clock::now() - tOptStart).count();
			auto tTotalTime = std::chrono::duration<float>(std::chrono::steady_clock::now() - tPerIter).count();
			
			// Aggregation
			if (loop == 1) objFirst = static_cast<float>(Cdisp);
			objLast = static_cast<float>(Cdisp);
			sumCG += tSolveTime;
			sumOpt += tOptTime;
			sumFilter += tFilterTime;
			sumIter += tTotalTime;
			
			float volFrac = std::accumulate(xPhys.begin(), xPhys.end(), 0.0f) / static_cast<float>(ne);
			float fval = volFrac - V0;
			
			// Print iteration results (matching MATLAB format)
			std::cout << std::scientific << std::setprecision(8);
			std::cout << " It.:" << std::setw(4) << loop 
					<< " Obj.:" << std::setw(16) << Cdisp 
					<< " Vol.:" << std::setw(6) << std::setprecision(4) << volFrac
					<< " Sharp.:" << std::setw(6) << sharpness
					<< " Cons.:" << std::setw(4) << std::setprecision(2) << fval
					<< " Ch.:" << std::setw(4) << change << "\n";
			std::cout << std::fixed << std::setprecision(2);
			std::cout << " It.: " << std::setw(4) << loop << " (Time)... Total per-It.: " << std::setw(8) << std::scientific << tTotalTime << "s;"
					<< " CG: " << std::setw(8) << tSolveTime << "s;"
					<< " Opti.: " << std::setw(8) << tOptTime << "s;"
					<< " Filtering: " << std::setw(8) << tFilterTime << "s.\n";
			std::cout << std::fixed;
		}
		// Exports
		auto tTotalTime = std::chrono::duration<float>(std::chrono::steady_clock::now() - tStartTotal).count();
		std::cout << "\n..........Performing Topology Optimization Costs (in total): " 
				<< std::scientific << std::setprecision(4) << tTotalTime << "s.\n";
		std::cout << std::fixed;
		
		// Determine thread count for summary (tries OpenMP if available, else fallback to 1)
		int thread_count = 1;
		#ifdef _OPENMP
			thread_count = omp_get_max_threads();
		#endif

		// Build summary text
		int origdimsX = pb.mesh.origResX, origdimsY = pb.mesh.origResY, origdimsZ = pb.mesh.origResZ;
		int dimsX = pb.mesh.resX, dimsY = pb.mesh.resY, dimsZ = pb.mesh.resZ;
		float avgPerIter = (loop > 0 ? sumIter / static_cast<float>(loop) : 0.0f);
		float pctCG = (sumIter > 0 ? 100.0f * sumCG / sumIter : 0.0f);
		float pctOpt = (sumIter > 0 ? 100.0f * sumOpt / sumIter : 0.0f);
		float pctFilt = (sumIter > 0 ? 100.0f * sumFilter / sumIter : 0.0f);
		std::ostringstream summary;
		summary << "PARAMETERS\n"
		        << "original dims: x=" << origdimsX << ", y=" << origdimsY << ", z=" << origdimsZ << "\n"
				<< "padded dims: x=" << dimsX << ", y=" << dimsY << ", z=" << dimsZ << "\n"
		        << "v0: " << V0 << "\n\n"
		        << "SOLVING\n"
		        << "iterations: " << loop << "\n"
		        << "cg iterations: " << pb.params.cgMaxIt << "\n\n"
		        << "RESULTS\n"
		        << "compliance solid: " << std::scientific << std::setprecision(6) << CsolidRef << "\n"
		        << "objective first: " << objFirst << "\n"
		        << "objective last: " << objLast << "\n\n"
		        << "TIME\n"
		        << "total solver time: " << sumCG << "\n"
		        << "time per iter: " << std::fixed << std::setprecision(4) << avgPerIter << "\n"
		        << "percentage time spent on cg: " << std::setprecision(2) << pctCG << "%\n"
		        << "percentage time spent on optim: " << pctOpt << "%\n"
		        << "percentage time spent on filtering: " << pctFilt << "%\n\n"
				<< "THREADS\n"
				<< thread_count << "\n\n"
				<< "NOTES\n";
		// Print to stdout so it appears in logs
		std::cout << "\n" << summary.str() << std::endl;
		
		// Persist artifacts with a shared tag
		const std::string tag = generate_unique_tag("GLOBAL");
		// Always write comments markdown
		{
			const std::string commentsDir = out_comments_dir_for_cwd();
			ensure_out_dir(commentsDir);
			std::ofstream md(commentsDir + tag + ".md");
			md << summary.str();
		}
		// Export STL with same tag
		const std::string stlDir = out_stl_dir_for_cwd();
		ensure_out_dir(stlDir);
		export_surface_stl(pb, xPhys, stlDir + tag + ".stl", 0.3f);
		std::cout << "STL file saved to: " << stlDir << tag << ".stl\n";
		std::cout << "\nDone.\n";
	}
	auto finalTime = std::chrono::duration<float>(std::chrono::steady_clock::now() - tSimulationsStart).count();
	std::cout << "\n............Total runtime time for " << simulation_count << " simulations" << std::scientific << std::setprecision(4) << finalTime << "s.\n";
}

} // namespace top3d
