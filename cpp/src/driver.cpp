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

namespace top3d {

void TOP3D_XL_GLOBAL(int nely, int nelx, int nelz, double V0, int nLoop, double rMin) {
	auto tStartTotal = std::chrono::steady_clock::now();
	
	Problem pb;
	InitialSettings(pb.params);
	double CsolidRef = 0.0;
	
	std::cout << "\n==========================Displaying Inputs==========================\n";
	std::cout << std::fixed << std::setprecision(4);
	std::cout << "..............................................Volume Fraction: " << std::setw(6) << V0 << "\n";
	std::cout << "..........................................Filter Radius: " << std::setw(6) << rMin << " Cells\n";
	std::cout << std::scientific << std::setprecision(4);
	std::cout << "................................................Cell Size: " << std::setw(10) << pb.params.cellSize << "\n";
	std::cout << std::fixed;
	std::cout << "...............................................#CG Iterations: " << std::setw(4) << pb.params.cgMaxIt << "\n";
	std::cout << std::scientific << std::setprecision(4);
	std::cout << "...........................................Youngs Modulus: " << std::setw(10) << pb.params.youngsModulus << "\n";
	std::cout << "....................................Youngs Modulus (Min.): " << std::setw(10) << pb.params.youngsModulusMin << "\n";
	std::cout << "...........................................Poissons Ratio: " << std::setw(10) << pb.params.poissonRatio << "\n";
	std::cout << std::fixed << std::setprecision(6);
	
	auto tStart = std::chrono::steady_clock::now();
	CreateVoxelFEAmodel_Cuboid(pb, nely, nelx, nelz);
	ApplyBoundaryConditions(pb);
	PDEFilter pfFilter = SetupPDEFilter(pb, rMin);
	// Build MG hierarchy and fixed masks once; reuse across solves
	mg::MGPrecondConfig mgcfgStatic_tv; mgcfgStatic_tv.nonDyadic = true; mgcfgStatic_tv.maxLevels = 5; mgcfgStatic_tv.weight = 0.6;
	mg::MGHierarchy H; std::vector<std::vector<uint8_t>> fixedMasks; mg::build_static_once(pb, mgcfgStatic_tv, H, fixedMasks);
    
	auto tModelTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count();
	std::cout << "Preparing Voxel-based FEA Model Costs " << std::setw(10) << std::setprecision(1) << tModelTime << "s\n";
	
	// Initialize design
	for (double& x: pb.density) x = V0;

	const int ne = pb.mesh.numElements;
	std::vector<double> x = pb.density;
	std::vector<double> xPhys = x;
	std::vector<double> ce(ne, 0.0);
	std::vector<double> eleMod(ne, pb.params.youngsModulus);
	std::vector<double> uFreeWarm; // warm-start buffer for PCG

	// Initialize reused vectors
	std::vector<double> xfull(pb.mesh.numDOFs, 0.0);
	std::vector<double> yfull(pb.mesh.numDOFs, 0.0);
	std::vector<double> pfull(pb.mesh.numDOFs, 0.0);
	std::vector<double> Apfull(pb.mesh.numDOFs, 0.0);
	std::vector<double> freeTmp(pb.freeDofIndex.size(), 0.0);

	// Solve fully solid for reference
	{
		tStart = std::chrono::steady_clock::now();
		std::vector<double> U(pb.mesh.numDOFs, 0.0);
		std::vector<double> bFree; restrict_to_free(pb, pb.F, bFree);
		std::vector<double> uFree; uFree.assign(bFree.size(), 0.0);
		// Preconditioner: reuse static MG context, per-iter diagonals and SIMP-modulated coarsest
		mg::MGPrecondConfig mgcfg; mgcfg.nonDyadic = true; mgcfg.maxLevels = 5; mgcfg.weight = 0.6;
		auto MG = mg::make_diagonal_preconditioner_from_static(pb, H, fixedMasks, eleMod, mgcfg);
        int pcgIters = PCG_free(pb, eleMod, bFree, uFree, pb.params.cgTol, pb.params.cgMaxIt, MG,
			&xfull, &yfull, &pfull, &Apfull, &freeTmp);
		scatter_from_free(pb, uFree, U);
		double Csolid = ComputeCompliance(pb, eleMod, U, ce);
		CsolidRef = Csolid;
		// Seed warm start for first optimization iteration
		uFreeWarm = uFree;
		auto tSolveTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count();
		std::cout << std::scientific << std::setprecision(6);
		std::cout << "Compliance of Fully Solid Domain: " << std::setw(16) << Csolid << "\n";
		std::cout << std::fixed;
		std::cout << " It.: " << std::setw(4) << 0 << " Solver Time: " << std::setw(4) << std::setprecision(0) << tSolveTime << "s.\n\n";
		std::cout << std::setprecision(6);
	}

	int loop=0;
	double change=1.0;
	double sharpness = 1.0;
	
	while (loop < nLoop && change > 1e-4 && sharpness > 0.01) {
		auto tPerIter = std::chrono::steady_clock::now();
		++loop;
		
		// Update modulus via SIMP
		for (int e=0;e<ne;e++) {
			double rho = std::clamp(xPhys[e], 0.0, 1.0);
			eleMod[e] = pb.params.youngsModulusMin + std::pow(rho, pb.params.simpPenalty) * (pb.params.youngsModulus - pb.params.youngsModulusMin);
		}
		
		// Solve KU=F
		auto tSolveStart = std::chrono::steady_clock::now();
		std::vector<double> U(pb.mesh.numDOFs, 0.0);
		std::vector<double> bFree; restrict_to_free(pb, pb.F, bFree);
		// Ensure warm-start vector matches current system size
		if (uFreeWarm.size() != bFree.size()) uFreeWarm.assign(bFree.size(), 0.0);
		// Reuse static MG context for current SIMP-modified modulus
		mg::MGPrecondConfig mgcfg; mgcfg.nonDyadic = true; mgcfg.maxLevels = 5; mgcfg.weight = 0.6;
		auto MG = mg::make_diagonal_preconditioner_from_static(pb, H, fixedMasks, eleMod, mgcfg);
        int pcgIters = PCG_free(pb, eleMod, bFree, uFreeWarm, pb.params.cgTol, pb.params.cgMaxIt, MG,
			&xfull, &yfull, &pfull, &Apfull, &freeTmp);
		scatter_from_free(pb, uFreeWarm, U);
		auto tSolveTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tSolveStart).count();
		
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
        std::vector<double> dc(ne, 0.0);
        for (int e=0;e<ne;e++) {
            double rho = std::clamp(xPhys[e], 0.0, 1.0);
            double dEdrho = pb.params.simpPenalty * std::pow(rho, pb.params.simpPenalty-1.0) * (pb.params.youngsModulus - pb.params.youngsModulusMin);
            double ceNorm = (CsolidRef > 0 ? ce[e] / CsolidRef : ce[e]);
            dc[e] = - dEdrho * ceNorm;
        }
		
		// PDE filter on dc (ft=1): filter(x.*dc)./max(1e-3,x)
		auto tFilterStart = std::chrono::steady_clock::now();
		{
			std::vector<double> xdc(ne);
			for (int e=0;e<ne;e++) xdc[e] = x[e]*dc[e];
			std::vector<double> dc_filt; ApplyPDEFilter(pb, pfFilter, xdc, dc_filt);
			for (int e=0;e<ne;e++) dc[e] = dc_filt[e] / std::max(1e-3, x[e]);
		}
		auto tFilterTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tFilterStart).count();
		
		// OC (Optimality Criteria)
		double l1=0.0, l2=1e9;
		double move=0.2;
		std::vector<double> xnew(ne, 0.0);
		while ((l2-l1)/(l1+l2) > 1e-6) {
			double lmid = 0.5*(l1+l2);
			for (int e=0;e<ne;e++) {
				double val = std::sqrt(std::max(1e-30, -dc[e]/lmid));
				double xe = std::clamp(x[e]*val, x[e]-move, x[e]+move);
				xe = std::clamp(xe, 0.0, 1.0);
				xnew[e] = std::max(1.0e-3, xe);
			}
			double vol = std::accumulate(xnew.begin(), xnew.end(), 0.0) / static_cast<double>(ne);
			if (vol - V0 > 0) l1 = lmid; else l2 = lmid;
		}
		change = 0.0; for (int e=0;e<ne;e++) change = std::max(change, std::abs(xnew[e]-x[e]));
		x.swap(xnew);
		xPhys = x; // no filter in this minimal port

		sharpness = 4.0 * std::accumulate(xPhys.begin(), xPhys.end(), 0.0, 
			[](double sum, double val) { return sum + val * (1.0 - val); }) / static_cast<double>(ne);
		auto tOptTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tOptStart).count();
		auto tTotalTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tPerIter).count();
		
		double volFrac = std::accumulate(xPhys.begin(), xPhys.end(), 0.0) / static_cast<double>(ne);
		double fval = volFrac - V0;
		
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
	auto tTotalTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tStartTotal).count();
	std::cout << "\n..........Performing Topology Optimization Costs (in total): " 
			  << std::scientific << std::setprecision(4) << tTotalTime << "s.\n";
	std::cout << std::fixed;
	
	const std::string stlDir = out_stl_dir_for_cwd();
	ensure_out_dir(stlDir);
	std::string stlFilename = generate_unique_filename("GLOBAL");
	export_surface_stl(pb, xPhys, stlDir + stlFilename, 0.3f);
	std::cout << "STL file saved to: " << stlDir << stlFilename << "\n";
	std::cout << "\nDone.\n";
}

} // namespace top3d
