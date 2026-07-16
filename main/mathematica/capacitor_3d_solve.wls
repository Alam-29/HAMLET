 mmmmmm(* Real 3D finite-difference solve of Laplace's equation for a finite
   rectangular parallel-plate capacitor, run through the Wolfram Engine.
   This is the 3D counterpart to the project's 2D finite-difference
   cross-section solver in src/laplace2d.py (which assumes an infinitely
   long plate, i.e. translation-invariant in the third dimension): here both
   in-plane plate dimensions are finite, so the fringing field is genuinely
   3D near every plate edge and corner, and the capacitance computed is the
   device's actual capacitance (not a per-unit-depth figure).

   Method: each electrode is a single grid layer held at a fixed potential
   (same representation src/laplace2d.py uses in 2D), and the interior is
   relaxed with damped Jacobi iteration on the 6-point 3D stencil, with a
   zero-normal-gradient ("open") outer boundary -- the direct 3D analogue of
   solve_parallel_plate_2d, so the two solvers are methodologically
   consistent and comparable.

   Usage:
     wolframscript -file capacitor_3d_solve.wls \
       <plateLength> <plateWidth> <gap> <voltage> <nx> <ny> <nz> <maxIter> <outDir>

   All lengths in meters. Writes field_grid.csv (a subsampled 3D grid of
   potential and field, independent of the solve resolution) and
   capacitance_summary.json into outDir.
*)

args = Rest[$ScriptCommandLine];
plateLength = If[Length[args] >= 1, ToExpression[args[[1]]], 0.03];
plateWidth  = If[Length[args] >= 2, ToExpression[args[[2]]], 0.02];
gap         = If[Length[args] >= 3, ToExpression[args[[3]]], 0.004];
voltage     = If[Length[args] >= 4, ToExpression[args[[4]]], 1.0];
nx          = If[Length[args] >= 5, ToExpression[args[[5]]], 71];
ny          = If[Length[args] >= 6, ToExpression[args[[6]]], 55];
nz          = If[Length[args] >= 7, ToExpression[args[[7]]], 71];
maxIter     = If[Length[args] >= 8, ToExpression[args[[8]]], 4000];
outDir      = If[Length[args] >= 9, args[[9]], "."];

If[! DirectoryQ[outDir], CreateDirectory[outDir, CreateIntermediateDirectories -> True]];

epsilon0 = 8.8541878128*^-12;

boxScaleXY = 2.2;
boxScaleZ = 9.0;
boxX = plateLength*boxScaleXY;
boxY = plateWidth*boxScaleXY;
boxZ = gap*boxScaleZ;

xs = N@Subdivide[-boxX/2, boxX/2, nx - 1];
ys = N@Subdivide[-boxY/2, boxY/2, ny - 1];
zs = N@Subdivide[-boxZ/2, boxZ/2, nz - 1];
dx = xs[[2]] - xs[[1]];
dy = ys[[2]] - ys[[1]];
dz = zs[[2]] - zs[[1]];

kTop = First@Ordering[Abs[zs - gap/2], 1];
kBot = First@Ordering[Abs[zs + gap/2], 1];
If[kTop == kBot,
  Print["ERROR: gap is too small for this z resolution (nz=", nz, ")"];
  Exit[1]];
electrodeGap = zs[[kTop]] - zs[[kBot]];

plateXBool = Map[Abs[#] <= plateLength/2 &, xs];
plateYBool = Map[Abs[#] <= plateWidth/2 &, ys];
electrodeLength = Max[Pick[xs, plateXBool]] - Min[Pick[xs, plateXBool]];
electrodeWidthActual = Max[Pick[ys, plateYBool]] - Min[Pick[ys, plateYBool]];

isFixedMask = ConstantArray[0.0, {nx, ny, nz}];
fixedVal = ConstantArray[0.0, {nx, ny, nz}];
Do[
  If[plateXBool[[i]] && plateYBool[[j]],
   isFixedMask[[i, j, kTop]] = 1.0; fixedVal[[i, j, kTop]] = voltage/2.0;
   isFixedMask[[i, j, kBot]] = 1.0; fixedVal[[i, j, kBot]] = -voltage/2.0;
   ],
  {i, 1, nx}, {j, 1, ny}];
freeMask = 1.0 - isFixedMask;

pot = fixedVal;

invdx2 = 1.0/dx^2;
invdy2 = 1.0/dy^2;
invdz2 = 1.0/dz^2;
denom = 2.0*(invdx2 + invdy2 + invdz2);
omega = 0.9;
tol = 1.0*^-5*voltage;

Print["Grid: ", nx, " x ", ny, " x ", nz, " = ", nx*ny*nz, " nodes"];
Print["Starting 3D Jacobi relaxation (max ", maxIter, " iterations)..."];
t1 = AbsoluteTime[];
maxDelta = Infinity;
iter = 0;
While[iter < maxIter && maxDelta > tol,
  iter += 1;

  pot[[1, ;; , ;;]] = pot[[2, ;; , ;;]];
  pot[[nx, ;; , ;;]] = pot[[nx - 1, ;; , ;;]];
  pot[[;; , 1, ;;]] = pot[[;; , 2, ;;]];
  pot[[;; , ny, ;;]] = pot[[;; , ny - 1, ;;]];
  pot[[;; , ;; , 1]] = pot[[;; , ;; , 2]];
  pot[[;; , ;; , nz]] = pot[[;; , ;; , nz - 1]];
  pot = fixedVal*isFixedMask + pot*freeMask;

  average = ConstantArray[0.0, {nx, ny, nz}];
  average[[2 ;; -2, 2 ;; -2, 2 ;; -2]] =
   (pot[[3 ;; , 2 ;; -2, 2 ;; -2]] + pot[[1 ;; -3, 2 ;; -2, 2 ;; -2]])*invdx2 +
    (pot[[2 ;; -2, 3 ;; , 2 ;; -2]] + pot[[2 ;; -2, 1 ;; -3, 2 ;; -2]])*invdy2 +
    (pot[[2 ;; -2, 2 ;; -2, 3 ;;]] + pot[[2 ;; -2, 2 ;; -2, 1 ;; -3]])*invdz2;
  average = average/denom;

  newPot = (1.0 - omega)*pot + omega*average;
  newPot = fixedVal*isFixedMask + newPot*freeMask;
  newPot[[1, ;; , ;;]] = pot[[1, ;; , ;;]];
  newPot[[nx, ;; , ;;]] = pot[[nx, ;; , ;;]];
  newPot[[;; , 1, ;;]] = pot[[;; , 1, ;;]];
  newPot[[;; , ny, ;;]] = pot[[;; , ny, ;;]];
  newPot[[;; , ;; , 1]] = pot[[;; , ;; , 1]];
  newPot[[;; , ;; , nz]] = pot[[;; , ;; , nz]];

  maxDelta = Max[Abs[newPot[[2 ;; -2, 2 ;; -2, 2 ;; -2]] - pot[[2 ;; -2, 2 ;; -2, 2 ;; -2]]]];
  pot = newPot;
  If[Mod[iter, 250] == 0, Print["  iter=", iter, "  maxDelta=", maxDelta]];
  ];
Print["Finished after ", iter, " iterations, maxDelta=", maxDelta,
  ", converged=", maxDelta <= tol, ", solve time=", AbsoluteTime[] - t1, " s"];

(* ---- Electric field via central differences (one-sided at box faces). ---- *)
Print["Computing field..."];
ex = ConstantArray[0.0, {nx, ny, nz}];
ey = ConstantArray[0.0, {nx, ny, nz}];
ez = ConstantArray[0.0, {nx, ny, nz}];
ex[[2 ;; -2, ;; , ;;]] = -(pot[[3 ;; , ;; , ;;]] - pot[[1 ;; -3, ;; , ;;]])/(2*dx);
ex[[1, ;; , ;;]] = -(pot[[2, ;; , ;;]] - pot[[1, ;; , ;;]])/dx;
ex[[nx, ;; , ;;]] = -(pot[[nx, ;; , ;;]] - pot[[nx - 1, ;; , ;;]])/dx;
ey[[;; , 2 ;; -2, ;;]] = -(pot[[;; , 3 ;; , ;;]] - pot[[;; , 1 ;; -3, ;;]])/(2*dy);
ey[[;; , 1, ;;]] = -(pot[[;; , 2, ;;]] - pot[[;; , 1, ;;]])/dy;
ey[[;; , ny, ;;]] = -(pot[[;; , ny, ;;]] - pot[[;; , ny - 1, ;;]])/dy;
ez[[;; , ;; , 2 ;; -2]] = -(pot[[;; , ;; , 3 ;;]] - pot[[;; , ;; , 1 ;; -3]])/(2*dz);
ez[[;; , ;; , 1]] = -(pot[[;; , ;; , 2]] - pot[[;; , ;; , 1]])/dz;
ez[[;; , ;; , nz]] = -(pot[[;; , ;; , nz]] - pot[[;; , ;; , nz - 1]])/dz;

energyIntegral = Total[ex^2 + ey^2 + ez^2, All]*dx*dy*dz;
energy = 0.5*epsilon0*energyIntegral;
capacitance = 2.0*energy/voltage^2;
idealCapacitance = epsilon0*plateLength*plateWidth/gap;
fringeRatio = capacitance/idealCapacitance;

Print["C_ideal = ", idealCapacitance, " F"];
Print["C_3d    = ", capacitance, " F"];
Print["fringe ratio = ", fringeRatio];

(* ---- Export a subsampled 3D grid (potential + field) for Python 3D plots.
        The solve itself keeps full resolution for accuracy; the export
        grid is independently capped (regardless of solve resolution)
        because Export-to-CSV of a full high-resolution 3D grid is by far
        the slowest step here, and the plots never need more than a few
        dozen points per axis. A fast manual CSV writer is used instead of
        Export[...,"CSV"], which is disproportionately slow on large
        row-lists. ---- *)
exportCSVFast[path_, rows_] := Module[{stream},
   stream = OpenWrite[path];
   Do[
    WriteString[stream, StringRiffle[ToString[#, InputForm] & /@ row, ","], "\n"],
    {row, rows}];
   Close[stream];
   ];

maxOutNx = 45; maxOutNy = 35; maxOutNz = 45;
outNx = Min[nx, maxOutNx]; outNy = Min[ny, maxOutNy]; outNz = Min[nz, maxOutNz];
xIdxOut = DeleteDuplicates[Round[Subdivide[1, nx, outNx - 1]]];
yIdxOut = DeleteDuplicates[Round[Subdivide[1, ny, outNy - 1]]];
zIdxOut = DeleteDuplicates[Round[Subdivide[1, nz, outNz - 1]]];

Print["Exporting field grid (", Length[xIdxOut], " x ", Length[yIdxOut], " x ",
  Length[zIdxOut], " subsample of the ", nx, "x", ny, "x", nz, " solve grid)..."];
t5 = AbsoluteTime[];
isFixedInt = Round[isFixedMask];
fieldRows = Flatten[Table[
    {xs[[i]], ys[[j]], zs[[k]], pot[[i, j, k]], ex[[i, j, k]], ey[[i, j, k]],
     ez[[i, j, k]], isFixedInt[[i, j, k]]},
    {i, xIdxOut}, {j, yIdxOut}, {k, zIdxOut}], 2];
exportCSVFast[FileNameJoin[{outDir, "field_grid.csv"}], fieldRows];
Print["Field grid exported (", Length[fieldRows], " rows) in ", AbsoluteTime[] - t5, " s"];

summary = <|
   "plate_length_m" -> plateLength,
   "plate_width_m" -> plateWidth,
   "gap_m" -> gap,
   "voltage_V" -> voltage,
   "electrode_length_m" -> electrodeLength,
   "electrode_width_m" -> electrodeWidthActual,
   "electrode_gap_m" -> electrodeGap,
   "box_x_m" -> boxX,
   "box_y_m" -> boxY,
   "box_z_m" -> boxZ,
   "grid_nx" -> Length[xIdxOut],
   "grid_ny" -> Length[yIdxOut],
   "grid_nz" -> Length[zIdxOut],
   "solve_nx" -> nx,
   "solve_ny" -> ny,
   "solve_nz" -> nz,
   "iterations" -> iter,
   "max_delta" -> maxDelta,
   "converged" -> (maxDelta <= tol),
   "ideal_capacitance_F" -> idealCapacitance,
   "capacitance_3d_F" -> capacitance,
   "fringe_ratio_3d" -> fringeRatio,
   "energy_J" -> energy,
   "solver" -> "Wolfram Engine, 3D damped-Jacobi finite-difference relaxation"
   |>;
Export[FileNameJoin[{outDir, "capacitance_summary.json"}], summary, "JSON"];

Print["Done. Wrote outputs to ", outDir];
