(* Hamiltonian-Geometric Optimizer Derivation, Benchmark Plots, and GIF

   Run with:
     wolframscript -file wolfram/hamiltonian_geometric_benchmark.wl

   This script assumes the Python benchmark has already created:
     visualizations/pinn_benchmark/pinn_training_history.csv
     visualizations/pinn_benchmark/pinn_optimizer_summary.csv

   It writes Wolfram-generated CSV/PNG/GIF artifacts back into:
     visualizations/pinn_benchmark/
*)

ClearAll["Global`*"];

Print["Hamiltonian-geometric optimizer symbolic check"];

(* General Hamiltonian H(theta,p) = 1/2 p^T Ginv(theta) p + L(theta).
   Keep the metric entries as explicit scalar functions; applying an Array as
   a function can trigger recursive evaluation in Wolfram. *)
theta = Array[\[Theta], 3];
mom = Array[p, 3];
gInv = Table[gInvEntry[i, j][Sequence @@ theta], {i, 3}, {j, 3}];
loss = lossFunction @@ theta;
hamiltonian = 1/2 mom . gInv . mom + loss;

thetaDot = Table[D[hamiltonian, mom[[i]]], {i, Length[mom]}];
pDot = Table[-D[hamiltonian, theta[[i]]], {i, Length[theta]}];

Print["theta_dot = dH/dp:"];
Print[TraditionalForm[thetaDot]];

Print["p_dot = -dH/dtheta:"];
Print[TraditionalForm[pDot]];

Print[
  "This gives p_dot_i = -D_i L - 1/2 p^T (D_i g^-1) p, ",
  "so the optimizer subtracts grad L + F_geo."
];

scriptDirectory = DirectoryName[$InputFileName];
projectRoot = DirectoryName[scriptDirectory];
historyPath = FileNameJoin[{projectRoot, "visualizations", "pinn_benchmark", "pinn_training_history.csv"}];
summaryPath = FileNameJoin[{projectRoot, "visualizations", "pinn_benchmark", "pinn_optimizer_summary.csv"}];
plotPath = FileNameJoin[{projectRoot, "visualizations", "pinn_benchmark", "wolfram_optimizer_convergence.png"}];
cleanHistoryPath = FileNameJoin[{projectRoot, "visualizations", "pinn_benchmark", "wolfram_training_history_clean.csv"}];
cleanSummaryPath = FileNameJoin[{projectRoot, "visualizations", "pinn_benchmark", "wolfram_optimizer_summary_clean.csv"}];
barPlotPath = FileNameJoin[{projectRoot, "visualizations", "pinn_benchmark", "wolfram_final_loss_bar.png"}];
gifPath = FileNameJoin[{projectRoot, "visualizations", "pinn_benchmark", "wolfram_optimizer_convergence.gif"}];

If[!FileExistsQ[historyPath],
  Print["No benchmark CSV found at: ", historyPath];
  Print["Run: python .\\main\\run_pinn_benchmark.py"];
  Quit[0];
];

historyRaw = Import[historyPath, "CSV"];
historyHeader = First[historyRaw];
historyIndex = AssociationThread[historyHeader -> Range[Length[historyHeader]]];
historyRows = <|
     "step" -> ToExpression[#[[historyIndex["step"]]]],
     "optimizer" -> #[[historyIndex["optimizer"]]],
     "loss" -> ToExpression[#[[historyIndex["loss"]]]]
     |> & /@ Rest[historyRaw];

summaryRaw = Import[summaryPath, "CSV"];
summaryHeader = First[summaryRaw];
summaryIndex = AssociationThread[summaryHeader -> Range[Length[summaryHeader]]];
summaryRows = <|
     "optimizer" -> #[[summaryIndex["optimizer"]]],
     "final_loss" -> ToExpression[#[[summaryIndex["final_loss"]]]],
     "pde_loss" -> ToExpression[#[[summaryIndex["pde_loss"]]]],
     "plate_loss" -> ToExpression[#[[summaryIndex["plate_loss"]]]],
     "outer_loss" -> ToExpression[#[[summaryIndex["outer_loss"]]]],
     "gradient_norm" -> ToExpression[#[[summaryIndex["gradient_norm"]]]],
     "spectral_entropy" -> ToExpression[#[[summaryIndex["spectral_entropy"]]]]
     |> & /@ Rest[summaryRaw];

optimizers = DeleteDuplicates[Lookup[historyRows, "optimizer"]];

series = Table[
  With[{rows = Select[historyRows, #["optimizer"] == opt &]},
    opt -> ({#["step"], #["loss"]} & /@ rows)
  ],
  {opt, optimizers}
];

plot = ListLogPlot[
  series[[All, 2]],
  PlotLegends -> Placed[series[[All, 1]], Right],
  Frame -> True,
  FrameLabel -> {"training step", "loss (log scale)"},
  PlotRange -> All,
  ImageSize -> 1100,
  PlotTheme -> "Scientific",
  PlotLabel -> "Optimizer convergence on capacitor PINN loss"
];

Export[plotPath, plot];
Print["Exported Wolfram convergence plot: ", plotPath];

Export[
  cleanHistoryPath,
  Prepend[({#["step"], #["optimizer"], #["loss"]} & /@ historyRows), {"step", "optimizer", "loss"}],
  "CSV"
];
Export[
  cleanSummaryPath,
  Prepend[
    ({#["optimizer"], #["final_loss"], #["pde_loss"], #["plate_loss"], #["outer_loss"],
       #["gradient_norm"], #["spectral_entropy"]} & /@ summaryRows),
    {"optimizer", "final_loss", "pde_loss", "plate_loss", "outer_loss", "gradient_norm", "spectral_entropy"}
  ],
  "CSV"
];
Print["Exported Wolfram clean history CSV: ", cleanHistoryPath];
Print["Exported Wolfram clean summary CSV: ", cleanSummaryPath];

finalLosses = Lookup[summaryRows, "final_loss"];
summaryNames = Lookup[summaryRows, "optimizer"];
best = First@MinimalBy[summaryRows, #["final_loss"] &];
barPlot = BarChart[
  Log10[finalLosses],
  ChartLabels -> Placed[summaryNames, Below],
  ChartStyle -> "Scientific",
  Frame -> True,
  FrameLabel -> {"optimizer", "log10(final loss)"},
  ImageSize -> 1000,
  PlotLabel -> Row[{"Final optimizer losses; best = ", best["optimizer"],
    " (", ScientificForm[best["final_loss"], 4], ")"}]
];
Export[barPlotPath, barPlot];
Print["Exported Wolfram final-loss plot: ", barPlotPath];

maxStep = Max[Lookup[historyRows, "step"]];
frameSteps = DeleteDuplicates[Round@Subdivide[1, maxStep, Min[24, maxStep - 1]]];
frames = Table[
   ListLogPlot[
    (Select[#, First[#] <= step &] & /@ series[[All, 2]]),
    PlotLegends -> Placed[series[[All, 1]], Right],
    Frame -> True,
    FrameLabel -> {"training step", "loss (log scale)"},
    PlotRange -> {{1, maxStep}, All},
    ImageSize -> 720,
    PlotTheme -> "Scientific",
    PlotLabel -> Row[{"Hamiltonian-Geometric optimizer benchmark, step ", step, "/", maxStep}]
    ],
   {step, frameSteps}
   ];
Export[gifPath, frames, "GIF", "DisplayDurations" -> ConstantArray[0.12, Length[frames]]];
Print["Exported Wolfram convergence GIF: ", gifPath];

Print["Final optimizer summary:"];
Print[Dataset[summaryRows]];
