# Locked protocols

Frozen CBraMod, C3/Cz/C4 independent CAR, 0.3–40 Hz, 200 Hz, microvolts/100, one-second patches. PhysioNet uses four seconds and flat2400 features. Cho2017 uses three seconds and flat1800. Source-selection experiments omit PCA to avoid target-dependent fitting.

Leave one subject out. Pool seeds 101/202/303 draw 16 sources. Rank each source by its average BA on the other 15, using a classifier trained only on itself. Select four sources. Random selection seeds 401/502/603. Logistic regression C=0.01, source-only StandardScaler, equal-subject loss contribution.

Each source bank contains 45 labels. Cho banks use seed 20261105 plus sorted subject index, 22/23 class-balanced draws with alternating extra class. PhysioNet uses the locked original 45-trial bank. Selected and random readouts use 180 labels. All-source matched readouts assign 12 labels to four sources and 11 to twelve. Full-bank readouts use 720.

Low-cost pilot seeds 701/802/903 draw six labels per class per source. Ranking uses 192 labels. Four chosen sources each receive 33 additional labels. Total unique source annotations equal 324, and final fitting uses 180. Full-source weighting uses v=tanh((score-pool mean)/0.10), w=1+0.45*(v-mean(v)). Weights average one and lie in (0.1,1.9). Self-BA uses stratified source-internal three-fold OOF scores, seed 17. Permutation controls preserve the transfer-weight distribution.

BA averages the two class recalls, then gives targets equal weight. Uncertainty averages repeated seeds within target before paired target bootstrap. Source pools overlap, so reported intervals are descriptive and conditional on these source populations and trial banks.
