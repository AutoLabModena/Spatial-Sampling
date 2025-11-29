# Spatial-Sampling

This repository provides codes and examples of use of the Spatial Sampling (SS) algorithm. <br>
The SS algorithm takes as input a time serie and provides as output the correspondig geometric path, filtering time discrepancies as speed variations or pauses. It does so by filtering the points on the input trajectory such that the obtained samples are equally-distanced, given a desired distance specified by a parameter $\delta$.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b4627ec6-be7a-4cf0-8958-7bca90fa15e5" width="300" />
  <img src="https://github.com/user-attachments/assets/c45a2497-a88c-4534-83d6-dc0d70b2ee2e" width="240" />
  <img src="https://github.com/user-attachments/assets/0f1cf840-826b-42cf-8296-0f41d75b8369" width="280" />
</p>

From left to right, the figures above depict the working principle of the SS algorithm for the 1D, 2D and 3D cases. The orginal trajectory and the spatially-sampled points are shown, respectively, in black and red. <br>
By imposing the same distance $\delta$ between consecutive samples, the SS algorithm computes a new version of the original trajectory, which consists of its relative geometric path. <br>

The algorithms are part of the paper entitled **Arc-Length-Based Warping for Robot Skill Synthesis from Multiple Demonstrations** by __G.Braglia, D.Tebaldi, A.E.Lazzaretti and L.Biagiotti__, from University of Modena and Reggio Emilia and Federal Technological University of Paranà. The repository also gathers results from the paper, providing additional material to validade the obtained results.<br>
If you find the algorithms useful for your work/research, please cite:
```bibtex
@INPROCEEDINGS{11245947,
  author={Braglia, Giovanni and Tebaldi, Davide and Lazzaretti, André E. and Biagiotti, Luigi},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Arc-Length-Based Warping for Robot Skill Synthesis from Multiple Demonstrations}, 
  year={2025},
  volume={},
  number={},
  pages={8111-8118},
  keywords={Motion planning;Accuracy;Tracking;Heuristic algorithms;Robustness;Trajectory;Recording;Synchronization;Robots;Intelligent robots;Learning from Demonstration;Motion and Path Planning;Datasets for Human Motion},
  doi={10.1109/IROS60139.2025.11245947}}
```

## Folders

- **Approximation_Comparizons_Codes/** : Python codes containing to reproduce the results obtained for the approximation analysis studied in the paper;
- **Metrics/Metrics.pdf**: auxiliary pdf to explain the metrics used in the experimental results of the paper;
- **Panda_CoManipulation_Dataset_Results/** : figure showcasing the results of the approximation comparizons codes computed for each element of the Panda Co-Manipulation Dataset;
- **Spatial_Sampling_Algorithm/** : a Python and MATLAB implementation of the SS algorithm with example scripts.

## Important Resources

This repository is partially based on the following works:

- Calinon, Sylvain. "A tutorial on task-parameterized movement learning and retrieval." Intelligent service robotics 9 (2016): 1-29.
- Shapira Weber, Ron A., et al. "Diffeomorphic temporal alignment nets." Advances in neural information processing systems 32 (2019).
- Tavenard, Romain, et al. "Tslearn, a machine learning toolkit for time series data." Journal of machine learning research 21.118 (2020): 1-6.
- Braglia, Tebaldi, Biagiotti, "Phase-independent Dynamic Movement Primitives with applications to human–robot co-manipulation and time optimal planning", Robotics and Autonomous Systems 194 (2025).
- Panda Co-Manipulation Dataset, github repository: https://github.com/AutoLabModena/Panda-Co-Manipulation-Dataset.git



## Questions & Suggestions
For any doubt, question or suggestion, please feel free to email at:
__giovanni.braglia@unimore.it__
