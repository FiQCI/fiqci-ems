# FiQCI EMS

FiQCI Error Mitigation Service (EMS) is a python library for error mitigation as part of the Finnish Quantum Computing Infrastructure.

This python package can be pre-installed on a HPC system or installed by the user. The main goal of the project is to allow users using FiQCI quantum computers to easily add flags to run error mitigation levels. A user can specify mitigation levels 0, 1, 2 or 3.


| **Mitigation Level** | **Mitigation Applied** | **Techniques used** |
|:---:|:---:|:---:|
| 0 | No Mitigation Applied | None |
| 1 | Readout Error Mitigation | M3 |
| 2 | Level 1 +   |  |
| 3 | Level 2 +  |  |

At a basic level the user does not need to do anything and mitigation level 1 is used which means readout error mitigation. At a medium level the user and specify different mitigation levels and at an advanced level the user can configure the error mitigation themselves e.g running Zero noise extrapolation but not gate twirling.
