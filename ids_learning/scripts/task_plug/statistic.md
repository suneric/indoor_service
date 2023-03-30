# Test Results
## Environment Common Settings:
- office room with 4 wall outlets (2 types, NEMA 15R and NEMA 20R), Outlet 1 and Outlet 2 are used in training, Outlet 3 and Outlet 4 are not used in training.
- 8 receptacles in total, with upper and lower in one wall outlet

## Policies
- random action
- raw image of first-look, gray-scaled, iteration 6850
- binary image with detected bounding box, iteration 6850

## Metrics of the Environment used for training
### 30 maximum tries, in office_room, grey wall, 2 lights, c=0.5
| Policy | **O1U** | **O1L** | **O2U** | **O2L** | O3U | O3L | O4U | O4L | **Train** | Test | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random | 4 | 5 | 7 | 3 | 4 | 6 | 2 | 3 | 15.83% | 12.5% | 14.17% |
| Blind | 19 | 16 | 15 | 13 | 14 | 15 | 10 | 14 | 52.5% | 44.17% | 48.33% |
| Grey | 27 | 30 | 29 | 29 | 22 | 18 | 26 | 27 | 95.83% | 77.5% | 86.67% |
| Binary | 29 | 24 | 29 | 25 | 28 | 23 | 26 | 26 | 89.17% | 85.83% | 87.5% |

### average goal distance and trajectory length (mm)
| Policy | O1U | O1L | O2U | O2L | O3U | O3L | O4U | O4L | Overall Ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random | 2.490 / 19.619 | 2.492 / 22.078 | 2.494 / 27.104 | 4.246 / 30.155 | 2.434 / 24.049| 3.571 / 18.967 | 0.848 / 29.579 | 3.870 / 24.526 | 0.11447 |
| Blind | 9.789 | 14.812 | 12.867 | 11.769 | 17.071 | 12.133 | 7.6 | 9.214 | - |
| Grey | 3.781 / 13.151 | 3.954 / 12.664 | 3.930 / 13.074 | 4.565 / 9.121 | 4.609 / 10.138 | 4.554 / 11.507 | 4.017 / 13.473 | 4.383 / 10.849| 33.793 / 93.997 (0.35959) |
| Binary | 3.847 / 9.819 | 4.089 / 14.308 | 3.957 / 13.856 | 4.646 / 9.966 | 4.817 / 13.052 | 4.352 / 13.495 | 3.965 / 13.642 | 4.411 / 13.202 | 34.084 / 101.34 (0.33633) |

### 50 maximum tries
| Policy | **O1U** | **O1L** | **O2U** | **O2L** | O3U | O3L | O4U | O4L | **Train** | Test | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random | 5 | 5 | 3 | 6 | 4 | 7 | 5 | 3 | 9.5% | 9.5% | 9.5% | 3.348 | 17.3 |
| Grey | 48 | 44 | 50 | 47 | 42 | 32 | 47 | 42 | 94.5% | 81.5% | 88% | 4.49 | 10.85 |
| Binary | 42 | 39 | 43 | 46 | 43 | 40 | 41 | 41 | 85% | 82.5% | 83.75% | 4.29 | 11.48 |

## Generalization
### 30 maximum tries, in office_room_1, color wall, one light c=0.2
| Policy | **O1U** | **O1L** | **O2U** | **O2L** | O3U | O3L | O4U | O4L | **Train** | Test | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Grey | 17 | 18 | 17 | 11 | 13 | 15 | 16 | 19 | 52.5% | 52.5% | 52.5% |
| Binary | 28 | 26 | 27 | 26 | 27 | 26 | 26 | 25 | 89.17% | 86.67% | 87.92% |

### 30 maximum tries, in office_room_2, brisk wall, one light c=1.0
| Policy | **O1U** | **O1L** | **O2U** | **O2L** | O3U | O3L | O4U | O4L | **Train** | Test | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Grey | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 1.67% | 1.67% | 1.67% |
| Binary | 26 | 22 | 30 | 23 | 27 | 24 | 27 | 28 | 84.17% | 88.33% | 86.25% |

### 30 maximum tries, in office_room_3, yellow wall, one light c=0.5, outlet color (blue, flat black, orange, indigo)
| Policy | **O1U** | **O1L** | **O2U** | **O2L** | O3U | O3L | O4U | O4L | **Train** | Test | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Grey | 6 | 1 | 2 | 2 | 0 | 1 | 0 | 0 | 9.17% | 0.83% | 5% |
| Binary | 28 | 25 | 24 | 26 | 28 | 26 | 25 | 27 | 85.83% | 88.33% | 87.08% |


## TEST on new trained policy 9950/10000
### office room
| Policy | **O1U** | **O1L** | **O2U** | **O2L** | O3U | O3L | O4U | O4L | **Train** | Test | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Grey | 29 | 30 | 27 | 28 | 6 | 2 | 24 | 22 | 95% | 45% | 70% |
| Binary | 30 | 27 | 26 | 27 | 27 | 30 | 30 | 25 | 91.67% | 93.33% | 92.5% |

### office room_1
| Policy | **O1U** | **O1L** | **O2U** | **O2L** | O3U | O3L | O4U | O4L | **Train** | Test | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Grey | 14 | 20 | 5 | 16 | 19 | 15 | 21 | 19 | 45.83% | 61.67% | 53.75% |
| Binary | 29 | 24 | 26 | 26 | 23 | 24 | 28 | 26 | 87.5% | 84.17% | 85.83% |

### office room_2
| Policy | **O1U** | **O1L** | **O2U** | **O2L** | O3U | O3L | O4U | O4L | **Train** | Test | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Grey | 16 | 21 | 18 | 13 | 13 | 8 | 8 | 9 | 56.67% | 31.67% | 44.17% |
| Binary | 24 | 24 | 28 | 28 | 27 | 20 | 27 | 23 | 86.67% | 80.83% | 83.75% |

### office room_3
| Policy | **O1U** | **O1L** | **O2U** | **O2L** | O3U | O3L | O4U | O4L | **Train** | Test | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Grey | 8 | 5 | 14 | 12 | 14 | 8 | 19 | 7 | 32.5% | 40% | 36.25% |
| Binary | 29 | 25 | 30 | 30 | 26 | 26 | 28 | 27 | 95% | 89.17% | 92.08% |

## TEST on larger heavier robot office_room
| Policy | **O1U** | **O1L** | **O2U** | **O2L** | O3U | O3L | O4U | O4L | **Train** | Test | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Grey | 26 | 29 | 26 | 30 | 8 | 4 | 18 | 24 | 92.5% | 45%  | 68.75% |
| Binary | 16 | 18 | 22 | 21 | 22 | 17 | 20 | 17 | 68.33% | 59.17% | 63.75% |

| Policy | **O1U** | **O1L** | **O2U** | **O2L** | O3U | O3L | O4U | O4L | **Train** | Test | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Grey | 26 | 30 | 27 | 30 | 12 | 3 | 18 | 21 | 94.17% | 45% | 69.58% |
| Binary | 23 | 17 | 18 | 18 | 19 | 13 | 20 | 19 | 63.33% | 59.17% | 61.25% |
