# Test Results
## Environment Common Settings:
- office room with 4 wall outlets (2 types, NEMA 15R and NEMA 20R), Outlet 1 and Outlet 2 are used in training, Outlet 3 and Outlet 4 are not used in training.
- 8 receptacles in total, with upper and lower in one wall outlet

## Policies
- random action
- raw image of first-look, gray-scaled, iteration 6850
- binary image with detected bounding box, iteration 6850

## Metrics of the Environment used for training
### 30 maximum tries
| Policy | **O1U** | **O1L** | **O2U** | **O2L** | O3U | O3L | O4U | O4L | **Train** | Test | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random | 6 | 3 | 5 | 5 | 2 | 1 | 8 | 5 | 16.83% | 13.33% | 14.58% |
| Grey | 29 | 30 | 27 | 28 | 25 | 18 | 27 | 27 | 95% | 80.83% | 87.92% |
| Binary | 27 | 23 | 29 | 26 | 27 | 24 | 27 | 30 | 87.5% | 90% | 88.75% |

### 50 maximum tries
| Policy | **O1U** | **O1L** | **O2U** | **O2L** | O3U | O3L | O4U | O4L | **Train** | Test | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random | 5 | 5 | 3 | 6 | 4 | 7 | 5 | 3 | 9.5% | 9.5% | 9.5% | 3.348 | 17.3 |
| Grey | 48 | 44 | 50 | 47 | 42 | 32 | 47 | 42 | 94.5% | 81.5% | 88% | 4.49 | 10.85 |
| Binary | 42 | 39 | 43 | 46 | 43 | 40 | 41 | 41 | 85% | 82.5% | 83.75% | 4.29 | 11.48 |

## Generalization
### 30 maximum tries, in office_room, grey wall, 2 lights, c=0.5
| Policy | **O1U** | **O1L** | **O2U** | **O2L** | O3U | O3L | O4U | O4L | **Train** | Test | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random | 4 | 5 | 7 | 3 | 4 | 6 | 2 | 3 | 15.83% | 12.5% | 14.17% |
| Grey | 27 | 30 | 29 | 29 | 22 | 18 | 26 | 27 | 95.83% | 77.5% | 86.67% |
| Binary | 29 | 24 | 29 | 25 | 28 | 23 | 26 | 26 | 89.17% | 85.83% | 87.5% |

### 30 maximum tries, in office_room_1, color wall, one light c=0.2
| Policy | **O1U** | **O1L** | **O2U** | **O2L** | O3U | O3L | O4U | O4L | **Train** | Test | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Grey | 17 | 18 | 17 | 11 | 13 | 15 | 16 | 19 | 52.5% | 52.5% | 52.5% |
| Binary | 28 | 26 | 27 | 26 | 27 | 26 | 26 | 25 | 89.17% | 86.67% | 87.92% |

### 30 maximum tries, in office_room_1, brisk wall, darker
