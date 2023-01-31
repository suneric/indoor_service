# Test Results

## Environment: office room with 4 wall outlets (2 types, NEMA 15R and NEMA 20R)
## 8 receptacles in total, with upper and lower in one wall outlet

## two policies
- raw image of first-look, gray-scaled
- binary image with detected bounding box

## Metrics
### Binary Vision 85.83%, unseen 86.67%, seen 85%
- outlet 1, receptacle upper: 28/30 success (0.93), 8.857 steps average
- outlet 1, receptacle lower: 24/30 success (0.8), 11.583 steps average
- **outlet 2, receptacle upper: 27/30 success (0.9), 14.185 steps average**
- **outlet 2, receptacle lower: 26/30 success (0.86), 10.269 steps average**
- **outlet 3, receptacle upper: 26/30 success (0.86), 8.69 steps average**
- **outlet 3, receptacle lower: 23/30 success (0.76), 11.56 steps average**
- outlet 4, receptacle upper: 26/30 success (0.86), 13.038 steps average
- outlet 4, receptacle lower: 26/30 success (0.86), 13.115 steps average
### Raw Vision 87.91% , unseen 80.83%, seen 95%
- outlet 1, receptacle upper: 29/30 success (0.96),  9.448 steps average
- outlet 1, receptacle lower: 17/30 success (0.56),  11.411 steps average
- **outlet 2, receptacle upper: 29/30 success (0.96),  12.689 steps average**
- **outlet 2, receptacle lower: 28/30 success (0.93),  10.107 steps average**
- **outlet 3, receptacle upper: 29/30 success (0.96),  12.482 steps average**
- **outlet 3, receptacle lower: 28/30 success (0.93),  11.07 steps average**
- outlet 4, receptacle upper:  24/30 success (0.8),  10.708 steps average
- outlet 4, receptacle lower:  27/30 success (0.9),  8.518 steps average
