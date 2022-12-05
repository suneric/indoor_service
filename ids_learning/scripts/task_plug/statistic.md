# training performance

<p align="center">
<img src="https://github.com/suneric/indoor_service/blob/main/ids_learning/scripts/task_plug/data/socket_plug_training_performance.png" width=80% height=80%>
</p>

# Environment Test on DQN(2950) with 50 tries
- env (training)[gray, type B without dash]:success count 44 rate 0.88 average steps in success 7.7727272727272725
- env 1 [type B with dash]:success count 6 rate 0.12 average steps in success 3.8333333333333335
- env 2 [type B with dash, darker]:success count 9 rate 0.18 average steps in success 3.3333333333333335
- env 3 [white, type B without dash]:success count 11 rate 0.22 average steps in success 3.090909090909091
- evn 4 [type B with dash, brisk wall]:success count 3 rate 0.06 average steps in success 2.6666666666666665
- env 5 [yellow, type B without dash, yellow wall]: success count 1 rate 0.02 average steps in success 2.0
- env 6 [red, type B without dash, green wall]:success count 9 rate 0.18 average steps in success 5.0


# Test with trained models after different training episodes
## 100 tries
-DQN
  - 2950 success count 77 rate 0.77 average steps in success 5.1688311688311686
  - 2900 success count 68 rate 0.68 average steps in success 6.676470588235294
  - 2850 success count 66 rate 0.66 average steps in success 6.454545454545454
## 50 tries
- DQN
  - 3000 success count 28 rate 0.56 average steps in success 6.25
  - **2950 success count 41 rate 0.82 average steps in success 5.2926829268292686**
  - **2900 success count 36 rate 0.72 average steps in success 5.638888888888889**
  - **2850 success count 36 rate 0.72 average steps in success 4.916666666666667**
  - 2800 success count 34 rate 0.68 average steps in success 6.0
  - 2750 success count 32 rate 0.64 average steps in success 4.96875
  - 2700 success count 25 rate 0.5 average steps in success 5.88
## 30 tries
- RANDOM, success count 12 rate 0.4 average steps in success 18.08
- DQN
  - 3000, success count 22 rate 0.73 average steps in success 6.18
  - **2950, success count 25 rate 0.83 average steps in success 5.04**
  - 2900, success count 16 rate 0.53 average steps in success 7.125
  - **2850, success count 22 rate 0.73 average steps in success 6.77**
  - 2800, success count 18 rate 0.6 average steps in success 4.17
  - **2750, success count 24 rate 0.8 average steps in success 5.0**
  - 2700, success count 16 rate 0.53 average steps in success 6.18
  - 2650, success count 13 rate 0.43 average steps in success 4.23
  - 2600, success count 14 rate 0.46 average steps in success 5.57
  - 2550, success count 16 rate 0.53 average steps in success 5.5
  - 2500, success count 19 rate 0.63 average steps in success 6.78
  - 2450 success count 18 rate 0.6 average steps in success 8.88888888888889
  - 2400 success count 17 rate 0.5666666666666667 average steps in success 8.058823529411764
  - 2350 success count 23 rate 0.7666666666666667 average steps in success 7.043478260869565
  - 2300 success count 13 rate 0.43333333333333335 average steps in success 7.230769230769231
  - 2250 success count 14 rate 0.4666666666666667 average steps in success 6.214285714285714
  - 2200 success count 20 rate 0.6666666666666666 average steps in success 6.8
  - 2150 success count 16 rate 0.5333333333333333 average steps in success 6.25
  - 2100 success count 23 rate 0.7666666666666667 average steps in success 6.391304347826087
  - 2050 success count 17 rate 0.5666666666666667 average steps in success 5.9411764705882355
  - 2000 success count 19 rate 0.6333333333333333 average steps in success 9.368421052631579
  - 1950 success count 16 rate 0.5333333333333333 average steps in success 6.875
  - 1900 success count 16 rate 0.5333333333333333 average steps in success 7.25
  - 1850 success count 13 rate 0.43333333333333335 average steps in success 7.384615384615385
  - 1800 success count 17 rate 0.5666666666666667 average steps in success 3.764705882352941
  - 1750 success count 12 rate 0.4 average steps in success 12.0
  - 1700 success count 19 rate 0.6333333333333333 average steps in success 10.947368421052632
  - 1650 success count 11 rate 0.36666666666666664 average steps in success 12.545454545454545
  - 1600 success count 19 rate 0.6333333333333333 average steps in success 11.105263157894736
  - 1550 success count 15 rate 0.5 average steps in success 4.6
  - 1500 success count 18 rate 0.6 average steps in success 9.38888888888889
  - 1450 success count 14 rate 0.4666666666666667 average steps in success 8.714285714285714
  - 1400 success count 14 rate 0.4666666666666667 average steps in success 7.5
  - 1350 success count 9 rate 0.3 average steps in success 11.222222222222221
  - 1300 success count 12 rate 0.4 average steps in success 8.916666666666666
  - 1250 success count 15 rate 0.5 average steps in success 8.2
  - 1200 success count 16 rate 0.5333333333333333 average steps in success 6.0
  - 1150 success count 13 rate 0.43333333333333335 average steps in success 11.692307692307692
  - 1100 success count 10 rate 0.3333333333333333 average steps in success 5.2
  - 1050 success count 5 rate 0.16666666666666666 average steps in success 8.2
  - 1000 success count 5 rate 0.16666666666666666 average steps in success 4.6
- PPO
  - 2900 success count 16 rate 0.5333333333333333 average steps in success 11.5625
  - 2850 success count 15 rate 0.5 average steps in success 9.466666666666667
  - 2800 success count 15 rate 0.5 average steps in success 10.666666666666666
  - 2750 success count 17 rate 0.5666666666666667 average steps in success 12.058823529411764
  - 2700 success count 17 rate 0.5666666666666667 average steps in success 12.176470588235293
  - 2650 success count 19 rate 0.6333333333333333 average steps in success 9.368421052631579
  - 2600 success count 15 rate 0.5 average steps in success 10.4
  - 2550 success count 20 rate 0.6666666666666666 average steps in success 12.55
  - 2500 success count 15 rate 0.5 average steps in success 8.8
  - 2450 success count 14 rate 0.4666666666666667 average steps in success 6.214285714285714
  - 2400 success count 14 rate 0.4666666666666667 average steps in success 13.785714285714286
  - 2350 success count 17 rate 0.5666666666666667 average steps in success 15.294117647058824
  - 2300 success count 17 rate 0.5666666666666667 average steps in success 8.294117647058824
  - 2250 success count 16 rate 0.5333333333333333 average steps in success 8.625
  - 2200 success count 19 rate 0.6333333333333333 average steps in success 16.789473684210527
  - 2150 success count 15 rate 0.5 average steps in success 6.266666666666667
  - 2100 success count 20 rate 0.6666666666666666 average steps in success 7.3
  - 2050 success count 13 rate 0.43333333333333335 average steps in success 10.538461538461538
  - 2000 success count 18 rate 0.6 average steps in success 15.666666666666666
  - 1950 success count 19 rate 0.6333333333333333 average steps in success 16.210526315789473
  - 1900 success count 18 rate 0.6 average steps in success 15.222222222222221
  - 1850 success count 15 rate 0.5 average steps in success 11.533333333333333
  - 1800 success count 16 rate 0.5333333333333333 average steps in success 6.8125
  - **1750 success count 21 rate 0.7 average steps in success 11.047619047619047**
  - 1700 success count 17 rate 0.5666666666666667 average steps in success 6.0
  - 1650 success count 19 rate 0.6333333333333333 average steps in success 13.0
  - 1600 success count 14 rate 0.4666666666666667 average steps in success 10.285714285714286
  - 1550 success count 19 rate 0.6333333333333333 average steps in success 9.210526315789474
  - 1500 success count 10 rate 0.3333333333333333 average steps in success 8.3
  - 1450 success count 15 rate 0.5 average steps in success 6.4
  - 1400 success count 20 rate 0.6666666666666666 average steps in success 10.15
  - 1350 success count 17 rate 0.5666666666666667 average steps in success 9.588235294117647
  - 1300 success count 17 rate 0.5666666666666667 average steps in success 15.764705882352942
  - 1250 success count 13 rate 0.43333333333333335 average steps in success 13.307692307692308
  - 1200 success count 17 rate 0.5666666666666667 average steps in success 9.764705882352942
  - 1150 success count 16 rate 0.5333333333333333 average steps in success 11.875
  - 1100 success count 15 rate 0.5 average steps in success 6.6
  - 1050 success count 15 rate 0.5 average steps in success 16.4
  - 1000 success count 18 rate 0.6 average steps in success 5.944444444444445
