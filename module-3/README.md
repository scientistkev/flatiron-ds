# Flatiron Module 3 V2.1 Final Project

## Demonstration of Classification Model 
The goal of this project was to demonstrate our knowledge of supervised classification modeling that we learned in Module 3 of the curriculum. 
Specifically, I chose a dataset on tree cover type in the Roosevelt National Forest in Northern Colorado from the UCI Machine Learning repository.
A link to that repository can be found here: https://archive.ics.uci.edu/ml/datasets/Covertype

## Summary of the Results
The best overall model for the multi-class classification of tree coverage in this project was **Random Forest**. The model proved to have
100% accuracy on the training dataset and 95% accuracy in validation or the test set. Although these results are to be looked at with some
skepticism, it is a good first pass of the classification modeling.

## Contents
This repository contains the following items:
- **README.md**: a comprehensive guide to the project and it's requirements
- **Module 3 Project - Forest Cover Type.ipynb**: the notebook with all of the relevant data science work
- **Blog: An Attempt at Multi-Class Classification**: the blog post describing some tips and tricks I utilized for the project
- **covtype.csv**: the dataset used for the project, in csv format
- **module-3-nontechnical-presentation.pptx**: A powerpoint of the non-technical presentation for the project

## Features and Definitions

Descriptions for all the features in the dataset are as follows: 

1. Elevation - Elevation in meters
2. Aspect - Aspect in degrees azimuth
3. Slope - Slope in degrees
4. Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features
5. Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features
6. Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway
7. Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
8. Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
9. Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
10. Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points
11. Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
12. Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
13. Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation

The wilderness areas are:

1. Rawah Wilderness Area
2. Neota Wilderness Area
3. Comanche Peak Wilderness Area
4. Cache la Poudre Wilderness Area

The soil types are:

1. Cathedral family - Rock outcrop complex, extremely stony.
2. Vanet - Ratake families complex, very stony.
3. Haploborolis - Rock outcrop complex, rubbly.
4. Ratake family - Rock outcrop complex, rubbly.
5. Vanet family - Rock outcrop complex complex, rubbly.
6. Vanet - Wetmore families - Rock outcrop complex, stony.
7. Gothic family.
8. Supervisor - Limber families complex.
9. Troutville family, very stony.
10. Bullwark - Catamount families - Rock outcrop complex, rubbly.
11. Bullwark - Catamount families - Rock land complex, rubbly.
12. Legault family - Rock land complex, stony.
13. Catamount family - Rock land - Bullwark family complex, rubbly.
14. Pachic Argiborolis - Aquolis complex.
15. unspecified in the USFS Soil and ELU Survey.
16. Cryaquolis - Cryoborolis complex.
17. Gateview family - Cryaquolis complex.
18. Rogert family, very stony.
19. Typic Cryaquolis - Borohemists complex.
20. Typic Cryaquepts - Typic Cryaquolls complex.
21. Typic Cryaquolls - Leighcan family, till substratum complex.
22. Leighcan family, till substratum, extremely bouldery.
23. Leighcan family, till substratum - Typic Cryaquolls complex.
24. Leighcan family, extremely stony.
25. Leighcan family, warm, extremely stony.
26. Granile - Catamount families complex, very stony.
27. Leighcan family, warm - Rock outcrop complex, extremely stony.
28. Leighcan family - Rock outcrop complex, extremely stony.
29. Como - Legault families complex, extremely stony.
30. Como family - Rock land - Legault family complex, extremely stony.
31. Leighcan - Catamount families complex, extremely stony.
32. Catamount family - Rock outcrop - Leighcan family complex, extremely stony.
33. Leighcan - Catamount families - Rock outcrop complex, extremely stony.
34. Cryorthents - Rock land complex, extremely stony.
35. Cryumbrepts - Rock outcrop - Cryaquepts complex.
36. Bross family - Rock land - Cryumbrepts complex, extremely stony.
37. Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.
38. Leighcan - Moran families - Cryaquolls complex, extremely stony.
39. Moran family - Cryorthents - Leighcan family complex, extremely stony.
40. Moran family - Cryorthents - Rock land complex, extremely stony.

## Prerequisites
`

## Conclusions

During my project, I found that the three most important factors to determining cover type, when we classify with a Random Forest
Classifier are: 1) Elevation, 2) Distance to Roadways, and 3) Distance to fire points. These makes sense as the distance to fire points 
will ultimately determine the soil type that develops there and the other flora and fauna that can flourish in and around the environ-
ment to support the cover type. In addition, certain tree cover will be better suited to survive near roadways and at certain elevations.

## Future Directions 
- **Class imbalance**: One of the biggest features of this dataset is class imbalance. Specifically, the classes of the target variable `Cover_Type` that are over-represented in the data are cover types 1 and 2, or Spruce/Fir and Lodgepole Pine are the most common target classes in this dataset. Working on some way of reducing the disparity, either by doing a _cost sensitive classifier_ if there's no way of resampling the data, or using some sort of synthetic sampling technique like SMOTE might help.
- **Applying SVM**: Although it appears that Random Forest is a great way to predict what cover type exists, it would be interesting to see how a "one vs one" or even a "one vs all" multi-clas classifier would work in this situation. Thus, applying SVM would be an interesting way of bringing in new classification technology to the project.
- **Feature Engineering**: most of the data contains "vertical" or "horizontal distances to certain points, but we know from experience that the Earth lies on certain curvatures that don't make just pure vertical and horizontal distances accurate. Therefore, going back and making Euclidian distances of each feature would be useful. 
- **Protecting against possible overfitting**: I have a high suspicion that my current model is overfit. Performing cross-validation is highly desirable.

## Author
Kevin McPherson, Flatiron School

$$ Acknowledgements
I would like to thank Abhineet Kulkarni for his guidance, and my cohort for their support
