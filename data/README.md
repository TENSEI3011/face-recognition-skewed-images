# Data Directory

Place your datasets here with the following structure:

## Gallery (Enrollment Images)
```
gallery/
  person_001/
    frontal_01.jpg   (High-quality frontal face)
    frontal_02.jpg
  person_002/
    frontal_01.jpg
```

## Probe (Test Images)  
```
probe/
  person_001/
    uav_or_pose_varied_01.jpg
  person_002/
    uav_or_pose_varied_01.jpg
```

## For Quick Testing
You can use the LFW dataset (http://vis-www.cs.umass.edu/lfw/):
- Use 3 images per identity as gallery
- Use remaining images as probe

## For UAV-Specific Testing
Use DroneSURF dataset: https://iab-rubric.org/index.php/dronesurf
Or CFP dataset: http://cfpw.io/ (for pose variation study)
