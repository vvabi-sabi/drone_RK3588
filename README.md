# UAV navigation by Firefly ROC-RK3588S
Use of various types of neural networks for aircraft navigation.
Different post processes are used for each model. Same as odometry algorithms.
Supports running models in parallel
```
YOLOv5:
1) inference
  photo --> YOLOv5 --> bbox
2) Post Process
  Bounding boxes around objects are applied to the original photo.
3) Localization
  Using the bbox centers of objects, a graph is built.
```

```
ResNet:
1) inference
  photo --> ResNet50 --> candidates (top 5)
2) Post Process
  The index of the first candidate is applied to the original photo.
3) Localization
  Using the Markov chains and nearest neighbors calculates position and direction.
```

```
ResNet:
1) inference
  photo --> UNet --> mask
2) Post Process
  The resulting mask is superimposed on the image.
3) Localization
  ...
```

```
Encoder:
1) inference
  photo --> AE --> vector
2) Post Process
  The vector is compared with all vectors from the database. 
  The index of the first candidate is applied to the original photo.
3) Localization
  vector --> db --> index, scale, angle (localization and orientation)
```