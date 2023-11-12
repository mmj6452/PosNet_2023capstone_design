# The IDOL Dataset
The IDOL (Inertial Deep Orientation-estimation and Localization) dataset consists of 20+ hours of pedestrian walking IMU data in indoor environments. Data was collected in 3 different buildings from 15 different users of varying body types (not all users are present in each building set). Data collection procedures were approved by an IRB.

Data was collected using a LiDAR-Visual-Inertial SLAM system (Kaarta Stencil) as ground truth, with the an iPhone 8 rigidly attached with the Stencil rig. The rig's ground truth position and orientation are recorded, along with IMU readings from both the Stencil internal XSens IMU (gyroscope, accelerometer) and the iPhone IMU (gyroscope, accelerometer, magnetometer). Because both systems were rigidly mounted to each other, the gyroscopic readings are identical (minus a reference frame transformation). However, as there is an offset in position between the Stencil and phone IMU, accelerometer readings differ slightly due the additional lever arm. The offset between the phone's true position and the Stencil's origin was less than the size of the phone, which we empirically determined (using a Vicon mocap studio) to be within the error margin of the Stencil estimate. All readings are sampled at 100Hz.

### Data subsets
At the root level, the dataset has been divided into 3 buildings. The recorded trajectories for each building dataset are grouped into two subsets: `known` and `unknown`. The dataset for building 1 also contains a subset entitled `train` because cross-subject performance was evaluated in that building. All `known` and `train` set users are from the same common pool of users, which is disjoint from the set of users present in all the `unknown` trajectories. This allows for testing the generalization of networks across users. These were the splits we used for evaluation of results in the paper, but these can be re-split arbitrarily, as we include subject IDs for each trajectory.

Each data subset has a `metadata.json` file, with information about each trajectory file in the subset. This information includes the subject ID for the trajectory and whether or not and when during the run a calibration was performed.

### Trajectory calibration
Calibrations involved rotating the data collection rig along each axis at either the start or end of the trajectory. This allows the magnetometer readings to be callibrated. This was not performed for all trajectories, especially if they were collected back-to-back in the same location, as there is minimal variation in magnetic readings. These can be omitted by truncating a few seconds of data at the relevant parts of the trajectory, although for our evaluation we left these sections in, as there was negligible impact on performance.

Each trajectory starts and ends in roughly the same location in each building. Each also starts and ends with the user quickly jostling the data collection rig in the air, generating a "synchronization spike" in the IMU data. This spike is used to perform time alignment of the iPhone and Stencil data, as we find this approach is slightly more accurate than relying on ntp-synchronized timestamps between devices due to the high sample rate. This synchronizationhas already been performed in the published data files. These artifacts can be omitted using a peak detection algorithm or simply truncating the beginning/end of a trajectory, although we leave them in the dataset because they have negligible impact on results.

### Trajectory global alignment
Trajectories in this dataset were aligned to a global map after being collected, so orientations and positions in one building are globally correct relative to trajectories in other buildings, minus a static positional (x,y) offset (i.e. the trajectories in all buildings begin at (0,0)). The initial pose of trajectories in Building 1 is set as the common origin. In order to recover the data as originally collected, a counter-clockwise rotation in the (x,y) plane must be applied to the position and orientation data in Buildings 2 and 3. This rotation offset is:

Building   | Rotation Offset (radians)
---        | ---
Building 2 | 1.8510
Building 3 | 0.2822


### Reading the data
Each trajectory is stored as a `.feather` file, encoded via Apache Arrow. Python's `pandas` library can be used to read these files as DataFrames using the following: 
```
import pandas as pd
df = pd.read_feather("path/to/feather/file")
```

Each trajectory file contains the following data columns:

Column Name                                                      | Data Description
---                                                              | ---
timestamp                                                        | Time of data point in seconds
orientW, orientX, orientY, orientZ                               | Ground truth orientation from Stencil as a quaternion
processedPosX, processedPosY, processedPosZ                      | Ground truth position from Stencil, smoothed to remove artifacts. This was used as ground truth in our work
iphoneOrientW, iphoneOrientX, iphoneOrientY, iphoneOrientZ       | iPhone CoreMotion API estimate of device orientation as a quaternion
iphoneAccX, iphoneAccY, iphoneAccZ                               | Raw acceleration reported by iPhone IMU
iphoneGyroX, iphoneGyroY, iphoneGyroZ                            | Raw angular velocity reported by iPhone IMU
iphoneMagX, iphoneMagY, iphoneMagZ                               | Magnetometer values reported by iPhone IMU
stencilAccX, stencilAccY, stencilAccZ                            | Raw acceleration reported by Stencil IMU
stencilGyroX, stencilGyroY, stencilGyroZ                         | Raw angular velocity reported by Stencil IMU

Note the iPhone and Stencil IMU raw measurements are not in the same reference frames. The Stencil IMU reference frame and ground truth reference frame are the same. The frames are roughly mapped as Stencil +x -> iPhone -x, Stencil +y -> iPhone -y, Stencil +z -> iPhone +z in right-handed coordinate frames.

