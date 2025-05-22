## Camera param estimation:
- Since a lot of the features extracted from the images behind the boat is the water and the person steering it, and these are so numerous, perhaps performing an ititial somewhat bad filtering of these features in the image can improve the matching? We could cut out an equally large part of the images since we always know how much of it will be occupied by the boat.

- Above point, the boat seems to be the issue in both directions with it containing a lot of keypoints. Perhaps a bad semantic segmentation can remove water and boat that is often foung, as well as maybe sun. We can also try to alter the dataset to only use images that actually have good keypoint matches by manually inspecting and then tryint to do a reconstruction of the parameters from these.

## Normalizing the images
After finding the camera parameters try to normalize how the images would look without fish eye distortion to perform segmentation on these instead, hopefully with higher accuracy.

## Segmenting the images
Use the current segmenter, filter the images to find the height of the island in every n (maybe 2-3) pixels and use this as an encoding. 

## Dumb initial image matcher
Search the image database after the most similar skyline encodings using an image, calibrating an error function to allow fairly similar matches.

## Represent the skylines as 2D curves and rotate
With the help of the cardinal directions interpolated from the trajectory found from the gps coordinates, attempt to reconstruct the skylines as they would appear along the cardinal directions. All points in the quadrant along that cardinal direction should be folded out towards the cardinal direction. That way, each image would be encoded as the skyline would look along the four cardinal directions, and this encoding would be done for any new images so the comparison in the database is normalized. (This is an improvement on the initially thought idea, since finding a centered relative point for each island was a difficult task to solve in a reliant repeatable manner)

## Create a more robust encoding that is size invariant 
Since the distance from islands would give a different encoding even when taking a photo of the same island we want to identify relative scales. See måldokument for initial explanation.

## AI similar skyline classifier
Easiest approach to let an MLP classifier find similar skylines is to let it say 0 or 1, if it is a similar enough skyline or not between two images compared at once. This would rely on the skyline encoding above. We could also let the encoding for a skyline be the one spit out from a CNN feature extraction layer. How to do this is not exactly clear yet though, since we want to normalize sections of the skyline along their closest cardinal direction cone. Perhaps we could extract the pixels, rotate as explained earlier. Attain 4 directions of a curve and then let a CNN somehow try to feature extract from it and then use this feature extraction as the encoding. The problem then is that the encodings are not scale invariant. Using CNN features likely works very well if we have a precomputed skyline database simulated from a DEM. 

If we want to investigate the method without DEM most likely we need to create our manual encoding making it size invariant. Ideally we would want the AI to spit out the most similar images in a database. This would be possible if we somehow vector encode each skyline and then try to find the most similar points in this vector space. Qdrant could be used to quickly retrieve the closest points.