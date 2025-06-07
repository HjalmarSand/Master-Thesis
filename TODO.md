## Update the paths and create some initial main pipeline method
that creates and repopulates folders so its easy to reconstruct the project for new users.

### Image retrieval continuation
## Test working with the undistorted images
Use the fisheye undistorter, after modifying it to undistort the entire dataset

## Filter out noise
The images are filled with noisy keypoints of the water and the boat that make matching more difficult. Initially we should filter out the sections that we know are always or almost always filled up with boat and water right behind the boat. Then again perform an image matcher. 

## Segmenting the images
Use the current segmenter, filter the images to find the height of the island in every n (maybe 2-3) pixels and use this as an encoding. 

~~Search the image database after the most similar skyline encodings using an image, calibrating an error function to allow fairly similar matches.~~

## Represent the skylines as 2D curves and rotate
~~With the help of the cardinal directions interpolated from the trajectory found from the gps coordinates, attempt to reconstruct the skylines as they would appear along the cardinal directions. All points in the quadrant along that cardinal direction should be folded out towards the cardinal direction. That way, each image would be encoded as the skyline would look along the four cardinal directions, and this encoding would be done for any new images so the comparison in the database is normalized. (This is an improvement on the initially thought idea, since finding a centered relative point for each island was a difficult task to solve in a reliant repeatable manner)~~

## Calculate distance to island
We can calculate the distance to islands based on how much they displace the water? Lets say that the horison cuts the image at the bottom of the 40 percentile of the image (i.e 40% bottom of the image should be water unless there is island there). Now if a certain column of the image consists of less than that the expected 40% water that means that there is a island there and the more of the water that is missing, the closer the image is. Formula to be determined. An error in this, if we are using skyline-based approach, is that we get an error in the discrepancy between the start of the island (i.e the distance to the island) in relation to our current position and the position of the skyline. For example if the trees that create the skyline are at the back of the island in relation to us the rotation calculations of the skyline will introduce error if the rotation calculations include the distance to the skyline as a parameter.

## Create a more robust encoding that is size invariant 
~~Since the distance from islands would give a different encoding even when taking a photo of the same island we want to identify relative scales. See måldokument for initial explanation. We basically have a grid of possible locations we can be in, and we need to find a way to extract information from a limited amount of these to make a prediction in other grid locations. Initially i thought we might want to identify anchor points for islands where we take the angle between the boat and the smallest in angle cardinal direction to this achor point, and rotate the 2D-skyline as we would expect it to be viewed from the cardinal direction. The problem here is, identifying a good anchor point is difficult and most images that are too close to the island might not have the anchor point in them at all. ~~

~~Another idea i thought was that we might not care about an achor point. We might just want to divide the 360 image into 90 degree cones and compare the skyline in these cones to the skylines in the database that has also been processed like this. We dont actually care about any rotation then though, we would just be looking at the cardinal directions and just regularly decide to encode the skyline in this grid point. It wouldnt help us understand how to localize ourselves in other grid points, because depending on how far away we are to the island, more and more of it would be in the 90 degree cone of some cardinal direction. Moving far away enough, the entire system of islands would be in a 90 degree cone in some cardinal direction for example. Do we have a solution? Perhaps we can for a "zoomed out" image compare only a smal middle section and check the likeness. The problem that arises then is, what if the boat is translated in relation to the island? Then what can be in the left corner could previously have been in the middle of a database image. Could there be a way to identify patterns invariant to their location in an image? AI could do this but then it would need to be trained on some patterns perhaps? Perhaps we just ~~

## AI similar skyline classifier
~~Easiest approach to let an MLP classifier find similar skylines is to let it say 0 or 1, if it is a similar enough skyline or not between two images compared at once. This would rely on the skyline encoding above. We could also let the encoding for a skyline be the one spit out from a CNN feature extraction layer. How to do this is not exactly clear yet though, since we want to normalize sections of the skyline along their closest cardinal direction cone. Perhaps we could extract the pixels, rotate as explained earlier. Attain 4 directions of a curve and then let a CNN somehow try to feature extract from it and then use this feature extraction as the encoding. The problem then is that the encodings are not scale invariant. Using CNN features likely works very well if we have a precomputed skyline database simulated from a DEM. ~~

~~If we want to investigate the method without DEM most likely we need to create our manual encoding making it size invariant. Ideally we would want the AI to spit out the most similar images in a database. This would be possible if we somehow vector encode each skyline and then try to find the most similar points in this vector space. Qdrant could be used to quickly retrieve the closest points.~~

## In no particular order
Från panorama normaliserade i väderstreck kolla om kolumner har en ö eller inte, jämför med databas
Behandla datan, kolla om man kan ta bort båt och andra konstanter i bilderna. En effektiv approach för detta hade varit att skapa en binär mask som där de konstanta delarna av bilden sätts till 0 och de relevanta/dynamiska delarna sätts till 1 och därmed sen används.
Starta med en simpel image retrieval.
Skapa en panorama och tilea den. 
Skapa en evaluering. 
Skapa en outline av öarna med hjälp av dem och skapa en polycylinder och jämför. 
kolla på datan, se om det finns överlap i trajectory så att vi kan ta bort bilder i databasen för att på så vis skapa en mer realistisk evaluering
Avvakta med vår novella idé 
Ta fram relevant statistik om vår data, exempelvis 
Skapa ett ramverk för att jämföra olika modeller
